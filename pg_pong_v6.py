"""
v6 — Maximum throughput: eliminate all Python overhead.
- NumPy inference (no PyTorch tensor creation per step)
- Pre-allocated arrays (no allocation in hot loop)
- Single process (no IPC overhead)
- PyTorch only for batch backward
- Built-in profiling

Usage:
    python pg_pong_v6.py
    python pg_pong_v6.py --resume
"""

import argparse
import csv
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

H = 200
BATCH_SIZE = 10
LEARNING_RATE = 1e-4
GAMMA = 0.99
D = 80 * 80


class PongPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def discount_rewards(r):
    discounted = np.zeros_like(r)
    running = 0.0
    for t in reversed(range(len(r))):
        if r[t] != 0:
            running = 0.0
        running = running * GAMMA + r[t]
        discounted[t] = running
    return discounted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = PongPolicy()
    if args.resume:
        model.load_state_dict(torch.load("save_torch_v6.pt", weights_only=True))
        print("Loaded checkpoint")

    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99)

    # extract numpy weights for fast inference
    W1 = model.fc1.weight.data.numpy()
    b1 = model.fc1.bias.data.numpy()
    W2 = model.fc2.weight.data.numpy().ravel()
    b2 = model.fc2.bias.data.numpy().item()

    def sync_weights():
        nonlocal W1, b1, W2, b2
        W1 = model.fc1.weight.data.numpy()
        b1 = model.fc1.bias.data.numpy()
        W2 = model.fc2.weight.data.numpy().ravel()
        b2 = model.fc2.bias.data.numpy().item()

    env = gym.make("ALE/Pong-v5")
    observation, _ = env.reset()

    # pre-allocate
    prev_x = np.zeros(D, dtype=np.float32)
    cur_x = np.zeros(D, dtype=np.float32)
    x = np.zeros(D, dtype=np.float32)
    h = np.zeros(H, dtype=np.float32)
    has_prev = False

    # episode buffers - pre-allocate max size (one episode ~2000 steps max)
    MAX_STEPS = 5000
    xs_buf = np.zeros((MAX_STEPS, D), dtype=np.float32)
    ys_buf = np.zeros(MAX_STEPS, dtype=np.float32)
    rs_buf = np.zeros(MAX_STEPS, dtype=np.float32)
    step_idx = 0

    running_reward = None
    reward_sum = 0.0
    episode_number = 0
    batch_loss = 0.0
    t_start = time.time()

    all_xs = []
    all_ys = []
    all_disc = []

    t_env = 0.0
    t_prepro = 0.0
    t_infer = 0.0
    t_backward = 0.0

    csv_file = open("train_log_v6.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "running_mean", "loss"])

    print(f"v6: NumPy inference, single process, pre-allocated arrays")

    while True:
        # === prepro inline (avoid function call overhead) ===
        t0 = time.perf_counter()
        I = observation[35:195:2, ::2, 0]  # crop + downsample in one slice
        I = I.astype(np.float32)
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        np.copyto(cur_x, I.ravel())
        if has_prev:
            np.subtract(cur_x, prev_x, out=x)
        else:
            np.copyto(x, cur_x)
            has_prev = True
        np.copyto(prev_x, cur_x)
        t1 = time.perf_counter()
        t_prepro += t1 - t0

        # === numpy inference (no PyTorch, no tensor creation) ===
        np.dot(W1, x, out=h)
        np.add(h, b1, out=h)
        np.maximum(h, 0, out=h)  # ReLU in-place
        logp = np.dot(W2, h) + b2
        aprob = 1.0 / (1.0 + np.exp(-logp))
        t2 = time.perf_counter()
        t_infer += t2 - t1

        # sample action
        if np.random.random() < aprob:
            action = 2
            y = 1.0
        else:
            action = 3
            y = 0.0

        # store in pre-allocated buffer
        xs_buf[step_idx] = x
        ys_buf[step_idx] = y

        # === env.step ===
        observation, reward, terminated, truncated, _ = env.step(action)
        t3 = time.perf_counter()
        t_env += t3 - t2

        rs_buf[step_idx] = reward
        reward_sum += reward
        step_idx += 1

        if reward != 0:
            marker = " !!!!!!!!" if reward == 1 else ""
            print(f"ep {episode_number}: game finished, reward: {reward:.0f}{marker}")

        if terminated or truncated:
            episode_number += 1

            # slice valid data from pre-allocated buffers
            ep_x = xs_buf[:step_idx].copy()
            ep_y = ys_buf[:step_idx].copy()
            ep_r = rs_buf[:step_idx].copy()

            disc_r = discount_rewards(ep_r)
            disc_r -= disc_r.mean()
            std = disc_r.std()
            if std > 0:
                disc_r /= std

            all_xs.append(ep_x)
            all_ys.append(ep_y)
            all_disc.append(disc_r)

            # reset
            step_idx = 0
            has_prev = False

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print(f"ep {episode_number} done. reward: {reward_sum:.0f} | running mean: {running_reward:.2f}")

            if episode_number % BATCH_SIZE == 0 and len(all_xs) > 0:
                t4 = time.perf_counter()

                batch_x = torch.from_numpy(np.vstack(all_xs))
                batch_y = torch.from_numpy(np.concatenate(all_ys))
                disc_tensor = torch.from_numpy(np.concatenate(all_disc).astype(np.float32))

                batch_p = model(batch_x).squeeze()
                log_prob = batch_y * torch.log(batch_p) + (1 - batch_y) * torch.log(1 - batch_p)
                loss = -(log_prob * disc_tensor).sum()
                batch_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sync_weights()

                t5 = time.perf_counter()
                t_backward += t5 - t4

                elapsed = time.time() - t_start
                eps_per_sec = BATCH_SIZE / elapsed if elapsed > 0 else 0
                t_total = t_prepro + t_infer + t_env + t_backward
                if t_total > 0:
                    print(f"  [batch] loss: {batch_loss / BATCH_SIZE:.2f} | {eps_per_sec:.2f} ep/s | "
                          f"prepro {t_prepro/t_total*100:.0f}% "
                          f"infer {t_infer/t_total*100:.0f}% "
                          f"env {t_env/t_total*100:.0f}% "
                          f"backward {t_backward/t_total*100:.0f}%")

                t_start = time.time()
                t_prepro = 0.0
                t_infer = 0.0
                t_env = 0.0
                t_backward = 0.0

                all_xs = []
                all_ys = []
                all_disc = []

            csv_writer.writerow([episode_number, reward_sum, f"{running_reward:.4f}", f"{batch_loss:.4f}"])
            csv_file.flush()

            if episode_number % 100 == 0:
                torch.save(model.state_dict(), "save_torch_v6.pt")
                print("  [checkpoint saved]")

            reward_sum = 0.0
            observation, _ = env.reset()


if __name__ == "__main__":
    main()
