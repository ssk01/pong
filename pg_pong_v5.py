"""
v5 — Optimized: action repeat (frameskip) + 40x40 input + 4 parallel envs + GPU training.
Built-in profiling: prints timing breakdown every batch update.

Usage:
    python pg_pong_v5.py
    python pg_pong_v5.py --resume
    python pg_pong_v5.py --num-envs 8
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
D = 40 * 40  # 40x40 = 1600, down from 80x80 = 6400

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class PongPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def prepro(I):
    """210x160x3 uint8 frame -> 1600 (40x40) float vector"""
    I = I[35:195]
    I = I[::4, ::4, 0]  # downsample by 4 (was 2) -> 40x40
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def prepro_batch(obs_batch):
    return np.array([prepro(obs) for obs in obs_batch])


def discount_rewards(r):
    discounted = np.zeros_like(r)
    running = 0
    for t in reversed(range(len(r))):
        if r[t] != 0:
            running = 0
        running = running * GAMMA + r[t]
        discounted[t] = running
    return discounted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-envs", type=int, default=4)
    args = parser.parse_args()

    num_envs = args.num_envs
    print(f"Train: {DEVICE}, Inference: CPU, Envs: {num_envs}, Input: 40x40, Frameskip: 8")

    model = PongPolicy().to(DEVICE)
    model_cpu = PongPolicy()
    if args.resume:
        model.load_state_dict(torch.load("save_torch_v5.pt", map_location=DEVICE, weights_only=True))
        print("Loaded checkpoint save_torch_v5.pt")
    model_cpu.load_state_dict(model.state_dict())

    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99)

    # frameskip=8: repeat each action for 8 ALE frames (default v5 is 4)
    envs = gym.make_vec(
        "ALE/Pong-v5",
        num_envs=num_envs,
        vectorization_mode="async",
        frameskip=8,
        repeat_action_probability=0.0,
    )
    observations, infos = envs.reset()

    prev_x = [None] * num_envs
    env_xs = [[] for _ in range(num_envs)]
    env_actions = [[] for _ in range(num_envs)]
    env_rewards = [[] for _ in range(num_envs)]

    all_xs = []
    all_actions = []
    all_discounted = []

    running_reward = None
    episode_number = 0
    batch_loss = 0.0
    t_start = time.time()

    # profiling accumulators
    t_prepro_total = 0
    t_infer_total = 0
    t_env_total = 0
    t_backward_total = 0
    profile_steps = 0

    csv_file = open("train_log_v5.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "running_mean", "loss"])

    while True:
        t0 = time.perf_counter()
        cur_xs = prepro_batch(observations)
        xs = np.zeros((num_envs, D), dtype=np.float32)
        for i in range(num_envs):
            if prev_x[i] is not None:
                xs[i] = cur_xs[i] - prev_x[i]
            prev_x[i] = cur_xs[i]
        t1 = time.perf_counter()
        t_prepro_total += t1 - t0

        with torch.no_grad():
            aprobs = model_cpu(torch.from_numpy(xs)).squeeze(-1).numpy()
        t2 = time.perf_counter()
        t_infer_total += t2 - t1

        randoms = np.random.uniform(size=num_envs)
        actions_int = np.where(randoms < aprobs, 2, 3)
        ys = np.where(randoms < aprobs, 1.0, 0.0)

        for i in range(num_envs):
            env_xs[i].append(xs[i])
            env_actions[i].append(ys[i])

        observations, rewards, terminateds, truncateds, infos = envs.step(actions_int)
        t3 = time.perf_counter()
        t_env_total += t3 - t2
        profile_steps += 1

        for i in range(num_envs):
            env_rewards[i].append(rewards[i])

        dones = terminateds | truncateds
        for i in range(num_envs):
            if not dones[i]:
                continue

            episode_number += 1
            reward_sum = sum(env_rewards[i])

            R = np.array(env_rewards[i], dtype=np.float32)
            disc_R = discount_rewards(R)
            disc_R -= disc_R.mean()
            std = disc_R.std()
            if std > 0:
                disc_R /= std

            all_xs.append(np.array(env_xs[i]))
            all_actions.append(np.array(env_actions[i]))
            all_discounted.append(disc_R)

            env_xs[i] = []
            env_actions[i] = []
            env_rewards[i] = []
            prev_x[i] = None

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print(f"ep {episode_number} (env {i}) reward: {reward_sum:.0f} | running mean: {running_reward:.2f}")

            if episode_number % BATCH_SIZE == 0 and len(all_xs) > 0:
                t4 = time.perf_counter()

                batch_x = torch.from_numpy(np.vstack(all_xs)).to(DEVICE)
                batch_y = torch.from_numpy(np.concatenate(all_actions)).float().to(DEVICE)
                disc_tensor = torch.from_numpy(np.concatenate(all_discounted)).float().to(DEVICE)

                batch_p = model(batch_x).squeeze()
                log_prob = batch_y * torch.log(batch_p) + (1 - batch_y) * torch.log(1 - batch_p)
                loss = -(log_prob * disc_tensor).sum()
                batch_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for p_cpu, p_gpu in zip(model_cpu.parameters(), model.parameters()):
                    p_cpu.data.copy_(p_gpu.data.cpu())

                t5 = time.perf_counter()
                t_backward_total += t5 - t4

                elapsed = time.time() - t_start
                eps_per_sec = BATCH_SIZE / elapsed if elapsed > 0 else 0

                # profiling report
                t_total = t_prepro_total + t_infer_total + t_env_total + t_backward_total
                if t_total > 0:
                    print(f"  [batch] loss: {batch_loss / BATCH_SIZE:.2f} | {eps_per_sec:.2f} ep/s | "
                          f"prepro {t_prepro_total/t_total*100:.0f}% "
                          f"infer {t_infer_total/t_total*100:.0f}% "
                          f"env {t_env_total/t_total*100:.0f}% "
                          f"backward {t_backward_total/t_total*100:.0f}%")

                t_start = time.time()
                t_prepro_total = 0
                t_infer_total = 0
                t_env_total = 0
                t_backward_total = 0
                profile_steps = 0

                all_xs = []
                all_actions = []
                all_discounted = []

            csv_writer.writerow([episode_number, reward_sum, f"{running_reward:.4f}", f"{batch_loss:.4f}"])
            csv_file.flush()

            if episode_number % 100 == 0:
                torch.save(model.state_dict(), "save_torch_v5.pt")
                print("  [checkpoint saved]")


if __name__ == "__main__":
    main()
