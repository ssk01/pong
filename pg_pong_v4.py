"""
v4 — 4 parallel environments for faster sampling.
Model is fixed during sampling, multiple envs step simultaneously.
Uses gymnasium.vector.AsyncVectorEnv (each env in its own process).

Usage:
    python pg_pong_v4.py
    python pg_pong_v4.py --resume
    python pg_pong_v4.py --num-envs 8
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
    """Single frame preprocessing."""
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def prepro_batch(obs_batch):
    """Preprocess a batch of observations."""
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
    print(f"Device: {DEVICE}, Parallel envs: {num_envs}")

    model = PongPolicy().to(DEVICE)
    model_cpu = PongPolicy()  # CPU copy for fast inference
    if args.resume:
        model.load_state_dict(torch.load("save_torch_v4.pt", map_location=DEVICE, weights_only=True))
        print("Loaded checkpoint save_torch_v4.pt")
    model_cpu.load_state_dict(model.state_dict())

    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99)

    # create parallel environments
    envs = gym.make_vec("ALE/Pong-v5", num_envs=num_envs, vectorization_mode="async")
    observations, infos = envs.reset()

    # per-env state
    prev_x = [None] * num_envs
    env_xs = [[] for _ in range(num_envs)]
    env_actions = [[] for _ in range(num_envs)]
    env_rewards = [[] for _ in range(num_envs)]

    # collected complete episodes for batch update
    all_xs = []
    all_actions = []
    all_discounted = []

    running_reward = None
    episode_number = 0
    batch_loss = 0.0
    t_start = time.time()

    csv_file = open("train_log_v4.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "running_mean", "loss"])

    while True:
        # preprocess all observations
        cur_xs = prepro_batch(observations)
        xs = np.zeros((num_envs, D), dtype=np.float32)
        for i in range(num_envs):
            if prev_x[i] is not None:
                xs[i] = cur_xs[i] - prev_x[i]
            prev_x[i] = cur_xs[i]

        # batch inference on CPU (fast, no GPU transfer overhead)
        with torch.no_grad():
            aprobs = model_cpu(torch.from_numpy(xs)).squeeze(-1).numpy()

        # sample actions for all envs
        randoms = np.random.uniform(size=num_envs)
        actions_int = np.where(randoms < aprobs, 2, 3)
        ys = np.where(randoms < aprobs, 1.0, 0.0)

        # store per-env data
        for i in range(num_envs):
            env_xs[i].append(xs[i])
            env_actions[i].append(ys[i])

        # step all envs simultaneously
        observations, rewards, terminateds, truncateds, infos = envs.step(actions_int)

        for i in range(num_envs):
            env_rewards[i].append(rewards[i])

        # check for finished episodes
        dones = terminateds | truncateds
        for i in range(num_envs):
            if not dones[i]:
                continue

            episode_number += 1
            reward_sum = sum(env_rewards[i])

            # discount & normalize
            R = np.array(env_rewards[i], dtype=np.float32)
            disc_R = discount_rewards(R)
            disc_R -= disc_R.mean()
            std = disc_R.std()
            if std > 0:
                disc_R /= std

            all_xs.append(np.array(env_xs[i]))
            all_actions.append(np.array(env_actions[i]))
            all_discounted.append(disc_R)

            # reset per-env buffers
            env_xs[i] = []
            env_actions[i] = []
            env_rewards[i] = []
            prev_x[i] = None

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print(f"ep {episode_number} (env {i}) reward: {reward_sum:.0f} | running mean: {running_reward:.2f}")

            # batch update every BATCH_SIZE episodes
            if episode_number % BATCH_SIZE == 0 and len(all_xs) > 0:
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

                # sync weights back to CPU model
                for p_cpu, p_gpu in zip(model_cpu.parameters(), model.parameters()):
                    p_cpu.data.copy_(p_gpu.data.cpu())

                elapsed = time.time() - t_start
                eps_per_sec = BATCH_SIZE / elapsed if elapsed > 0 else 0
                print(f"  [batch update] avg_loss: {batch_loss / BATCH_SIZE:.2f} | speed: {eps_per_sec:.2f} ep/s")
                t_start = time.time()

                all_xs = []
                all_actions = []
                all_discounted = []

            csv_writer.writerow([episode_number, reward_sum, f"{running_reward:.4f}", "0"])
            csv_file.flush()

            if episode_number % 100 == 0:
                torch.save(model.state_dict(), "save_torch_v4.pt")
                print("  [checkpoint saved]")


if __name__ == "__main__":
    main()
