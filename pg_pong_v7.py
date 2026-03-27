"""
v7 — CNN Policy Gradient.
Conv layers for spatial feature extraction, GPU training.
Same PG algorithm, just a better network architecture.

Usage:
    python pg_pong_v7.py
    python pg_pong_v7.py --resume
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

BATCH_SIZE = 10
LEARNING_RATE = 1e-4
GAMMA = 0.99

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class PongCNN(nn.Module):
    """
    CNN policy network.
    Input: (N, 1, 80, 80) difference frames
    Output: probability of UP
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),  # -> 16 x 19 x 19
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # -> 32 x 8 x 8
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def prepro(I):
    """210x160x3 uint8 frame -> (80, 80) float32"""
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32)


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

    print(f"v7 CNN | Device: {DEVICE}")

    model = PongCNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} (vs FC: 1,281,001)")

    if args.resume:
        model.load_state_dict(torch.load("save_torch_v7.pt", map_location=DEVICE, weights_only=True))
        print("Loaded checkpoint")

    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99)

    # CPU copy for fast per-step inference, compiled for speed
    model_cpu = PongCNN()
    model_cpu.load_state_dict(model.state_dict())
    model_cpu = torch.compile(model_cpu)

    env = gym.make("ALE/Pong-v5")
    observation, _ = env.reset()
    prev_x = None

    xs = []
    actions = []
    rewards = []

    running_reward = None
    reward_sum = 0.0
    episode_number = 0
    batch_loss = 0.0
    t_start = time.time()

    csv_file = open("train_log_v7.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "running_mean", "loss"])

    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros((80, 80), dtype=np.float32)
        prev_x = cur_x

        # CPU inference: (1, 1, 80, 80)
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
            aprob = model_cpu(x_tensor).item()

        if np.random.random() < aprob:
            action, y = 2, 1.0
        else:
            action, y = 3, 0.0

        xs.append(x)
        actions.append(y)

        observation, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward
        rewards.append(reward)

        if reward != 0:
            marker = " !!!!!!!!" if reward == 1 else ""
            print(f"ep {episode_number}: game finished, reward: {reward:.0f}{marker}")

        if terminated or truncated:
            episode_number += 1

            R = np.array(rewards, dtype=np.float32)
            disc_R = discount_rewards(R)
            disc_R -= disc_R.mean()
            std = disc_R.std()
            if std > 0:
                disc_R /= std

            # batch forward + backward on GPU: (N, 1, 80, 80)
            batch_x = torch.from_numpy(np.array(xs)).unsqueeze(1).to(DEVICE)
            batch_y = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
            disc_tensor = torch.from_numpy(disc_R).to(DEVICE)

            batch_p = model(batch_x).squeeze()
            log_prob = batch_y * torch.log(batch_p) + (1 - batch_y) * torch.log(1 - batch_p)
            loss = -(log_prob * disc_tensor).sum()
            batch_loss += loss.item()

            loss.backward()

            if episode_number % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
                model_cpu._orig_mod.load_state_dict(model.state_dict())

                elapsed = time.time() - t_start
                eps_per_sec = BATCH_SIZE / elapsed if elapsed > 0 else 0
                print(f"  [batch] loss: {batch_loss / BATCH_SIZE:.2f} | {eps_per_sec:.2f} ep/s")
                batch_loss = 0.0
                t_start = time.time()

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print(f"ep {episode_number} done. reward: {reward_sum:.0f} | running mean: {running_reward:.2f}")

            csv_writer.writerow([episode_number, reward_sum, f"{running_reward:.4f}", f"{loss.item():.4f}"])
            csv_file.flush()

            if episode_number % 100 == 0:
                torch.save(model.state_dict(), "save_torch_v7.pt")
                print("  [checkpoint saved]")

            reward_sum = 0.0
            xs = []
            actions = []
            rewards = []
            observation, _ = env.reset()
            prev_x = None


if __name__ == "__main__":
    main()
