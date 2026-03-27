"""
PyTorch v2 — batch forward/backward optimization.
Play with no_grad, then one batched forward + one backward per episode.

Usage:
    python pg_pong_torch_v2.py
    python pg_pong_torch_v2.py --render
    python pg_pong_torch_v2.py --resume
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
DEVICE = torch.device("cpu")


class PongPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


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
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    model = PongPolicy().to(DEVICE)

    if args.resume:
        model.load_state_dict(torch.load("save_torch_v2.pt", map_location=DEVICE, weights_only=True))
        print("Loaded checkpoint save_torch_v2.pt")

    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99)

    env = gym.make("ALE/Pong-v5", render_mode="human" if args.render else None)
    observation, _ = env.reset()
    prev_x = None

    xs = []
    actions = []
    rewards = []

    running_reward = None
    reward_sum = 0
    episode_number = 0
    batch_loss = 0.0
    t_start = time.time()

    csv_file = open("train_log_v2.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "running_mean", "loss"])

    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
        prev_x = cur_x

        with torch.no_grad():
            aprob = model(torch.from_numpy(x).to(DEVICE)).item()

        if np.random.uniform() < aprob:
            action, y = 2, 1.0
        else:
            action, y = 3, 0.0

        xs.append(x)
        actions.append(y)

        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        rewards.append(reward)

        if reward != 0:
            marker = " !!!!!!!!" if reward == 1 else ""
            print(f"ep {episode_number}: game finished, reward: {reward:.0f}{marker}")

        if terminated or truncated:
            episode_number += 1

            R = np.array(rewards, dtype=np.float32)
            discounted_R = discount_rewards(R)
            discounted_R -= discounted_R.mean()
            std = discounted_R.std()
            if std > 0:
                discounted_R /= std

            # batch forward + backward: one graph, one pass
            batch_x = torch.from_numpy(np.vstack(xs)).to(DEVICE)
            batch_y = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
            disc_tensor = torch.from_numpy(discounted_R).to(DEVICE)

            batch_p = model(batch_x).squeeze()
            log_prob = batch_y * torch.log(batch_p) + (1 - batch_y) * torch.log(1 - batch_p)
            loss = -(log_prob * disc_tensor).sum()
            batch_loss += loss.item()

            loss.backward()

            if episode_number % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
                elapsed = time.time() - t_start
                eps_per_sec = BATCH_SIZE / elapsed if elapsed > 0 else 0
                print(f"  [batch update] avg_loss: {batch_loss / BATCH_SIZE:.2f} | speed: {eps_per_sec:.2f} ep/s")
                batch_loss = 0.0
                t_start = time.time()

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            csv_writer.writerow([episode_number, reward_sum, f"{running_reward:.4f}", f"{loss.item():.4f}"])
            csv_file.flush()
            print(f"ep {episode_number} done. reward: {reward_sum:.0f} | running mean: {running_reward:.2f}")

            if episode_number % 100 == 0:
                torch.save(model.state_dict(), "save_torch_v2.pt")
                print(f"  [checkpoint saved]")

            reward_sum = 0
            xs = []
            actions = []
            rewards = []
            observation, _ = env.reset()
            prev_x = None


if __name__ == "__main__":
    main()
