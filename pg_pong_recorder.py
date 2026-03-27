"""
Sample recorder — records full training trajectory for offline experiments.

Saves per-episode data sufficient for:
1. Behavior cloning (sequential replay)
2. Reward-filtered learning (only good episodes)
3. Curriculum learning (ordered by quality)
4. Experience replay (shuffled)
5. PPO-style off-policy reuse (has action probabilities)

Storage format: episodes saved in batches to .npz files
Each step stores: (diff_frame, action, reward, action_prob)
Each episode stores: (episode_id, total_reward, running_mean, num_steps)

Usage:
    python pg_pong_recorder.py
    python pg_pong_recorder.py --resume
"""

import argparse
import csv
import json
import numpy as np
import os
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
SAVE_DIR = "replay_data"


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

    os.makedirs(SAVE_DIR, exist_ok=True)

    model = PongPolicy()
    if args.resume:
        model.load_state_dict(torch.load("save_torch_rec.pt", weights_only=True))
        print("Loaded checkpoint")

    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99)

    env = gym.make("ALE/Pong-v5")
    observation, _ = env.reset()
    prev_x = None

    # per-step buffers
    xs = []
    actions = []
    rewards = []
    aprobs = []

    # episode batch buffer for disk writes
    episode_batch = []
    BATCH_SAVE_SIZE = 100  # save to disk every 100 episodes

    # episode metadata index
    episode_index = []

    running_reward = None
    reward_sum = 0.0
    episode_number = 0
    batch_loss = 0.0
    t_start = time.time()

    # training buffers
    all_xs = []
    all_actions = []
    all_discounted = []

    csv_file = open("train_log_rec.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "running_mean", "loss"])

    print(f"Recording samples to {SAVE_DIR}/")

    while True:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
        prev_x = cur_x

        with torch.no_grad():
            aprob = model(torch.from_numpy(x)).item()

        if np.random.random() < aprob:
            action, y = 2, 1.0
        else:
            action, y = 3, 0.0

        # record everything
        # store diff frame as int8 (-1, 0, 1) to save space
        xs.append(x.astype(np.int8))
        actions.append(y)
        aprobs.append(aprob)

        observation, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward
        rewards.append(reward)

        if reward != 0:
            marker = " !!!!!!!!" if reward == 1 else ""
            print(f"ep {episode_number}: game finished, reward: {reward:.0f}{marker}")

        if terminated or truncated:
            episode_number += 1

            # compute discounted rewards
            R = np.array(rewards, dtype=np.float32)
            disc_R = discount_rewards(R)

            # save episode to batch buffer
            episode_data = {
                "episode_id": episode_number,
                "total_reward": reward_sum,
                "num_steps": len(xs),
                "xs": np.array(xs, dtype=np.int8),          # (T, 6400) int8
                "actions": np.array(actions, dtype=np.float32),  # (T,)
                "rewards": np.array(rewards, dtype=np.float32),  # (T,)
                "aprobs": np.array(aprobs, dtype=np.float32),    # (T,) for importance sampling
                "discounted_rewards": disc_R,                     # (T,)
            }
            episode_batch.append(episode_data)

            # save metadata
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            episode_index.append({
                "episode_id": episode_number,
                "total_reward": float(reward_sum),
                "running_mean": float(running_reward),
                "num_steps": len(xs),
            })

            # flush to disk every BATCH_SAVE_SIZE episodes
            if len(episode_batch) >= BATCH_SAVE_SIZE:
                batch_id = episode_number // BATCH_SAVE_SIZE
                save_path = os.path.join(SAVE_DIR, f"batch_{batch_id:04d}.npz")
                np.savez_compressed(
                    save_path,
                    episode_ids=np.array([e["episode_id"] for e in episode_batch]),
                    total_rewards=np.array([e["total_reward"] for e in episode_batch]),
                    num_steps=np.array([e["num_steps"] for e in episode_batch]),
                    # concatenate all steps, use num_steps to reconstruct episodes
                    all_xs=np.vstack([e["xs"] for e in episode_batch]),
                    all_actions=np.concatenate([e["actions"] for e in episode_batch]),
                    all_rewards=np.concatenate([e["rewards"] for e in episode_batch]),
                    all_aprobs=np.concatenate([e["aprobs"] for e in episode_batch]),
                    all_discounted=np.concatenate([e["discounted_rewards"] for e in episode_batch]),
                )
                print(f"  [saved {save_path}, {len(episode_batch)} episodes]")
                episode_batch = []

                # save index
                with open(os.path.join(SAVE_DIR, "index.json"), "w") as f:
                    json.dump(episode_index, f)

            # --- standard training (same as v2) ---
            disc_R_norm = disc_R.copy()
            disc_R_norm -= disc_R_norm.mean()
            std = disc_R_norm.std()
            if std > 0:
                disc_R_norm /= std

            all_xs.append(np.array(xs, dtype=np.float32))
            all_actions.append(np.array(actions))
            all_discounted.append(disc_R_norm)

            if episode_number % BATCH_SIZE == 0 and len(all_xs) > 0:
                batch_x = torch.from_numpy(np.vstack(all_xs))
                batch_y = torch.from_numpy(np.concatenate(all_actions)).float()
                disc_tensor = torch.from_numpy(np.concatenate(all_discounted).astype(np.float32))

                batch_p = model(batch_x).squeeze()
                log_prob = batch_y * torch.log(batch_p) + (1 - batch_y) * torch.log(1 - batch_p)
                loss = -(log_prob * disc_tensor).sum()
                batch_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                elapsed = time.time() - t_start
                eps_per_sec = BATCH_SIZE / elapsed if elapsed > 0 else 0
                print(f"  [batch] loss: {batch_loss / BATCH_SIZE:.2f} | {eps_per_sec:.2f} ep/s")
                batch_loss = 0.0
                t_start = time.time()

                all_xs = []
                all_actions = []
                all_discounted = []

            print(f"ep {episode_number} done. reward: {reward_sum:.0f} | running mean: {running_reward:.2f}")

            csv_writer.writerow([episode_number, reward_sum, f"{running_reward:.4f}", f"{batch_loss:.4f}"])
            csv_file.flush()

            if episode_number % 100 == 0:
                torch.save(model.state_dict(), "save_torch_rec.pt")
                print("  [checkpoint saved]")

            # reset
            reward_sum = 0.0
            xs = []
            actions = []
            rewards = []
            aprobs = []
            observation, _ = env.reset()
            prev_x = None


if __name__ == "__main__":
    main()
