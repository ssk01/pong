"""
Shared utilities for offline RL experiments (exp1-4).
Loads replay_data/ samples and provides different iteration strategies.
"""

import json
import numpy as np
import os
import torch
import torch.nn as nn

D = 80 * 80
H = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 2048  # steps per gradient update

REPLAY_DIR = os.path.join(os.path.dirname(__file__), "..", "replay_data")


class PongPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def load_all_episodes():
    """Load all episodes from replay_data/ into a list of dicts."""
    with open(os.path.join(REPLAY_DIR, "index.json")) as f:
        index = json.load(f)

    # find all batch files
    batch_files = sorted([
        f for f in os.listdir(REPLAY_DIR) if f.startswith("batch_") and f.endswith(".npz")
    ])

    episodes = []
    for bf in batch_files:
        data = np.load(os.path.join(REPLAY_DIR, bf))
        ep_ids = data["episode_ids"]
        rewards = data["total_rewards"]
        num_steps = data["num_steps"]
        all_xs = data["all_xs"]
        all_actions = data["all_actions"]
        all_rewards = data["all_rewards"]
        all_discounted = data["all_discounted"]
        all_aprobs = data["all_aprobs"]

        offset = 0
        for i in range(len(ep_ids)):
            n = num_steps[i]
            episodes.append({
                "episode_id": int(ep_ids[i]),
                "total_reward": float(rewards[i]),
                "xs": all_xs[offset:offset+n].astype(np.float32),
                "actions": all_actions[offset:offset+n],
                "rewards": all_rewards[offset:offset+n],
                "discounted_rewards": all_discounted[offset:offset+n],
                "aprobs": all_aprobs[offset:offset+n],
            })
            offset += n

    print(f"Loaded {len(episodes)} episodes from {len(batch_files)} batch files")
    return episodes


def train_on_episodes(episodes, model, optimizer, csv_writer, max_epochs=5, label=""):
    """Train model on a list of episodes, one epoch = one pass through all episodes."""
    total_steps = sum(len(ep["xs"]) for ep in episodes)
    print(f"[{label}] {len(episodes)} episodes, {total_steps:,} total steps, {max_epochs} epochs")

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for ep in episodes:
            xs = torch.from_numpy(ep["xs"])
            actions = torch.from_numpy(ep["actions"]).float()
            disc_r = ep["discounted_rewards"].copy()
            disc_r -= disc_r.mean()
            std = disc_r.std()
            if std > 0:
                disc_r /= std
            disc_tensor = torch.from_numpy(disc_r.astype(np.float32))

            # batch forward + backward
            probs = model(xs).squeeze()
            log_prob = actions * torch.log(probs + 1e-8) + (1 - actions) * torch.log(1 - probs + 1e-8)
            loss = -(log_prob * disc_tensor).sum()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += len(ep["xs"])

        avg_loss = epoch_loss / len(episodes)
        print(f"  epoch {epoch+1}/{max_epochs} | avg_loss: {avg_loss:.2f} | steps: {epoch_steps:,}")
        csv_writer.writerow([epoch+1, avg_loss, epoch_steps])

    return model


def evaluate(model, num_episodes=50):
    """Evaluate trained model on live Pong."""
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)

    env = gym.make("ALE/Pong-v5")
    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        prev_x = None
        total_reward = 0

        while True:
            cur_x = obs[35:195:2, ::2, 0].astype(np.float32)
            cur_x[cur_x == 144] = 0; cur_x[cur_x == 109] = 0; cur_x[cur_x != 0] = 1
            cur_x = cur_x.ravel()
            x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
            prev_x = cur_x

            with torch.no_grad():
                p = model(torch.from_numpy(x)).item()
            action = 2 if np.random.random() < p else 3
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            if term or trunc:
                break

        rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            print(f"  eval {ep+1}/{num_episodes} | mean: {np.mean(rewards):.2f}")

    env.close()
    mean_r = np.mean(rewards)
    print(f"  EVAL RESULT: mean={mean_r:.2f}, std={np.std(rewards):.2f}")
    return mean_r
