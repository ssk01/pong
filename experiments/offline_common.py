"""
Shared utilities for offline RL experiments (exp1-4).
Loads replay_data/ samples in streaming fashion to avoid OOM.
"""

import json
import numpy as np
import os
import torch
import torch.nn as nn

D = 80 * 80
H = 200
LEARNING_RATE = 1e-4

REPLAY_DIR = os.path.join(os.path.dirname(__file__), "..", "replay_data")


class PongPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def get_batch_files(last_n=50):
    """Return sorted list of batch file paths. Only use last_n files to keep it manageable."""
    all_files = sorted([
        os.path.join(REPLAY_DIR, f)
        for f in os.listdir(REPLAY_DIR)
        if f.startswith("batch_") and f.endswith(".npz")
    ])
    if last_n and last_n < len(all_files):
        print(f"Using last {last_n} of {len(all_files)} batch files")
        return all_files[-last_n:]
    return all_files


def load_index():
    """Load episode index (lightweight metadata only)."""
    with open(os.path.join(REPLAY_DIR, "index.json")) as f:
        return json.load(f)


def iter_episodes_from_batch(batch_path):
    """Yield episodes one by one from a single batch file. Memory efficient."""
    data = np.load(batch_path)
    ep_ids = data["episode_ids"]
    rewards = data["total_rewards"]
    num_steps = data["num_steps"]
    all_xs = data["all_xs"]
    all_actions = data["all_actions"]
    all_discounted = data["all_discounted"]

    offset = 0
    for i in range(len(ep_ids)):
        n = num_steps[i]
        yield {
            "episode_id": int(ep_ids[i]),
            "total_reward": float(rewards[i]),
            "xs": all_xs[offset:offset+n].astype(np.float32),
            "actions": all_actions[offset:offset+n],
            "discounted_rewards": all_discounted[offset:offset+n],
        }
        offset += n


def train_on_batch_file(batch_path, model, optimizer, filter_fn=None):
    """Train on one batch file (100 episodes). Batches all steps for one forward+backward."""
    # collect all steps from this batch file
    all_xs, all_actions, all_disc = [], [], []
    ep_count = 0

    for ep in iter_episodes_from_batch(batch_path):
        if filter_fn and not filter_fn(ep):
            continue
        disc_r = ep["discounted_rewards"].copy()
        disc_r -= disc_r.mean()
        std = disc_r.std()
        if std > 0:
            disc_r /= std
        all_xs.append(ep["xs"])
        all_actions.append(ep["actions"])
        all_disc.append(disc_r)
        ep_count += 1

    if ep_count == 0:
        return 0.0, 0

    # one batched forward + backward for entire batch file
    xs = torch.from_numpy(np.vstack(all_xs))
    actions = torch.from_numpy(np.concatenate(all_actions)).float()
    disc_tensor = torch.from_numpy(np.concatenate(all_disc).astype(np.float32))

    probs = model(xs).squeeze()
    log_prob = actions * torch.log(probs + 1e-8) + (1 - actions) * torch.log(1 - probs + 1e-8)
    loss = -(log_prob * disc_tensor).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return loss.item(), ep_count


def train_streaming(model, optimizer, csv_writer, max_epochs=3, label="",
                    batch_order_fn=None, filter_fn=None):
    """Train model by streaming through batch files. Never loads all data at once."""
    batch_files = get_batch_files()
    print(f"[{label}] {len(batch_files)} batch files, {max_epochs} epochs")

    for epoch in range(max_epochs):
        files = list(batch_files)
        if batch_order_fn:
            files = batch_order_fn(files, epoch)

        epoch_loss = 0.0
        epoch_eps = 0

        for bf in files:
            avg_loss, ep_count = train_on_batch_file(bf, model, optimizer, filter_fn)
            epoch_loss += avg_loss * ep_count
            epoch_eps += ep_count

        avg = epoch_loss / max(epoch_eps, 1)
        print(f"  epoch {epoch+1}/{max_epochs} | avg_loss: {avg:.2f} | episodes: {epoch_eps}")
        csv_writer.writerow([epoch+1, avg, epoch_eps])

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
