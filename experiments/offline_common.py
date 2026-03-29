"""
Shared utilities for offline RL experiments (exp1-4).
Loads replay_data/ samples in streaming fashion to avoid OOM.
"""

import gc
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


def get_batch_files(last_n=10):
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
    """Yield episodes one by one from a single batch file. Uses mmap to avoid loading all into RAM."""
    data = np.load(batch_path, mmap_mode='r')
    ep_ids = np.array(data["episode_ids"])
    rewards = np.array(data["total_rewards"])
    num_steps = np.array(data["num_steps"])

    offset = 0
    for i in range(len(ep_ids)):
        n = int(num_steps[i])
        # copy slices to regular arrays (mmap slices are read-only)
        yield {
            "episode_id": int(ep_ids[i]),
            "total_reward": float(rewards[i]),
            "xs": np.array(data["all_xs"][offset:offset+n], dtype=np.float32),
            "actions": np.array(data["all_actions"][offset:offset+n]),
            "discounted_rewards": np.array(data["all_discounted"][offset:offset+n]),
        }
        offset += n
    del data
    gc.collect()


MINI_BATCH = 2048


def train_on_batch_file(batch_path, model, optimizer, filter_fn=None):
    """Train on one batch file using mini-batches of raw steps. Fast."""
    data = np.load(batch_path, mmap_mode='r')
    num_steps = np.array(data["num_steps"])
    total_rewards = np.array(data["total_rewards"])

    # figure out which episodes to include
    offsets = np.concatenate([[0], np.cumsum(num_steps)])
    include_mask = np.ones(len(num_steps), dtype=bool)
    if filter_fn:
        for i in range(len(num_steps)):
            if not filter_fn({"total_reward": float(total_rewards[i])}):
                include_mask[i] = False

    # collect indices of included steps
    step_indices = []
    disc_values = []
    for i in range(len(num_steps)):
        if not include_mask[i]:
            continue
        start, end = int(offsets[i]), int(offsets[i+1])
        step_indices.append(np.arange(start, end))
        # compute discounted rewards for this episode
        rewards = np.array(data["all_rewards"][start:end])
        disc_r = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            if rewards[t] != 0:
                running = 0.0
            running = running * 0.99 + rewards[t]
            disc_r[t] = running
        disc_r -= disc_r.mean()
        std = disc_r.std()
        if std > 0:
            disc_r /= std
        disc_values.append(disc_r)

    if not step_indices:
        del data
        return 0.0, 0

    all_indices = np.concatenate(step_indices)
    all_disc = np.concatenate(disc_values).astype(np.float32)
    ep_count = int(include_mask.sum())
    total = len(all_indices)

    # shuffle and train in mini-batches
    perm = np.random.permutation(total)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, total, MINI_BATCH):
        idx = perm[start:start + MINI_BATCH]
        raw_idx = all_indices[idx]

        xs = torch.from_numpy(np.array(data["all_xs"][raw_idx], dtype=np.float32))
        actions = torch.from_numpy(np.array(data["all_actions"][raw_idx])).float()
        disc_tensor = torch.from_numpy(all_disc[idx])

        probs = model(xs).squeeze()
        log_prob = actions * torch.log(probs + 1e-8) + (1 - actions) * torch.log(1 - probs + 1e-8)
        loss = -(log_prob * disc_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    del data
    gc.collect()
    return total_loss / max(n_batches, 1), ep_count


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
        gc.collect()

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
