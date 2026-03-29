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


def get_batch_files():
    """Return all batch file paths sorted."""
    all_files = sorted([
        os.path.join(REPLAY_DIR, f)
        for f in os.listdir(REPLAY_DIR)
        if f.startswith("batch_") and f.endswith(".npz")
    ])
    print(f"Using all {len(all_files)} batch files ({len(all_files)*100} episodes)")
    return all_files


def load_index():
    """Load episode index (lightweight metadata only)."""
    with open(os.path.join(REPLAY_DIR, "index.json")) as f:
        return json.load(f)


def iter_episodes_from_batch(batch_path):
    """Yield episodes one by one from a single batch file. Uses mmap to avoid loading all into RAM."""
    data = np.load(batch_path)
    ep_ids = data["episode_ids"]
    rewards = data["total_rewards"]
    num_steps = data["num_steps"]

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


MAX_SAMPLE_STEPS = 4096  # sample at most this many steps per batch file


def train_on_batch_file(batch_path, model, optimizer, filter_fn=None, loss_type="pg"):
    """Train on one batch file. Samples MAX_SAMPLE_STEPS random steps to control memory.

    loss_type options:
        "pg"       - REINFORCE: -log π(a|s) × R  (on-policy, wrong for offline)
        "bc"       - Behavior Cloning: cross-entropy on expert actions
        "weighted" - Weighted BC: cross-entropy × normalized reward
    """
    data = np.load(batch_path)
    num_steps = data["num_steps"]
    total_rewards = data["total_rewards"]
    total_steps_in_file = int(num_steps.sum())

    offsets = np.concatenate([[0], np.cumsum(num_steps)])
    include_mask = np.ones(len(num_steps), dtype=bool)
    if filter_fn:
        for i in range(len(num_steps)):
            if not filter_fn({"total_reward": float(total_rewards[i])}):
                include_mask[i] = False

    # precompute per-step weights based on loss_type
    all_weights = np.zeros(total_steps_in_file, dtype=np.float32)
    ep_count = 0
    included_step_mask = np.zeros(total_steps_in_file, dtype=bool)

    for i in range(len(num_steps)):
        if not include_mask[i]:
            continue
        start, end = int(offsets[i]), int(offsets[i+1])
        included_step_mask[start:end] = True
        ep_count += 1

        if loss_type == "pg":
            rewards = np.array(data["all_rewards"][start:end])
            running = 0.0
            for t in reversed(range(len(rewards))):
                if rewards[t] != 0:
                    running = 0.0
                running = running * 0.99 + rewards[t]
                all_weights[start + t] = running
            chunk = all_weights[start:end]
            chunk -= chunk.mean()
            std = chunk.std()
            if std > 0:
                chunk /= std
            all_weights[start:end] = chunk
        elif loss_type == "bc":
            all_weights[start:end] = 1.0
        elif loss_type == "weighted":
            # weight by episode reward (normalized to 0-1 range)
            w = (float(total_rewards[i]) + 21.0) / 26.0  # -21→0, +5→1
            all_weights[start:end] = max(w, 0.01)  # floor to avoid zero

    if ep_count == 0:
        del data
        return 0.0, 0

    included_indices = np.where(included_step_mask)[0]
    if len(included_indices) > MAX_SAMPLE_STEPS:
        sample_idx = np.random.choice(included_indices, MAX_SAMPLE_STEPS, replace=False)
    else:
        sample_idx = included_indices
    np.random.shuffle(sample_idx)

    xs = torch.from_numpy(np.array(data["all_xs"][sample_idx], dtype=np.float32))
    actions = torch.from_numpy(np.array(data["all_actions"][sample_idx])).float()
    weights = torch.from_numpy(all_weights[sample_idx])

    del data
    gc.collect()

    probs = model(xs).squeeze()

    if loss_type == "pg":
        log_prob = actions * torch.log(probs + 1e-8) + (1 - actions) * torch.log(1 - probs + 1e-8)
        loss = -(log_prob * weights).mean()
    else:
        # BC / weighted BC: cross-entropy loss
        bce = -(actions * torch.log(probs + 1e-8) + (1 - actions) * torch.log(1 - probs + 1e-8))
        loss = (bce * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return loss.item(), ep_count


def train_streaming(model, optimizer, csv_writer, max_epochs=3, label="",
                    batch_order_fn=None, filter_fn=None, loss_type="pg"):
    """Train model by streaming through batch files. Never loads all data at once."""
    batch_files = get_batch_files()
    print(f"[{label}] {len(batch_files)} batch files, {max_epochs} epochs, loss={loss_type}")

    for epoch in range(max_epochs):
        files = list(batch_files)
        if batch_order_fn:
            files = batch_order_fn(files, epoch)

        epoch_loss = 0.0
        epoch_eps = 0

        for bf in files:
            avg_loss, ep_count = train_on_batch_file(bf, model, optimizer, filter_fn, loss_type=loss_type)
            epoch_loss += avg_loss * ep_count
            epoch_eps += ep_count

        avg = epoch_loss / max(epoch_eps, 1)
        print(f"  epoch {epoch+1}/{max_epochs} | avg_loss: {avg:.4f} | episodes: {epoch_eps}")
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
