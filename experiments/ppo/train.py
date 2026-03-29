"""
PPO (Proximal Policy Optimization) for Pong.
Based on v4b architecture (4 parallel envs, pure CPU).

Key differences from REINFORCE (v0-v6):
1. Actor-Critic: shared network with policy + value heads
2. Fixed-step rollouts: collect N steps per env, not full episodes
3. Multiple epochs: reuse same batch K times with clip constraint
4. GAE: Generalized Advantage Estimation for lower variance
5. Gradient clipping: prevents NaN explosions (v5's problem)

Usage:
    python train.py
    python train.py --resume
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

# --- hyperparameters ---
D = 80 * 80
H = 200
NUM_ENVS = 4
N_STEPS = 128        # steps per env per rollout
K_EPOCHS = 4         # PPO epochs per batch
MINI_BATCH_SIZE = 128
CLIP_EPS = 0.2
LR = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


class PongActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(H, 1)
        self.value_head = nn.Linear(H, 1)

    def forward(self, x):
        h = self.shared(x)
        prob = torch.sigmoid(self.policy_head(h))
        value = self.value_head(h)
        return prob.squeeze(-1), value.squeeze(-1)

    def get_value(self, x):
        h = self.shared(x)
        return self.value_head(h).squeeze(-1)


def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """Generalized Advantage Estimation."""
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"PPO | {NUM_ENVS} envs | {N_STEPS} steps | {K_EPOCHS} epochs | clip={CLIP_EPS}")

    model = PongActorCritic()
    if args.resume:
        model.load_state_dict(torch.load("save_ppo.pt", weights_only=True))
        print("Loaded checkpoint")

    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    envs = gym.make_vec("ALE/Pong-v5", num_envs=NUM_ENVS, vectorization_mode="async")
    observations, _ = envs.reset()

    prev_x = [None] * NUM_ENVS
    running_reward = None
    episode_number = 0
    total_steps = 0
    t_start = time.time()

    csv_file = open("train_log_ppo.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "running_mean", "policy_loss", "value_loss", "entropy"])

    # current episode reward tracking
    ep_rewards = [0.0] * NUM_ENVS

    while True:
        # === Rollout: collect N_STEPS per env ===
        mb_obs = np.zeros((N_STEPS, NUM_ENVS, D), dtype=np.float32)
        mb_actions = np.zeros((N_STEPS, NUM_ENVS), dtype=np.float32)
        mb_probs = np.zeros((N_STEPS, NUM_ENVS), dtype=np.float32)
        mb_values = np.zeros((N_STEPS, NUM_ENVS), dtype=np.float32)
        mb_rewards = np.zeros((N_STEPS, NUM_ENVS), dtype=np.float32)
        mb_dones = np.zeros((N_STEPS, NUM_ENVS), dtype=np.float32)

        for step in range(N_STEPS):
            # preprocess
            xs = np.zeros((NUM_ENVS, D), dtype=np.float32)
            for i in range(NUM_ENVS):
                cur = prepro(observations[i])
                xs[i] = cur - prev_x[i] if prev_x[i] is not None else np.zeros(D, dtype=np.float32)
                prev_x[i] = cur

            # inference
            with torch.no_grad():
                x_tensor = torch.from_numpy(xs)
                probs, values = model(x_tensor)
                probs_np = probs.numpy()
                values_np = values.numpy()

            # sample actions
            randoms = np.random.uniform(size=NUM_ENVS)
            actions_binary = (randoms < probs_np).astype(np.float32)
            actions_env = np.where(actions_binary == 1, 2, 3)

            # store
            mb_obs[step] = xs
            mb_actions[step] = actions_binary
            mb_probs[step] = probs_np
            mb_values[step] = values_np

            # step envs
            observations, rewards, terminateds, truncateds, infos = envs.step(actions_env)
            dones = terminateds | truncateds
            mb_rewards[step] = rewards
            mb_dones[step] = dones.astype(np.float32)

            # track episode rewards
            for i in range(NUM_ENVS):
                ep_rewards[i] += rewards[i]
                if dones[i]:
                    episode_number += 1
                    reward_sum = ep_rewards[i]
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

                    if rewards[i] != 0:
                        marker = " !!!!!!!!" if reward_sum > 0 else ""
                        print(f"ep {episode_number} (env {i}) reward: {reward_sum:.0f} | running mean: {running_reward:.2f}{marker}")

                    csv_writer.writerow([episode_number, reward_sum, f"{running_reward:.4f}", "", "", ""])
                    csv_file.flush()

                    if episode_number % 100 == 0:
                        torch.save(model.state_dict(), "save_ppo.pt")
                        print("  [checkpoint saved]")

                    ep_rewards[i] = 0.0
                    prev_x[i] = None

            total_steps += NUM_ENVS

        # === Compute advantages using GAE ===
        with torch.no_grad():
            # get value of last observation for bootstrapping
            last_xs = np.zeros((NUM_ENVS, D), dtype=np.float32)
            for i in range(NUM_ENVS):
                cur = prepro(observations[i])
                last_xs[i] = cur - prev_x[i] if prev_x[i] is not None else np.zeros(D, dtype=np.float32)
            next_values = model.get_value(torch.from_numpy(last_xs)).numpy()

        # compute GAE per env, then flatten
        mb_advantages = np.zeros_like(mb_rewards)
        mb_returns = np.zeros_like(mb_rewards)
        for i in range(NUM_ENVS):
            adv, ret = compute_gae(
                mb_rewards[:, i], mb_values[:, i], mb_dones[:, i],
                next_values[i], GAMMA, GAE_LAMBDA
            )
            mb_advantages[:, i] = adv
            mb_returns[:, i] = ret

        # flatten: (N_STEPS, NUM_ENVS, ...) → (N_STEPS * NUM_ENVS, ...)
        batch_size = N_STEPS * NUM_ENVS
        b_obs = torch.from_numpy(mb_obs.reshape(batch_size, D))
        b_actions = torch.from_numpy(mb_actions.reshape(batch_size))
        b_old_probs = torch.from_numpy(mb_probs.reshape(batch_size))
        b_advantages = torch.from_numpy(mb_advantages.reshape(batch_size))
        b_returns = torch.from_numpy(mb_returns.reshape(batch_size))

        # normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # === PPO Update: K epochs over the same batch ===
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        indices = np.arange(batch_size)
        for epoch in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, batch_size, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_idx = indices[start:end]

                obs_mb = b_obs[mb_idx]
                actions_mb = b_actions[mb_idx]
                old_probs_mb = b_old_probs[mb_idx]
                advantages_mb = b_advantages[mb_idx]
                returns_mb = b_returns[mb_idx]

                # forward pass with current policy
                new_probs, new_values = model(obs_mb)

                # compute new log probs
                new_log_probs = actions_mb * torch.log(new_probs + 1e-8) + (1 - actions_mb) * torch.log(1 - new_probs + 1e-8)
                old_log_probs = actions_mb * torch.log(old_probs_mb + 1e-8) + (1 - actions_mb) * torch.log(1 - old_probs_mb + 1e-8)

                # ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # clipped surrogate loss
                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = nn.functional.mse_loss(new_values, returns_mb)

                # entropy bonus
                entropy = -(new_probs * torch.log(new_probs + 1e-8) + (1 - new_probs) * torch.log(1 - new_probs + 1e-8)).mean()

                # total loss
                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # logging
        elapsed = time.time() - t_start
        fps = total_steps / elapsed if elapsed > 0 else 0
        avg_pl = total_policy_loss / n_updates
        avg_vl = total_value_loss / n_updates
        avg_ent = total_entropy / n_updates
        rm_str = f"{running_reward:.2f}" if running_reward is not None else "N/A"
        print(f"  [PPO update] steps: {total_steps} | fps: {fps:.0f} | "
              f"policy_loss: {avg_pl:.4f} | value_loss: {avg_vl:.4f} | entropy: {avg_ent:.4f} | "
              f"running_mean: {rm_str}")


if __name__ == "__main__":
    main()
