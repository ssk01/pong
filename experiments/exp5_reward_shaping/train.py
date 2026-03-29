"""Exp5: Reward shaping — small positive reward for returning the ball.
Based on v4b architecture (4 env, pure CPU).
Modified: +0.01 reward when ball moves away from agent after being near."""
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

D = 80 * 80
H = 200
BATCH_SIZE = 10
LEARNING_RATE = 1e-4
GAMMA = 0.99
NUM_ENVS = 4
RETURN_REWARD = 0.01  # small reward for returning the ball


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


def detect_ball_return(prev_frame, cur_frame):
    """Detect if ball changed direction from approaching to moving away.
    Returns True if ball was moving left (toward agent) and now moves right."""
    if prev_frame is None or cur_frame is None:
        return False
    # ball is the small moving object — find its x position
    prev_2d = prev_frame.reshape(80, 80)
    cur_2d = cur_frame.reshape(80, 80)
    # right half is agent's side (x > 40), ball moving right = away from agent
    prev_cols = np.where(prev_2d.sum(axis=0) > 0)[0]
    cur_cols = np.where(cur_2d.sum(axis=0) > 0)[0]
    if len(prev_cols) == 0 or len(cur_cols) == 0:
        return False
    prev_ball_x = prev_cols.mean()
    cur_ball_x = cur_cols.mean()
    # ball was moving left (decreasing x) and now moving right (increasing x) = return
    return cur_ball_x > prev_ball_x + 2  # threshold to avoid noise


def discount_rewards(r):
    discounted = np.zeros_like(r)
    running = 0.0
    for t in reversed(range(len(r))):
        if r[t] != 0 and abs(r[t]) >= 0.5:  # only reset on real score events, not shaping rewards
            running = 0.0
        running = running * GAMMA + r[t]
        discounted[t] = running
    return discounted


def main():
    print(f"Exp5: Reward Shaping | return_reward={RETURN_REWARD} | {NUM_ENVS} envs")

    model = PongPolicy()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    envs = gym.make_vec("ALE/Pong-v5", num_envs=NUM_ENVS, vectorization_mode="async")
    observations, _ = envs.reset()

    prev_x = [None] * NUM_ENVS
    prev_raw = [None] * NUM_ENVS  # raw preprocessed frames for ball detection
    env_xs = [[] for _ in range(NUM_ENVS)]
    env_actions = [[] for _ in range(NUM_ENVS)]
    env_rewards = [[] for _ in range(NUM_ENVS)]
    env_shaped_count = [0] * NUM_ENVS

    all_xs, all_actions, all_discounted = [], [], []

    running_reward = None
    episode_number = 0
    t_start = time.time()

    csv_file = open("train_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "running_mean", "shaped_rewards"])

    while True:
        xs = np.zeros((NUM_ENVS, D), dtype=np.float32)
        cur_raws = []
        for i in range(NUM_ENVS):
            cur = prepro(observations[i])
            cur_raws.append(cur.copy())
            xs[i] = cur - prev_x[i] if prev_x[i] is not None else np.zeros(D, dtype=np.float32)
            prev_x[i] = cur

        with torch.no_grad():
            aprobs = model(torch.from_numpy(xs)).squeeze(-1).numpy()

        randoms = np.random.uniform(size=NUM_ENVS)
        actions_int = np.where(randoms < aprobs, 2, 3)
        ys = np.where(randoms < aprobs, 1.0, 0.0)

        for i in range(NUM_ENVS):
            env_xs[i].append(xs[i])
            env_actions[i].append(ys[i])

        observations, rewards, terminateds, truncateds, infos = envs.step(actions_int)

        # reward shaping: detect ball returns
        for i in range(NUM_ENVS):
            shaped_r = float(rewards[i])
            if rewards[i] == 0 and detect_ball_return(prev_raw[i], cur_raws[i]):
                shaped_r += RETURN_REWARD
                env_shaped_count[i] += 1
            env_rewards[i].append(shaped_r)
            prev_raw[i] = cur_raws[i]

        dones = terminateds | truncateds
        for i in range(NUM_ENVS):
            if not dones[i]:
                continue

            episode_number += 1
            reward_sum = sum(r for r in env_rewards[i] if abs(r) >= 0.5)  # original reward only

            R = np.array(env_rewards[i], dtype=np.float32)
            disc_R = discount_rewards(R)
            disc_R -= disc_R.mean()
            std = disc_R.std()
            if std > 0:
                disc_R /= std

            all_xs.append(np.array(env_xs[i]))
            all_actions.append(np.array(env_actions[i]))
            all_discounted.append(disc_R)

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            shaped = env_shaped_count[i]
            print(f"ep {episode_number} (env {i}) reward: {reward_sum:.0f} | running mean: {running_reward:.2f} | shaped: {shaped}")

            csv_writer.writerow([episode_number, reward_sum, f"{running_reward:.4f}", shaped])
            csv_file.flush()

            env_xs[i] = []
            env_actions[i] = []
            env_rewards[i] = []
            env_shaped_count[i] = 0
            prev_x[i] = None
            prev_raw[i] = None

            if episode_number % BATCH_SIZE == 0 and len(all_xs) > 0:
                batch_x = torch.from_numpy(np.vstack(all_xs))
                batch_y = torch.from_numpy(np.concatenate(all_actions)).float()
                disc_tensor = torch.from_numpy(np.concatenate(all_discounted).astype(np.float32))

                batch_p = model(batch_x).squeeze()
                log_prob = batch_y * torch.log(batch_p + 1e-8) + (1 - batch_y) * torch.log(1 - batch_p + 1e-8)
                loss = -(log_prob * disc_tensor).sum()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                elapsed = time.time() - t_start
                eps_per_sec = BATCH_SIZE / elapsed if elapsed > 0 else 0
                print(f"  [batch] loss: {loss.item()/BATCH_SIZE:.2f} | {eps_per_sec:.2f} ep/s")
                t_start = time.time()

                all_xs, all_actions, all_discounted = [], [], []

            if episode_number % 100 == 0:
                torch.save(model.state_dict(), "model.pt")

            if running_reward is not None and running_reward > 0:
                print(f"\n=== REWARD SHAPING REACHED 0 at ep {episode_number}! ===")
                torch.save(model.state_dict(), "model_final.pt")
                csv_file.close()
                envs.close()
                return


if __name__ == "__main__":
    main()
