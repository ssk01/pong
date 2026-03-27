"""Generate baseline training report from train_log.csv (PyTorch v1)."""

import csv
import numpy as np
import matplotlib.pyplot as plt

# read data
episodes, rewards, running_means, losses = [], [], [], []
with open("train_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row["episode"]))
        rewards.append(float(row["reward"]))
        running_means.append(float(row["running_mean"]))
        losses.append(float(row["loss"]))

episodes = np.array(episodes)
rewards = np.array(rewards)
running_means = np.array(running_means)
losses = np.array(losses)

# moving average for rewards (window=50)
win = 50
reward_ma = np.convolve(rewards, np.ones(win)/win, mode='valid')

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f"Pong Training Baseline (PyTorch v1) — {len(episodes)} episodes", fontsize=14, fontweight='bold')

# 1. Reward per episode + running mean
ax1 = axes[0]
ax1.scatter(episodes, rewards, alpha=0.15, s=5, color="steelblue", label="Episode Reward")
ax1.plot(episodes, running_means, color="red", linewidth=2, label="Running Mean (α=0.01)")
ax1.plot(episodes[win-1:], reward_ma, color="orange", linewidth=1.5, label=f"MA-{win}")
ax1.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Break-even")
ax1.set_ylabel("Reward")
ax1.set_ylim(-22, 5)
ax1.legend(loc="upper left", fontsize=9)
ax1.set_title("Reward Progression")
ax1.grid(True, alpha=0.3)

# 2. Loss
ax2 = axes[1]
ax2.plot(episodes, losses, alpha=0.3, color="orange", linewidth=0.8)
loss_ma = np.convolve(losses, np.ones(win)/win, mode='valid')
ax2.plot(episodes[win-1:], loss_ma, color="darkred", linewidth=2, label=f"Loss MA-{win}")
ax2.set_ylabel("Loss")
ax2.legend(loc="upper left", fontsize=9)
ax2.set_title("Policy Loss")
ax2.grid(True, alpha=0.3)

# 3. Reward distribution over time (heatmap-style)
ax3 = axes[2]
bin_size = 100
n_bins = len(episodes) // bin_size
bin_means = [rewards[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)]
bin_maxs = [rewards[i*bin_size:(i+1)*bin_size].max() for i in range(n_bins)]
bin_mins = [rewards[i*bin_size:(i+1)*bin_size].min() for i in range(n_bins)]
bin_x = [(i+0.5)*bin_size for i in range(n_bins)]

ax3.bar(bin_x, bin_means, width=bin_size*0.8, alpha=0.6, color="steelblue", label="Mean reward per 100 ep")
ax3.plot(bin_x, bin_maxs, 'g^-', markersize=6, label="Best game per 100 ep")
ax3.plot(bin_x, bin_mins, 'rv-', markersize=4, alpha=0.5, label="Worst game per 100 ep")
ax3.set_ylabel("Reward")
ax3.set_xlabel("Episode")
ax3.legend(loc="upper left", fontsize=9)
ax3.set_title("Per-100-Episode Statistics")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("baseline_v1.png", dpi=150)
plt.show()

# print summary
print("\n=== Baseline Summary (PyTorch v1) ===")
print(f"Total episodes: {len(episodes)}")
print(f"Running mean: {running_means[0]:.2f} → {running_means[-1]:.2f}")
print(f"Best single game: {rewards.max():.0f} (ep {episodes[rewards.argmax()]})")
print(f"Worst single game: {rewards.min():.0f}")
print(f"Last 100 ep avg: {rewards[-100:].mean():.2f}")
print(f"Last 100 ep best: {rewards[-100:].max():.0f}")
