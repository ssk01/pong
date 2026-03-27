"""
Live training curve plotter.
Reads train_log.csv and auto-refreshes every 2 seconds.

Usage:
    python plot_live.py
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import os


def read_csv():
    if not os.path.exists("train_log.csv"):
        return [], [], [], []
    episodes, rewards, running_means, losses = [], [], [], []
    with open("train_log.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            running_means.append(float(row["running_mean"]))
            losses.append(float(row["loss"]))
    return episodes, rewards, running_means, losses


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("Pong Policy Gradient Training", fontsize=14)


def update(frame):
    episodes, rewards, running_means, losses = read_csv()
    if not episodes:
        return

    ax1.clear()
    ax1.plot(episodes, rewards, alpha=0.3, color="steelblue", linewidth=0.8, label="Episode Reward")
    ax1.plot(episodes, running_means, color="red", linewidth=2, label="Running Mean")
    ax1.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Break-even")
    ax1.set_ylabel("Reward")
    ax1.set_ylim(-22, 22)
    ax1.legend(loc="upper left")
    ax1.set_title(f"Episode {episodes[-1]} | Running Mean: {running_means[-1]:.2f}")
    ax1.grid(True, alpha=0.3)

    ax2.clear()
    ax2.plot(episodes, losses, color="orange", alpha=0.5, linewidth=0.8)
    # moving average for loss
    if len(losses) >= 10:
        window = 10
        ma = [sum(losses[max(0, i - window):i]) / min(i, window) for i in range(1, len(losses) + 1)]
        ax2.plot(episodes, ma, color="darkred", linewidth=2, label=f"Loss (MA-{window})")
        ax2.legend(loc="upper left")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Episode")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()


ani = animation.FuncAnimation(fig, update, interval=2000)
plt.show()
