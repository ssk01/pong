"""Exp2: Only learn from good episodes (reward > -15)."""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from offline_common import PongPolicy, load_all_episodes, train_on_episodes, evaluate, LEARNING_RATE

model = PongPolicy()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
episodes = load_all_episodes()

# filter: only episodes with reward > -15
good_episodes = [ep for ep in episodes if ep["total_reward"] > -15]
print(f"Filtered: {len(good_episodes)}/{len(episodes)} episodes (reward > -15)")

csv_file = open("results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "avg_loss", "steps"])

print("=== Exp2: Filtered (reward > -15 only) ===")
train_on_episodes(good_episodes, model, optimizer, csv_writer, max_epochs=5, label="filtered")

torch.save(model.state_dict(), "model.pt")
mean_reward = evaluate(model, num_episodes=50)

csv_writer.writerow(["EVAL", mean_reward, ""])
csv_file.close()
print(f"\nFINAL: mean_reward = {mean_reward:.2f}")
