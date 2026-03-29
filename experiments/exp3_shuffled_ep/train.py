"""Exp3: Shuffled episode order, but keep intra-episode step order."""
import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from offline_common import PongPolicy, load_all_episodes, train_on_episodes, evaluate, LEARNING_RATE

model = PongPolicy()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
episodes = load_all_episodes()

# shuffle episode order (destroy curriculum)
random.seed(42)
random.shuffle(episodes)
print(f"Shuffled {len(episodes)} episodes (curriculum destroyed)")

csv_file = open("results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "avg_loss", "steps"])

print("=== Exp3: Shuffled episodes (no curriculum) ===")
train_on_episodes(episodes, model, optimizer, csv_writer, max_epochs=5, label="shuffled_ep")

torch.save(model.state_dict(), "model.pt")
mean_reward = evaluate(model, num_episodes=50)

csv_writer.writerow(["EVAL", mean_reward, ""])
csv_file.close()
print(f"\nFINAL: mean_reward = {mean_reward:.2f}")
