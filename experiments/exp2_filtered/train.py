"""Exp2: Only learn from good episodes (reward > -15)."""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from offline_common import PongPolicy, train_streaming, evaluate, LEARNING_RATE

model = PongPolicy()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

csv_file = open("results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "avg_loss", "episodes"])

print("=== Exp2: Filtered (reward > -18 only) ===")
# Use all batch files but only train on episodes with reward > -18
train_streaming(model, optimizer, csv_writer, max_epochs=3, label="filtered",
                filter_fn=lambda ep: ep["total_reward"] > -18)

torch.save(model.state_dict(), "model.pt")
mean_reward = evaluate(model, num_episodes=50)

csv_writer.writerow(["EVAL", mean_reward, ""])
csv_file.close()
print(f"\nFINAL: mean_reward = {mean_reward:.2f}")
