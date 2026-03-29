"""Exp3: Shuffled episode order, but keep intra-episode step order."""
import sys, os, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from offline_common import PongPolicy, train_streaming, evaluate, LEARNING_RATE

model = PongPolicy()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

csv_file = open("results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "avg_loss", "episodes"])

def shuffle_batches(files, epoch):
    """Shuffle batch file order each epoch (destroys curriculum)."""
    random.seed(42 + epoch)
    files = list(files)
    random.shuffle(files)
    return files

print("=== Exp3: Shuffled episodes (no curriculum) ===")
train_streaming(model, optimizer, csv_writer, max_epochs=5, label="shuffled_ep",
                batch_order_fn=shuffle_batches)

torch.save(model.state_dict(), "model.pt")
mean_reward = evaluate(model, num_episodes=50)

csv_writer.writerow(["EVAL", mean_reward, ""])
csv_file.close()
print(f"\nFINAL: mean_reward = {mean_reward:.2f}")
