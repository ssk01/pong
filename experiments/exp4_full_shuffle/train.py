"""Exp4: Fully shuffled steps across episodes (experience replay style)."""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
from offline_common import PongPolicy, load_all_episodes, evaluate, LEARNING_RATE

model = PongPolicy()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
episodes = load_all_episodes()

# flatten all steps into one big array
all_xs = np.vstack([ep["xs"] for ep in episodes])
all_actions = np.concatenate([ep["actions"] for ep in episodes])
all_disc = np.concatenate([ep["discounted_rewards"] for ep in episodes])
total_steps = len(all_xs)
print(f"Flattened {total_steps:,} steps from {len(episodes)} episodes")

csv_file = open("results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "avg_loss", "steps"])

BATCH = 2048

print("=== Exp4: Fully shuffled steps (experience replay) ===")
for epoch in range(5):
    # shuffle all steps
    perm = np.random.permutation(total_steps)
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, total_steps, BATCH):
        idx = perm[start:start+BATCH]

        xs = torch.from_numpy(all_xs[idx])
        actions = torch.from_numpy(all_actions[idx]).float()
        disc_r = all_disc[idx].copy()
        disc_r -= disc_r.mean()
        std = disc_r.std()
        if std > 0:
            disc_r /= std
        disc_tensor = torch.from_numpy(disc_r.astype(np.float32))

        probs = model(xs).squeeze()
        log_prob = actions * torch.log(probs + 1e-8) + (1 - actions) * torch.log(1 - probs + 1e-8)
        loss = -(log_prob * disc_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    print(f"  epoch {epoch+1}/5 | avg_loss: {avg_loss:.4f} | batches: {n_batches}")
    csv_writer.writerow([epoch+1, avg_loss, total_steps])

torch.save(model.state_dict(), "model.pt")
mean_reward = evaluate(model, num_episodes=50)

csv_writer.writerow(["EVAL", mean_reward, ""])
csv_file.close()
print(f"\nFINAL: mean_reward = {mean_reward:.2f}")
