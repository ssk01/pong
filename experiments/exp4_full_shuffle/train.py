"""Exp4: Fully shuffled steps across episodes (experience replay style).
Streaming version: load one batch at a time, collect steps, shuffle within batch."""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
from offline_common import PongPolicy, get_batch_files, iter_episodes_from_batch, evaluate, LEARNING_RATE

model = PongPolicy()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

csv_file = open("results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "avg_loss", "batches"])

MINI_BATCH = 2048
batch_files = get_batch_files()

print(f"=== Exp4: Fully shuffled steps (experience replay) ===")
print(f"{len(batch_files)} batch files")

for epoch in range(5):
    # shuffle batch file order
    file_order = np.random.permutation(len(batch_files))
    epoch_loss = 0.0
    n_batches = 0

    for fi in file_order:
        # load one batch file, flatten all steps, shuffle
        all_xs, all_acts, all_disc = [], [], []
        for ep in iter_episodes_from_batch(batch_files[fi]):
            disc_r = ep["discounted_rewards"].copy()
            disc_r -= disc_r.mean()
            std = disc_r.std()
            if std > 0:
                disc_r /= std
            all_xs.append(ep["xs"])
            all_acts.append(ep["actions"])
            all_disc.append(disc_r)

        if not all_xs:
            continue

        xs = np.vstack(all_xs)
        acts = np.concatenate(all_acts)
        disc = np.concatenate(all_disc).astype(np.float32)
        n = len(xs)

        # fully shuffle steps within this batch
        perm = np.random.permutation(n)

        for start in range(0, n, MINI_BATCH):
            idx = perm[start:start+MINI_BATCH]
            bx = torch.from_numpy(xs[idx])
            ba = torch.from_numpy(acts[idx]).float()
            bd = torch.from_numpy(disc[idx])

            probs = model(bx).squeeze()
            log_prob = ba * torch.log(probs + 1e-8) + (1 - ba) * torch.log(1 - probs + 1e-8)
            loss = -(log_prob * bd).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    print(f"  epoch {epoch+1}/5 | avg_loss: {avg_loss:.4f} | mini_batches: {n_batches}")
    csv_writer.writerow([epoch+1, avg_loss, n_batches])

torch.save(model.state_dict(), "model.pt")
mean_reward = evaluate(model, num_episodes=50)

csv_writer.writerow(["EVAL", mean_reward, ""])
csv_file.close()
print(f"\nFINAL: mean_reward = {mean_reward:.2f}")
