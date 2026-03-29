"""
Run all offline experiments with different loss types and data orderings.
Each combo: train 3 epochs on all 247 batches, then evaluate 50 games.
"""
import sys, os, csv, random, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from offline_common import PongPolicy, train_streaming, evaluate, LEARNING_RATE

EXPERIMENTS = [
    # (name, loss_type, shuffle_batches, filter_fn)
    ("seq_pg",       "pg",       False, None),
    ("seq_bc",       "bc",       False, None),
    ("seq_weighted", "weighted", False, None),
    ("shuf_pg",      "pg",       True,  None),
    ("shuf_bc",      "bc",       True,  None),
    ("shuf_weighted","weighted", True,  None),
    ("filt_bc",      "bc",       False, lambda ep: ep["total_reward"] > -15),
    ("filt_weighted","weighted", False, lambda ep: ep["total_reward"] > -15),
]


def shuffle_fn(files, epoch):
    random.seed(42 + epoch)
    files = list(files)
    random.shuffle(files)
    return files


results = []
total_start = time.time()

for name, loss_type, shuffle, filter_fn in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name} | loss={loss_type} | shuffle={shuffle} | filter={'yes' if filter_fn else 'no'}")
    print(f"{'='*60}")

    model = PongPolicy()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    csv_file = open(f"results_{name}.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "avg_loss", "episodes"])

    t0 = time.time()
    train_streaming(
        model, optimizer, csv_writer,
        max_epochs=3, label=name,
        batch_order_fn=shuffle_fn if shuffle else None,
        filter_fn=filter_fn,
        loss_type=loss_type,
    )
    train_time = time.time() - t0

    torch.save(model.state_dict(), f"model_{name}.pt")
    mean_reward = evaluate(model, num_episodes=30)

    csv_writer.writerow(["EVAL", mean_reward, ""])
    csv_file.close()

    results.append((name, loss_type, shuffle, filter_fn is not None, mean_reward, train_time))
    print(f">>> {name}: mean_reward={mean_reward:.2f}, train_time={train_time:.0f}s")

# Final summary
elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"ALL DONE in {elapsed/60:.1f} minutes")
print(f"{'='*60}")
print(f"{'Name':<20} {'Loss':<10} {'Shuffle':<8} {'Filter':<8} {'Reward':<10} {'Time':<8}")
print("-" * 64)
for name, lt, shuf, filt, reward, t in results:
    print(f"{name:<20} {lt:<10} {str(shuf):<8} {str(filt):<8} {reward:<10.2f} {t:<8.0f}s")

# Save summary
with open("summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "loss_type", "shuffle", "filter", "mean_reward", "train_time_s"])
    for r in results:
        w.writerow(r)

print("\nFINAL: all experiments complete")
