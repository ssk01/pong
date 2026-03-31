"""
Visualize learned W1 weights of a trained Pong policy network.
Each of the 200 hidden neurons has 6400 weights → reshape to 80x80 to see
what spatial pattern each neuron is sensitive to.

Usage:
    python viz_weights.py                          # default: save_torch_rec.pt
    python viz_weights.py --model save_torch.pt    # any checkpoint
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="save_torch_rec.pt", help="path to .pt checkpoint")
    parser.add_argument("--top", type=int, default=40, help="show top N neurons by variance")
    parser.add_argument("--output", default="weights_viz.png")
    args = parser.parse_args()

    state = torch.load(args.model, weights_only=True, map_location="cpu")
    W1 = state["fc1.weight"].numpy()  # (200, 6400)

    # sort by variance (most structured neurons first)
    var = W1.var(axis=1)
    order = np.argsort(-var)

    cols = 8
    rows = args.top // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2))
    fig.suptitle(f"Learned W1 Weights — {args.top} of {W1.shape[0]} neurons\n{args.model}", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i >= args.top:
            ax.axis("off")
            continue
        idx = order[i]
        w = W1[idx].reshape(80, 80)
        ax.imshow(w, cmap="gray", interpolation="nearest")
        ax.set_title(f"#{idx}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
