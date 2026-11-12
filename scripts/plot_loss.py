#!/usr/bin/env python3
"""Plots the training loss written by examples/xor_mlp (for the README).

Run from the repo root (setup in scripts/requirements.txt):
    ./build/examples/xor_mlp build/loss.csv
    .venv/bin/python scripts/plot_loss.py build/loss.csv docs/img/loss.png
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, no window needed (WSL has no display)
import matplotlib.pyplot as plt  # noqa: E402  (must come after choosing the backend)

# Colours: one series, so one accent; text and grid stay neutral.
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def read_loss(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [int(r["epoch"]) for r in rows], [float(r["loss"]) for r in rows]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: plot_loss.py <loss.csv> <out.png>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    epochs, losses = read_loss(src)

    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(epochs, losses, color=SERIES, linewidth=1.6, solid_capstyle="round", solid_joinstyle="round")
    ax.set_yscale("log")  # the loss falls by many orders of magnitude

    # One direct label for the moment that matters: the first epoch with loss below 0.01.
    learned = next((i for i, loss in enumerate(losses) if loss < 0.01), None)
    if learned is not None:
        ax.plot(epochs[learned], losses[learned], "o", color=SERIES, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(f"loss < 0.01 after {epochs[learned]} epochs", (epochs[learned], losses[learned]),
                    xytext=(10, 4), textcoords="offset points", color=TEXT_SECONDARY, fontsize=9)

    # Where it flattens out: an error of ~1e-7 per prediction is the float32 step size near 1.
    if losses[-1] < 1e-10:
        ax.annotate("float32 limit: predictions within ~1e-7 of the targets",
                    (epochs[-1], losses[-1]), xytext=(0, 64), textcoords="offset points",
                    ha="right", color=TEXT_SECONDARY, fontsize=9)

    ax.set_title("XOR, MLP 2-8-1 (ReLU): training loss", color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel("epoch", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("mean squared error (log scale)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlim(0, epochs[-1])
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, which="both")

    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, facecolor=SURFACE)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
