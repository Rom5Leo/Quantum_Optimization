"""Reusable plotting helpers for QAOA optimization results.

Generalized from the four-panel visualization used across the antenna notebooks: a network
map, an optimizer convergence curve, a per-variable before/after comparison, and a sampled-
histogram view. Each function draws onto a supplied matplotlib ``Axes`` (or creates one), so
they compose into whatever panel layout a project needs — the antenna, dispatch, and routing
projects all reuse these rather than re-writing matplotlib each time.

Style is deliberately minimal (no imposed theme) so projects can restyle freely.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(history: Sequence[float], ax=None, *, maximize: bool = False,
                     optimum: float | None = None, title: str = "Optimizer convergence"):
    """Plot the optimizer cost per iteration.

    Parameters
    ----------
    history  : cost value at each optimizer iteration (e.g. QAOAResult.history).
    maximize : if True, plot the negated history so "up = better" (throughput-style).
    optimum  : optional reference line (e.g. the classical optimum) drawn dashed.
    """
    ax = ax or plt.gca()
    y = [-h for h in history] if maximize else list(history)
    ax.plot(y, lw=2)
    if optimum is not None:
        ax.axhline(-optimum if maximize else optimum, ls="--",
                   label="reference optimum")
        ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective" + (" (higher better)" if maximize else " (lower better)"))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_before_after(before: Sequence[float], after: Sequence[float], ax=None, *,
                      labels: tuple[str, str] = ("before", "after"),
                      xlabel: str = "variable", ylabel: str = "value",
                      title: str = "Before vs after"):
    """Grouped bar chart comparing a per-variable quantity before and after optimization.

    Used for e.g. per-antenna tilt (baseline vs optimized) or per-variable anything. The two
    sequences must be the same length.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if len(before) != len(after):
        raise ValueError("before and after must have the same length")
    ax = ax or plt.gca()
    x = np.arange(len(before))
    w = 0.38
    ax.bar(x - w / 2, before, w, label=labels[0])
    ax.bar(x + w / 2, after, w, label=labels[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_counts(counts: Mapping[str, int], ax=None, *, top: int = 12,
                title: str = "Sampled bitstrings"):
    """Bar chart of the most frequently sampled bitstrings.

    Shows the ``top`` highest-count states, sorted descending — the QAOA read-out at a glance
    (a peaked distribution means good concentration; near-uniform means weak concentration).
    """
    ax = ax or plt.gca()
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("counts")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_network_2d(positions, edges=None, ax=None, *, node_labels=True,
                    title: str = "Network"):
    """Scatter nodes at 2-D ``positions`` and draw ``edges`` between them.

    Parameters
    ----------
    positions : array-like of shape (n, 2), the (x, y) of each node.
    edges     : optional iterable of (i, j) index pairs to draw as lines (e.g. the
                interference/coupling graph). Drawn beneath the nodes.
    node_labels : annotate each node with its index.
    """
    positions = np.asarray(positions, dtype=float)
    ax = ax or plt.gca()
    if edges:
        for i, j in edges:
            ax.plot([positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]], color="#c0c0c0", lw=1, zorder=1)
    ax.scatter(positions[:, 0], positions[:, 1], s=180, zorder=2)
    if node_labels:
        for idx, (x, y) in enumerate(positions):
            ax.annotate(str(idx), (x, y), color="white", ha="center", va="center",
                        fontsize=8, fontweight="bold", zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return ax
