"""Tests for qc_core.viz.plots.

Plotting is hard to assert on numerically, so these are "smoke tests": they use the headless
Agg backend and confirm each helper runs, returns an Axes, and draws the expected number of
artists. That catches the common breakages (shape mismatches, empty inputs, wrong API) without
trying to pixel-compare images.
"""

import matplotlib
matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt

import numpy as np
import pytest

from qc_core.viz.plots import (
    plot_convergence,
    plot_before_after,
    plot_counts,
    plot_network_2d,
)


def teardown_function(_):
    plt.close("all")


def test_convergence_runs_and_has_a_line():
    ax = plot_convergence([5.0, 3.0, 2.0, 1.5])
    assert len(ax.lines) >= 1


def test_convergence_maximize_flips_sign():
    ax = plot_convergence([-1.0, -2.0, -3.0], maximize=True)
    ydata = ax.lines[0].get_ydata()
    assert list(ydata) == [1.0, 2.0, 3.0]  # negated


def test_convergence_optimum_adds_reference_line():
    ax = plot_convergence([5.0, 4.0], optimum=3.0)
    # one data line + one axhline
    assert len(ax.lines) >= 2


def test_before_after_bar_counts():
    ax = plot_before_after([1, 2, 3], [3, 2, 1])
    # two bar groups of 3 bars each = 6 patches
    assert len(ax.patches) == 6


def test_before_after_length_mismatch_raises():
    with pytest.raises(ValueError):
        plot_before_after([1, 2, 3], [1, 2])


def test_counts_limits_to_top():
    counts = {f"{i:04b}": (16 - i) for i in range(16)}  # 16 distinct strings
    ax = plot_counts(counts, top=5)
    assert len(ax.patches) == 5  # only the top 5 shown


def test_network_2d_draws_nodes_and_edges():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    edges = [(0, 1), (1, 2)]
    ax = plot_network_2d(pos, edges=edges)
    # one PathCollection for the scatter of nodes
    assert len(ax.collections) >= 1
    # two edge lines
    assert len(ax.lines) == 2
