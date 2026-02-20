#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

POSSIBLE_VALS = np.array(
    [1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    dtype=float,
)

# Saaty's RI values for dimensions 1..15
GIVEN_RI = np.array(
    [0, 0, 0.52, 0.89, 1.11, 1.25, 1.35, 1.40, 1.45, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a GIF showing RI approximation convergence as simulation steps increase."
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=5,
        help="Minimum number of simulated matrices shown in the GIF.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum number of simulated matrices shown in the GIF.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=15,
        help="Largest matrix dimension + 1, matching notebook convention (default: 15 => dims 1..14 plotted).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Approximate number of animation frames.",
    )
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ri_convergence_5_to_5000.gif"),
        help="Output GIF path.",
    )
    return parser.parse_args()


def build_frame_steps(min_steps: int, max_steps: int, approx_frames: int) -> np.ndarray:
    if min_steps < 1:
        raise ValueError("min_steps must be >= 1")
    if max_steps < min_steps:
        raise ValueError("max_steps must be >= min_steps")

    # Mix log and linear spacing to keep detail early while covering the full range.
    linear = np.linspace(min_steps, max_steps, num=max(2, approx_frames), dtype=int)
    log_count = max(2, approx_frames // 2)
    logspace = np.geomspace(min_steps, max_steps, num=log_count).astype(int)

    steps = np.unique(np.concatenate(([min_steps], linear, logspace, [max_steps])))
    return steps[(steps >= min_steps) & (steps <= max_steps)]


def sample_reciprocal_matrix(dim: int, rng: np.random.Generator, tri_indices: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    matrix = np.eye(dim, dtype=float)
    vals = rng.choice(POSSIBLE_VALS, size=tri_indices[0].size)
    matrix[tri_indices] = vals
    matrix[(tri_indices[1], tri_indices[0])] = 1.0 / vals
    return matrix


def simulate_running_ri(
    frame_steps: np.ndarray,
    max_steps: int,
    max_dimension: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    dimensions = np.arange(2, max_dimension)
    tri_lookup = {dim: np.triu_indices(dim, k=1) for dim in dimensions}

    running_sum = np.zeros(max_dimension, dtype=float)
    estimates_by_frame: list[np.ndarray] = []
    rmse_by_frame: list[float] = []

    frame_idx = 0
    next_capture = int(frame_steps[frame_idx])
    for step in range(1, max_steps + 1):
        for dim in dimensions:
            matrix = sample_reciprocal_matrix(dim, rng, tri_lookup[dim])
            max_eig_val = np.max(np.linalg.eigvals(matrix).real)
            ci = (max_eig_val - dim) / (dim - 1)
            running_sum[dim] += ci

        if step == next_capture:
            current_ri = np.zeros(max_dimension, dtype=float)
            current_ri[dimensions] = running_sum[dimensions] / step

            estimates_by_frame.append(current_ri[1:max_dimension].copy())
            valid_dims = np.arange(3, max_dimension)
            rmse = np.sqrt(np.mean((current_ri[valid_dims] - GIVEN_RI[valid_dims]) ** 2))
            rmse_by_frame.append(float(rmse))

            frame_idx += 1
            if frame_idx >= len(frame_steps):
                break
            next_capture = int(frame_steps[frame_idx])

    return np.array(estimates_by_frame), np.array(rmse_by_frame)


def build_animation(
    frame_steps: np.ndarray,
    estimates: np.ndarray,
    rmses: np.ndarray,
    max_dimension: int,
    fps: int,
    output_path: Path,
) -> None:
    x_values = np.arange(1, max_dimension)
    given_curve = GIVEN_RI[: max_dimension - 1]

    fig, (ax_curve, ax_rmse) = plt.subplots(1, 2, figsize=(12, 5))

    (line_estimate,) = ax_curve.plot([], [], color="tab:red", linewidth=2, label="Estimated RI")
    ax_curve.plot(x_values, given_curve, color="tab:blue", linewidth=2, label="Given RI")
    ax_curve.set_title("RI Curve Convergence")
    ax_curve.set_xlabel("Matrix dimension (n x n)")
    ax_curve.set_ylabel("Random Index (RI)")
    ax_curve.set_xlim(x_values.min(), x_values.max())
    y_max = max(float(np.max(given_curve)), float(np.max(estimates))) * 1.1
    ax_curve.set_ylim(0, y_max)
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(loc="lower right")
    step_text = ax_curve.text(0.02, 0.95, "", transform=ax_curve.transAxes, va="top")

    ax_rmse.plot(frame_steps, rmses, color="tab:gray", linewidth=1.5, label="RMSE")
    (rmse_point,) = ax_rmse.plot([], [], "o", color="tab:orange", markersize=8)
    ax_rmse.set_title("Error vs Simulation Steps")
    ax_rmse.set_xlabel("Matrices simulated")
    ax_rmse.set_ylabel("RMSE (dims 3..14)")
    ax_rmse.set_xlim(frame_steps.min(), frame_steps.max())
    ax_rmse.set_ylim(0, float(np.max(rmses) * 1.1))
    ax_rmse.grid(alpha=0.3)

    def init() -> tuple:
        line_estimate.set_data(x_values, estimates[0])
        rmse_point.set_data([frame_steps[0]], [rmses[0]])
        step_text.set_text(f"Matrices simulated: {int(frame_steps[0])}")
        return line_estimate, rmse_point, step_text

    def update(frame_index: int) -> tuple:
        line_estimate.set_data(x_values, estimates[frame_index])
        rmse_point.set_data([frame_steps[frame_index]], [rmses[frame_index]])
        step_text.set_text(f"Matrices simulated: {int(frame_steps[frame_index])}")
        return line_estimate, rmse_point, step_text

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frame_steps),
        interval=1000 / max(1, fps),
        blit=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.max_dimension > len(GIVEN_RI):
        raise ValueError(f"max_dimension must be <= {len(GIVEN_RI)} for the built-in GIVEN_RI list")

    frame_steps = build_frame_steps(args.min_steps, args.max_steps, args.frames)

    estimates, rmses = simulate_running_ri(
        frame_steps=frame_steps,
        max_steps=args.max_steps,
        max_dimension=args.max_dimension,
        seed=args.seed,
    )

    build_animation(
        frame_steps=frame_steps,
        estimates=estimates,
        rmses=rmses,
        max_dimension=args.max_dimension,
        fps=args.fps,
        output_path=args.output,
    )

    print(f"Saved GIF: {args.output}")
    print(f"Frames: {len(frame_steps)}")
    print(f"Steps shown: {int(frame_steps[0])} -> {int(frame_steps[-1])}")


if __name__ == "__main__":
    main()
