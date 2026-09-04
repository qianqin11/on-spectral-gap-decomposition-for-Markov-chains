"""Section 7.2 dimension path with n=128 and xi=8 fixed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
LOCAL_CODE = HERE / "code"
sys.path.insert(0, str(LOCAL_CODE))

from section7_dirichlet_study import estimate_problem, fit_power  # noqa: E402


K_GRID = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=int)
N_FIXED = 128
XI_FIXED = 8


def make_plot(
    data: pd.DataFrame,
    y: str,
    lower: str,
    upper: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> list[dict[str, float | str]]:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    styles = [
        ("j = 1", "#D55E00", "o"),
        ("j = k", "#0072B2", "D"),
    ]
    fits: list[dict[str, float | str]] = []
    for role, color, marker in styles:
        role_data = data[data["coordinate_role"] == role].sort_values("k")
        x_values = role_data["k"].to_numpy(dtype=float)
        y_values = role_data[y].to_numpy(dtype=float)
        yerr = np.vstack([
            y_values - role_data[lower].to_numpy(dtype=float),
            role_data[upper].to_numpy(dtype=float) - y_values,
        ])
        ax.errorbar(
            x_values,
            y_values,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=13,
            linewidth=2.8,
            capsize=8,
            capthick=2.4,
            linestyle="none",
            label=role,
        )
        power, r_squared = fit_power(role_data, "k", y)
        fits.append({
            "coordinate_role": role,
            "estimated_power": power,
            "r_squared": r_squared,
        })

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if y == "ratio_S_bar_S":
        ax.set_ylim(0.118, 0.282)
        ax.set_yticks([0.12, 0.16, 0.20, 0.24, 0.28])
        ax.set_yticklabels([
            r"$0.12$",
            r"$0.16$",
            r"$0.20$",
            r"$0.24$",
            r"$0.28$",
        ])
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(K_GRID)
    ax.set_xticklabels([str(value) for value in K_GRID])
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.set_xlabel(r"$k$", fontsize=38, labelpad=8)
    y_label_size = 40 if y == "ratio_S_bar_S" else 36
    y_tick_size = 35 if y == "ratio_S_bar_S" else 29
    ax.set_ylabel(ylabel, fontsize=y_label_size, labelpad=10)
    ax.set_title(title, fontsize=32, pad=14, linespacing=1.0)
    ax.tick_params(axis="x", labelsize=29, pad=7)
    ax.tick_params(axis="y", labelsize=y_tick_size, pad=7)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=1)
    top_margin = 0.79 if "\n" in title else 0.87
    left_margin = 0.23 if y == "ratio_S_bar_S" else 0.21
    fig.subplots_adjust(
        bottom=0.33,
        left=left_margin,
        right=0.98,
        top=top_margin,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return fits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=8_172_028)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="regenerate the figures from the existing raw-results CSV",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    results_dir = args.output_root / "results"
    figures_dir = args.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_results_path = results_dir / "xi8_coordinate_dirichlet_forms.csv"
    if args.plots_only:
        results = pd.read_csv(raw_results_path)
    else:
        rng = np.random.default_rng(args.seed)
        frames = []
        for index, k in enumerate(K_GRID, start=1):
            print(
                f"Estimating k={k}, n={N_FIXED}, xi={XI_FIXED} "
                f"({index}/{len(K_GRID)})",
                flush=True,
            )
            frames.append(
                estimate_problem(
                    rng,
                    int(k),
                    N_FIXED,
                    XI_FIXED,
                    args.pairs,
                    args.batch_size,
                )
            )

        results = pd.concat(frames, ignore_index=True)
        results["coordinate_role"] = np.where(
            results["coordinate"] == 1, "j = 1", "j = k"
        )
        results.to_csv(raw_results_path, index=False)

    if (results["maximum_log_envelope_error"].max() >= 1e-8 or
            (results[["ratio_S_bar_S", "ratio_S_posterior_variance"]] <= 0).any().any()):
        raise RuntimeError("Numerical validation failed")

    fit_rows = []
    specifications = [
        (
            "ratio_S_bar_S",
            "ratio_S_bar_S_lower_95",
            "ratio_S_bar_S_upper_95",
            r"$\widehat{\mathcal{E}}_S(f_j)/\widehat{\mathcal{E}}_{\bar{S}}(f_j)$",
            r"Dirichlet-form ratio vs $k$",
            figures_dir / "endpoint_coordinates_k_E_S_to_E_bar_S_xi8.png",
        ),
        (
            "ratio_S_posterior_variance",
            "ratio_S_posterior_variance_lower_95",
            "ratio_S_posterior_variance_upper_95",
            r"$\widehat{\mathcal{E}}_S(f_j)/\widehat{\mathrm{var}}_\pi(f_j)$",
            "Coordinatewise Rayleigh quotient\n" + r"vs $k$",
            figures_dir / "endpoint_coordinates_k_E_S_to_posterior_variance_xi8.png",
        ),
    ]
    for y, lower, upper, ylabel, title, path in specifications:
        current_fits = make_plot(results, y, lower, upper, ylabel, title, path)
        for fit in current_fits:
            fit_rows.append({"ratio": y, **fit})
    pd.DataFrame(fit_rows).to_csv(results_dir / "xi8_power_fits.csv", index=False)

    if not args.plots_only:
        (results_dir / "configuration.txt").write_text(
            "\n".join([
                f"seed = {args.seed}",
                f"n_pairs per (k, n, xi) = {args.pairs}",
                f"rows per memory-safe batch = {args.batch_size}",
                "fixed active coordinates q0 = 8",
                "k values = " + ", ".join(map(str, K_GRID)),
                f"fixed n = {N_FIXED}",
                f"fixed xi = {XI_FIXED}",
                "implementation = original full-dimensional sampler",
                "S and bar S directions = conditionally independent given beta",
                "random-number generator = NumPy Generator(PCG64)",
            ]) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
