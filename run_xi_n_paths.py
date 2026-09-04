"""Full-dimensional xi and n paths with q0=8.

The xi path fixes (k,n)=(16,128), and the n path fixes (k,xi)=(16,8).
Coordinate 1 is informed and coordinate k=16 is prior-only.
"""

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

import section7_dirichlet_study as study  # noqa: E402


GRID = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=int)
Q0 = 8
FIXED_VALUE = 8


def make_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    lower: str,
    upper: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> list[dict[str, float | str]]:
    show_legend = x == "xi"
    fig, ax = plt.subplots(figsize=(11, 7.5))
    styles = [
        ("j = 1", "#D55E00", "o"),
        ("j = k", "#0072B2", "D"),
    ]
    fits: list[dict[str, float | str]] = []
    for role, color, marker in styles:
        role_data = data[data["coordinate_role"] == role].sort_values(x)
        x_values = role_data[x].to_numpy(dtype=float)
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
        power, r_squared = study.fit_power(role_data, x, y)
        fits.append({
            "path": x,
            "ratio": y,
            "coordinate_role": role,
            "estimated_power": power,
            "r_squared": r_squared,
        })

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if y == "ratio_S_bar_S":
        if x == "xi":
            ax.set_ylim(0.185, 0.305)
            ticks = [0.20, 0.225, 0.25, 0.275, 0.30]
            labels = [r"$0.20$", r"$0.225$", r"$0.25$", r"$0.275$", r"$0.30$"]
        else:
            ax.set_ylim(0.217, 0.282)
            ticks = [0.22, 0.24, 0.26, 0.28]
            labels = [r"$0.22$", r"$0.24$", r"$0.26$", r"$0.28$"]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif x == "n" and y == "ratio_S_posterior_variance":
        ax.set_ylim(0.0035, 0.033)
        ax.set_yticks([0.004, 0.01, 0.03])
        ax.set_yticklabels([r"$0.004$", r"$0.01$", r"$0.03$"])
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(GRID)
    ax.set_xticklabels([str(value) for value in GRID])
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.set_xlabel(r"$\xi$" if x == "xi" else r"$n$", fontsize=38, labelpad=8)
    enlarged_n_rayleigh_axis = x == "n" and y == "ratio_S_posterior_variance"
    y_label_size = 42 if enlarged_n_rayleigh_axis else (40 if y == "ratio_S_bar_S" else 38)
    y_tick_size = 37 if enlarged_n_rayleigh_axis else (35 if y == "ratio_S_bar_S" else 32)
    ax.set_ylabel(ylabel, fontsize=y_label_size, labelpad=10)
    ax.set_title(title, fontsize=32, pad=14, linespacing=1.0)
    ax.tick_params(axis="x", labelsize=29, pad=7)
    ax.tick_params(axis="y", labelsize=y_tick_size, pad=7)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=1)
    if show_legend:
        ax.legend(
            title="test function",
            fontsize=27,
            title_fontsize=28,
            loc="best",
            ncol=1,
            handletextpad=0.6,
            frameon=False,
        )
    top_margin = 0.79 if "\n" in title else 0.87
    bottom_margin = 0.33
    left_margin = 0.24 if enlarged_n_rayleigh_axis else 0.22
    fig.subplots_adjust(
        bottom=bottom_margin,
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
    parser.add_argument("--seed", type=int, default=8_172_030)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="regenerate tables and figures from the existing raw-results CSV",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    study.Q0 = Q0
    results_dir = args.output_root / "results"
    figures_dir = args.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = results_dir / "coordinate_dirichlet_forms.csv"

    if args.plots_only:
        results = pd.read_csv(raw_results_path)
    else:
        rng = np.random.default_rng(args.seed)
        frames = []
        configurations = (
            [(16, 128, int(xi), "xi") for xi in GRID]
            + [(16, int(n), FIXED_VALUE, "n") for n in GRID]
        )
        for index, (k, n, xi, path) in enumerate(configurations, start=1):
            print(
                f"Estimating path={path}, k={k}, n={n}, xi={xi} "
                f"({index}/{len(configurations)})",
                flush=True,
            )
            current = study.estimate_problem(
                rng,
                k,
                n,
                xi,
                args.pairs,
                args.batch_size,
            )
            current["path"] = path
            frames.append(current)

        results = pd.concat(frames, ignore_index=True)
        results["coordinate_role"] = np.where(
            results["coordinate"] == 1, "j = 1", "j = k"
        )
        if results["maximum_log_envelope_error"].max() >= 1e-8:
            raise RuntimeError("Ideal-line rejection-envelope validation failed")
        results.to_csv(raw_results_path, index=False)

    fit_rows = []
    outcomes = [
        (
            "ratio_S_bar_S",
            "ratio_S_bar_S_lower_95",
            "ratio_S_bar_S_upper_95",
            r"$\widehat{\mathcal{E}}_S(f_j)/\widehat{\mathcal{E}}_{\bar{S}}(f_j)$",
            "Dirichlet-form ratio",
            "E_S_to_E_bar_S",
        ),
        (
            "ratio_S_posterior_variance",
            "ratio_S_posterior_variance_lower_95",
            "ratio_S_posterior_variance_upper_95",
            r"$\widehat{\mathcal{E}}_S(f_j)/\widehat{\mathrm{var}}_\pi(f_j)$",
            "Coordinatewise Rayleigh quotient",
            "E_S_to_posterior_variance",
        ),
    ]
    for path in ("xi", "n"):
        path_data = results[results["path"] == path].copy()
        for y, lower, upper, ylabel, title_prefix, suffix in outcomes:
            title = (
                f"{title_prefix} vs "
                + (r"$\xi$" if path == "xi" else r"$n$")
            )
            if title_prefix.startswith("Coordinatewise"):
                title = title.replace(" quotient vs ", " quotient\nvs ")
            fit_rows.extend(make_plot(
                path_data,
                path,
                y,
                lower,
                upper,
                ylabel,
                title,
                figures_dir / f"endpoint_coordinates_{path}_{suffix}.png",
            ))

    results.to_csv(
        results_dir / "endpoint_coordinate_ratios_with_intervals.csv",
        index=False,
    )
    pd.DataFrame(fit_rows).to_csv(
        results_dir / "endpoint_coordinate_power_fits.csv",
        index=False,
    )
    if not args.plots_only:
        (results_dir / "configuration.txt").write_text(
            "\n".join([
                f"seed = {args.seed}",
                f"n_pairs per configuration = {args.pairs}",
                f"rows per memory-safe batch = {args.batch_size}",
                f"active coordinates q0 = {Q0}",
                "path values = " + ", ".join(map(str, GRID)),
                "xi path fixed values = k=16, n=128",
                "n path fixed values = k=16, xi=8",
                "implementation = original full-dimensional sampler",
                "S and bar S directions = conditionally independent given beta",
                "random-number generator = NumPy Generator(PCG64)",
            ]) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
