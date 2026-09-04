"""Regenerate the three parameter paths and all six Section 7 figures.

The two component runners intentionally retain their original independent
seeds so a full run reproduces the supplied raw results exactly:

* k path: seed 8172028, (n, xi) = (128, 8);
* xi and n paths: seed 8172030, with (k, n) = (16, 128) and
  (k, xi) = (16, 8), respectively.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
K_PATH_SEED = 8_172_028
XI_N_PATHS_SEED = 8_172_030
PATH_VALUES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def run_component(script: Path, output_root: Path, options: list[str]) -> None:
    command = [sys.executable, str(script), "--output-root", str(output_root), *options]
    subprocess.run(command, cwd=HERE, check=True)


def copy_figures(output_root: Path) -> None:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sources = {
        HERE / "path_outputs" / "k_path" / "figures"
        / "endpoint_coordinates_k_E_S_to_E_bar_S_xi8.png":
            "endpoint_coordinates_k_E_S_to_E_bar_S.png",
        HERE / "path_outputs" / "k_path" / "figures"
        / "endpoint_coordinates_k_E_S_to_posterior_variance_xi8.png":
            "endpoint_coordinates_k_E_S_to_posterior_variance.png",
        HERE / "path_outputs" / "xi_n_paths" / "figures"
        / "endpoint_coordinates_xi_E_S_to_E_bar_S.png":
            "endpoint_coordinates_xi_E_S_to_E_bar_S.png",
        HERE / "path_outputs" / "xi_n_paths" / "figures"
        / "endpoint_coordinates_xi_E_S_to_posterior_variance.png":
            "endpoint_coordinates_xi_E_S_to_posterior_variance.png",
        HERE / "path_outputs" / "xi_n_paths" / "figures"
        / "endpoint_coordinates_n_E_S_to_E_bar_S.png":
            "endpoint_coordinates_n_E_S_to_E_bar_S.png",
        HERE / "path_outputs" / "xi_n_paths" / "figures"
        / "endpoint_coordinates_n_E_S_to_posterior_variance.png":
            "endpoint_coordinates_n_E_S_to_posterior_variance.png",
    }
    for source, filename in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Expected generated figure is missing: {source}")
        shutil.copy2(source, figures / filename)


def combine_results(
    output_root: Path,
    pairs: int,
    batch_size: int,
    write_configuration: bool,
) -> None:
    """Rebuild the consolidated tables and configuration from path outputs."""
    k_results_dir = output_root / "path_outputs" / "k_path" / "results"
    xi_n_results_dir = output_root / "path_outputs" / "xi_n_paths" / "results"
    combined_results_dir = output_root / "results"
    combined_results_dir.mkdir(parents=True, exist_ok=True)

    k_results = pd.read_csv(k_results_dir / "xi8_coordinate_dirichlet_forms.csv")
    k_results["path"] = "k"
    k_results["simulation_seed"] = K_PATH_SEED

    xi_n_results = pd.read_csv(xi_n_results_dir / "coordinate_dirichlet_forms.csv")
    xi_n_results["simulation_seed"] = XI_N_PATHS_SEED
    xi_n_results = xi_n_results.reindex(columns=k_results.columns)

    pd.concat([k_results, xi_n_results], ignore_index=True).to_csv(
        combined_results_dir / "combined_coordinate_dirichlet_forms.csv",
        index=False,
    )

    k_fits = pd.read_csv(k_results_dir / "xi8_power_fits.csv")
    k_fits["path"] = "k"
    xi_n_fits = pd.read_csv(
        xi_n_results_dir / "endpoint_coordinate_power_fits.csv"
    ).reindex(columns=k_fits.columns)
    pd.concat([k_fits, xi_n_fits], ignore_index=True).to_csv(
        combined_results_dir / "combined_power_fits.csv",
        index=False,
    )

    if write_configuration:
        (combined_results_dir / "configuration.txt").write_text(
            "\n".join([
                "active coordinates q0 = 8",
                "path values = " + ", ".join(map(str, PATH_VALUES)),
                "k path = vary k with n=128 and xi=8",
                "xi path = vary xi with k=16 and n=128",
                "n path = vary n with k=16 and xi=8",
                f"n_pairs per configuration = {pairs}",
                f"rows per memory-safe batch = {batch_size}",
                f"k path seed = {K_PATH_SEED}",
                f"xi and n paths seed = {XI_N_PATHS_SEED}",
                "random-number generator = NumPy Generator(PCG64)",
                "S and bar S directions = conditionally independent given beta",
                "implementation = original full-dimensional sampler",
            ]) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="regenerate all figures from the supplied raw-results CSV files",
    )
    args = parser.parse_args()

    common_options = [
        "--pairs", str(args.pairs),
        "--batch-size", str(args.batch_size),
    ]
    if args.plots_only:
        common_options.append("--plots-only")

    run_component(
        HERE / "run_k_path.py",
        HERE / "path_outputs" / "k_path",
        common_options,
    )
    run_component(
        HERE / "run_xi_n_paths.py",
        HERE / "path_outputs" / "xi_n_paths",
        common_options,
    )
    copy_figures(HERE)
    combine_results(
        HERE,
        args.pairs,
        args.batch_size,
        write_configuration=not args.plots_only,
    )
    print(f"All six figures are in {(HERE / 'figures').resolve()}")
    print(f"Consolidated results are in {(HERE / 'results').resolve()}")


if __name__ == "__main__":
    main()
