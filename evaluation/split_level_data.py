"""
Master Training Data Splitter for Mario RL Agent
=================================================
Discovers master training CSV files in outputs/, parses architecture, mode,
strategy, and seed, computes global and level cumulative step counters, and
splits into 2,000-episode level-specific files in evaluation/<arch>/data/.

Supported Architectures:
- tabular: Tabular Q-Learning baseline
- hybrid_expected: Hybrid Expected SARSA model

Mode-dependent step accounting:
- Curriculum: global_cumulative_steps continues across all 10,000 master episodes;
  level_cumulative_steps resets per level (adaptation speed).
- Separate: global_cumulative_steps == level_cumulative_steps (isolated scratch runs;
  pre-training steps = 0).

Includes seed collision check and multi-seed scalability.
"""

import os
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = PROJECT_ROOT / "evaluation"

# Master file matching regex:
# e.g. training_metrics_tabular_curriculum_balanced.csv -> (tabular, curriculum, balanced, seed1)
# e.g. training_metrics_hybrid_expected_separate_dense_seed2.csv -> (hybrid_expected, separate, dense, seed2)
# e.g. training_metrics_curriculum_balanced.csv -> (tabular [default], curriculum, balanced, seed1)
FILENAME_PATTERN = re.compile(
    r"^training_metrics_(?:(tabular|hybrid_expected)_)?(curriculum|separate)_(balanced|dense|sparse)(?:_(seed\w+|\d+))?\.csv$"
)


def parse_filename(filename: str) -> Tuple[str, str, str, str]:
    """Extract (architecture, mode, strategy, seed) from filename."""
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unrecognized training metrics filename pattern: {filename}")
    raw_arch, mode, strategy, raw_seed = match.groups()
    arch = raw_arch if raw_arch else "tabular"
    if raw_seed is None:
        seed = "seed1"
    elif raw_seed.startswith("seed"):
        seed = raw_seed
    else:
        seed = f"seed{raw_seed}"
    return arch, mode, strategy, seed


def split_all_master_files(
    outputs_dir: Path = OUTPUTS_DIR,
    eval_dir: Path = EVAL_DIR,
    arch_filter: str = "all",
) -> Dict[str, List[Path]]:
    """
    Discovers all master training metrics CSVs, parses metadata,
    checks for seed collisions, computes cumulative step metrics,
    and writes split per-level CSVs into eval_dir/<arch>/data/.
    """
    master_files = glob.glob(str(outputs_dir / "training_metrics_*.csv"))
    if not master_files:
        raise FileNotFoundError(f"No master training metrics CSV files found in {outputs_dir}")

    seen_keys: Dict[Tuple[str, str, str, str], str] = {}
    generated_by_arch: Dict[str, List[Path]] = {}

    print(f"[Splitter] Discovered {len(master_files)} master CSV files in {outputs_dir}")

    for file_path_str in sorted(master_files):
        file_path = Path(file_path_str)
        arch, mode, strategy, seed = parse_filename(file_path.name)
        key = (arch, mode, strategy, seed)

        if key in seen_keys:
            raise ValueError(
                f"Duplicate master seed collision detected for {key}!\n"
                f"  Existing: {seen_keys[key]}\n"
                f"  Incoming: {file_path.name}\n"
                f"Please ensure unique seed naming in outputs/."
            )
        seen_keys[key] = file_path.name

        if arch_filter != "all" and arch != arch_filter:
            continue

        target_data_dir = eval_dir / arch / "data"
        target_data_dir.mkdir(parents=True, exist_ok=True)

        if arch not in generated_by_arch:
            generated_by_arch[arch] = []

        print(f"  -> Processing {file_path.name} | Arch: {arch}, Mode: {mode}, Strategy: {strategy}, Seed: {seed}")
        df = pd.read_csv(file_path)

        # Validate required columns
        required_cols = {"level_id", "level_episode", "steps", "won", "total_reward"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"File {file_path.name} is missing required columns: {missing}")

        # Ensure correct episode ordering
        if "global_episode" in df.columns:
            df = df.sort_values("global_episode").reset_index(drop=True)
        else:
            df = df.sort_values(["level_id", "level_episode"]).reset_index(drop=True)

        # Mode-dependent step accumulation:
        # Curriculum: global_cumulative_steps is continuous across all 10,000 episodes
        if mode == "curriculum":
            df["global_cumulative_steps"] = df["steps"].cumsum()

        # Split into level-specific files
        levels = sorted(df["level_id"].unique())
        for lvl in levels:
            df_lvl = df[df["level_id"] == lvl].copy().reset_index(drop=True)

            # Within-level adaptation steps: resets to 0 at the start of each level
            df_lvl["level_cumulative_steps"] = df_lvl["steps"].cumsum()

            # Separate mode: each level is trained cold with no pre-training
            if mode == "separate":
                df_lvl["global_cumulative_steps"] = df_lvl["level_cumulative_steps"]

            out_filename = f"{mode}_{strategy}_level_{lvl}_{seed}.csv"
            out_filepath = target_data_dir / out_filename
            df_lvl.to_csv(out_filepath, index=False)
            generated_by_arch[arch].append(out_filepath)

    total_gen = sum(len(files) for files in generated_by_arch.values())
    print(f"[Splitter] Successfully generated {total_gen} split level files across {len(generated_by_arch)} architectures.")
    for arch, files in generated_by_arch.items():
        print(f"  - {arch}: {len(files)} files in {eval_dir / arch / 'data'}")

    return generated_by_arch


def main():
    parser = argparse.ArgumentParser(description="Split Mario training metrics CSVs into per-level files.")
    parser.add_argument(
        "--arch",
        choices=["all", "tabular", "hybrid_expected"],
        default="all",
        help="Architecture to process (default: all)",
    )
    args = parser.parse_args()
    split_all_master_files(arch_filter=args.arch)


if __name__ == "__main__":
    main()
