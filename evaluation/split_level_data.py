"""
Master Training Data Splitter for Mario RL Agent
=================================================
Discovers master training CSV files in outputs/, parses architecture, mode,
strategy, and seed, computes global and level cumulative step counters, and
splits into level-specific files in evaluation/<arch>/data/ for Levels 1-7
(explicitly excluding Level 8).

Supported Architectures:
- tabular: Tabular Q-Learning baseline
- hybrid_expected: Hybrid Expected SARSA model

Mode-dependent step accounting:
- Curriculum: global_cumulative_steps continues across Levels 1-7;
  level_cumulative_steps resets per level (adaptation speed).
- Separate: global_cumulative_steps == level_cumulative_steps (isolated scratch runs;
  pre-training steps = 0).

Includes seed collision check and multi-seed scalability.
"""

import argparse
import glob
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = PROJECT_ROOT / "evaluation"

ALLOWED_LEVELS = (1, 2, 3, 4, 5, 6, 7)

# Matches e.g.:
# - training_metrics_tabular_curriculum_balanced_lvl1-8_seed42.csv
# - training_metrics_hybrid_expected_separate_dense_seed2.csv
# - training_metrics_tabular_curriculum_balanced.csv
FILENAME_PATTERN = re.compile(
    r"^training_metrics_(?:(tabular|hybrid_expected)_)?(curriculum|separate)_(balanced|dense|sparse)(?:_lvl[\d-]+)?(?:_(seed\w+|\d+))?\.csv$"
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
    max_level: int = 7,
) -> Dict[str, List[Path]]:
    """
    Discovers master training metrics CSVs, parses metadata, checks for seed
    collisions, computes cumulative step metrics, and writes split per-level CSVs
    into eval_dir/<arch>/data/ for levels 1..max_level (default 1..7, excluding Level 8).
    """
    master_files = sorted(glob.glob(str(outputs_dir / "training_metrics_*.csv")))
    if not master_files:
        raise FileNotFoundError(f"No master training metrics CSV files found in {outputs_dir}")

    seen_keys: Dict[Tuple[str, str, str, str], str] = {}
    generated_by_arch: Dict[str, List[Path]] = {}

    parsed_files: List[Tuple[Path, str, str, str, str]] = []
    explicit_seed_combos = set()
    for file_path_str in master_files:
        file_path = Path(file_path_str)
        try:
            arch, mode, strategy, seed = parse_filename(file_path.name)
        except ValueError:
            print(f"  -> Skipping non-benchmark file: {file_path.name}")
            continue
        parsed_files.append((file_path, arch, mode, strategy, seed))
        if "_seed" in file_path.name:
            explicit_seed_combos.add((arch, mode, strategy))

    print(f"[Splitter] Discovered {len(parsed_files)} benchmark CSV files in {outputs_dir} (Levels 1-{max_level} only)")

    for file_path, arch, mode, strategy, seed in parsed_files:
        if (arch, mode, strategy) in explicit_seed_combos and "_seed" not in file_path.name:
            print(f"  -> Skipping legacy unseeded {file_path.name} in favor of multi-seed run files.")
            continue

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
        generated_by_arch.setdefault(arch, [])

        df = pd.read_csv(file_path)
        required_cols = {"level_id", "level_episode", "steps", "won", "total_reward"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"File {file_path.name} is missing required columns: {missing}")

        # Explicitly exclude Level 8 (and any level > max_level)
        df = df[df["level_id"].between(1, max_level)].copy()
        if df.empty:
            continue

        if "global_episode" in df.columns:
            df = df.sort_values("global_episode").reset_index(drop=True)
        else:
            df = df.sort_values(["level_id", "level_episode"]).reset_index(drop=True)

        if mode == "curriculum":
            df["global_cumulative_steps"] = df["steps"].cumsum()

        levels = sorted(int(lvl) for lvl in df["level_id"].unique())
        for lvl in levels:
            df_lvl = df[df["level_id"] == lvl].copy().reset_index(drop=True)
            df_lvl["level_cumulative_steps"] = df_lvl["steps"].cumsum()
            if mode == "separate":
                df_lvl["global_cumulative_steps"] = df_lvl["level_cumulative_steps"]

            out_filename = f"{mode}_{strategy}_level_{lvl}_{seed}.csv"
            out_filepath = target_data_dir / out_filename
            df_lvl.to_csv(out_filepath, index=False)
            generated_by_arch[arch].append(out_filepath)

    # Remove any leftover level > max_level split CSVs if present
    for arch_dir in eval_dir.iterdir() if eval_dir.exists() else []:
        data_dir = arch_dir / "data"
        if data_dir.is_dir():
            for stale in data_dir.glob("*_level_*.csv"):
                m = re.search(r"_level_(\d+)_", stale.name)
                if m and int(m.group(1)) > max_level:
                    stale.unlink()

    total_gen = sum(len(files) for files in generated_by_arch.values())
    print(f"[Splitter] Successfully generated {total_gen} split level files across {len(generated_by_arch)} architectures.")
    for arch, files in generated_by_arch.items():
        print(f"  - {arch}: {len(files)} files in {eval_dir / arch / 'data'}")

    return generated_by_arch


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Mario training metrics CSVs into per-level files (Levels 1-7).")
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
