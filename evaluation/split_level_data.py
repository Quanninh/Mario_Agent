"""
Master Training Data Splitter for Mario RL Agent
=================================================
Discovers master training CSV files in outputs/, parses mode, strategy, and seed,
computes global and level cumulative step counters, and splits into 2,000-episode
level-specific files in evaluation/data/.

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
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DATA_DIR = PROJECT_ROOT / "evaluation" / "data"

# Master file matching regex:
# e.g. training_metrics_curriculum_balanced.csv -> (curriculum, balanced, seed1)
# e.g. training_metrics_separate_dense_seed42.csv -> (separate, dense, seed42)
FILENAME_PATTERN = re.compile(
    r"^training_metrics_(curriculum|separate)_(balanced|dense|sparse)(?:_(seed\w+|\d+))?\.csv$"
)


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """Extract (mode, strategy, seed) from filename."""
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unrecognized training metrics filename pattern: {filename}")
    mode, strategy, raw_seed = match.groups()
    if raw_seed is None:
        seed = "seed1"
    elif raw_seed.startswith("seed"):
        seed = raw_seed
    else:
        seed = f"seed{raw_seed}"
    return mode, strategy, seed


def split_all_master_files(outputs_dir: Path = OUTPUTS_DIR, eval_data_dir: Path = EVAL_DATA_DIR) -> List[Path]:
    """
    Discovers all master training metrics CSVs, parses metadata,
    checks for seed collisions, computes cumulative step metrics,
    and writes split per-level CSVs into eval_data_dir.
    """
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    master_files = glob.glob(str(outputs_dir / "training_metrics_*.csv"))
    if not master_files:
        raise FileNotFoundError(f"No master training metrics CSV files found in {outputs_dir}")

    seen_keys: Dict[Tuple[str, str, str], str] = {}
    generated_files: List[Path] = []

    print(f"[Splitter] Discovered {len(master_files)} master CSV files in {outputs_dir}")

    for file_path_str in sorted(master_files):
        file_path = Path(file_path_str)
        mode, strategy, seed = parse_filename(file_path.name)
        key = (mode, strategy, seed)

        if key in seen_keys:
            raise ValueError(
                f"Duplicate master seed collision detected for {key}!\n"
                f"  Existing: {seen_keys[key]}\n"
                f"  Incoming: {file_path.name}\n"
                f"Please ensure unique seed naming in outputs/."
            )
        seen_keys[key] = file_path.name

        print(f"  -> Processing {file_path.name} | Mode: {mode}, Strategy: {strategy}, Seed: {seed}")
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
            out_filepath = eval_data_dir / out_filename
            df_lvl.to_csv(out_filepath, index=False)
            generated_files.append(out_filepath)

    print(f"[Splitter] Successfully generated {len(generated_files)} split level files in {eval_data_dir}")
    return generated_files


if __name__ == "__main__":
    split_all_master_files()
