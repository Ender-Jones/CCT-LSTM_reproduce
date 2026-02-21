"""Smoke test runner for dataset_visualization/data_mining.py.
!!! WARNING !!!
This script will delete the output directory(scripts/dataset_visualization/data_mining_smoke/) before running.
!!! WARNING !!!

This script builds a tiny temporary UBFC-Phys dataset by symlinking a small
subset of subjects from the real dataset path stored in UBFC_path.txt, then
runs the full data mining pipeline on that subset.

Goals:
1) Fast feedback loop for refactor debugging.
2) Stable small-sample regression checks.
3) Keep smoke outputs isolated from normal data_mining outputs.

When adding new plots/outputs:
- Add the new function call in run_pipeline_on_subset() with main-flow order.
- Add expected output checks (recommended: strict mode for required files).
- If new CSV files are added or changed by design, regenerate smoke_summary.json
  and treat updated csv_sha256 as the new baseline.
- If only image styling changes, prioritize file presence checks over hash checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import warnings
from pathlib import Path
import random

import pandas as pd

import data_mining as dm
import data_mining_common as dmc



def select_subject_ids(
    subject_ids: list[str],
    subject_group_map: dict[str, str],
    denominator: int,
    min_subjects: int,
) -> list[str]:
    """Select a small deterministic subset of subjects for smoke testing."""
    total = len(subject_ids)
    if total == 0:
        print(f"[SMOKE] [ERROR] No subjects to select from, returning empty list.")
        return []

    target_n = max(min_subjects, math.ceil(total / denominator))
    target_n = min(target_n, total)

    groups: dict[str, list[str]] = {}
    for subject_id in subject_ids:
        # holy pythonic
        groups.setdefault(subject_group_map.get(subject_id, "unknown"), []).append(subject_id)

    group_names = sorted(groups.keys())
    selected: list[str] = []

    # Try to include one subject per group first when possible.
    if len(group_names) >= 2 and target_n >= 2:
        for group_name in group_names:
            candidates = groups[group_name]
            if candidates:
                selected.append(random.choice(candidates))

    remaining = target_n - len(selected)
    if remaining > 0:
        rest_pool = [sid for sid in subject_ids if sid not in set(selected)]
        for _ in range(remaining):
            selected.append(random.choice(rest_pool))

    # sort the selected subject ids naturally by using integer conversion (e.g., s1, s2, ..., s56)
    selected = sorted(set(selected), key=lambda x: int(x[1:]) if x[1:].isdigit() else float('inf'))
    return selected[:target_n]


def build_smoke_dataset(
    dataset_path: Path,
    subject_group_map: dict[str, str],
    selected_subject_ids: list[str],
    temp_root: Path,
) -> tuple[Path, dict[str, str]]:
    """Create a temporary mini dataset with symlinked subject folders."""
    smoke_root = temp_root / "UBFC-Phys-Smoke"
    smoke_data_dir = smoke_root / "Data"
    smoke_data_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in selected_subject_ids:
        src = dataset_path / subject_id
        dst = smoke_data_dir / subject_id
        dst.symlink_to(src, target_is_directory=True)

    smoke_manifest = {sid: subject_group_map[sid] for sid in selected_subject_ids}
    manifest_df = pd.DataFrame(
        {"subject": list(smoke_manifest.keys()), "group": list(smoke_manifest.values())}
    )
    manifest_df.to_csv(smoke_root / "master_manifest.csv", index=False)

    return smoke_data_dir, smoke_manifest


def sha256_of_file(path: Path) -> str:
    """Compute SHA256 for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline_on_subset(smoke_data_dir: Path, smoke_manifest: dict[str, str]) -> dict:
    """Run the same pipeline order as data_mining.py main block."""
    eda_features = dm.extract_eda_features(smoke_data_dir, smoke_manifest)
    ppg_features = dm.extract_ppg_features(smoke_data_dir, smoke_manifest)
    rppg_features = dm.extract_rppg_features(smoke_data_dir, smoke_manifest)

    merged_df = dm.merge_eda_and_ppg_features(eda_features, ppg_features, rppg_features)

    dm.plot_scatter_tonic_vs_hr_hrv(merged_df)
    dm.plot_strip_phasic_by_task(merged_df)
    dm.plot_box_phasic_std_by_task(merged_df)
    dm.plot_box_rmssd_by_task(merged_df)
    dm.plot_box_hr_median_by_task(merged_df)
    dm.plot_box_rppg_rmssd_by_task(merged_df)
    dm.plot_box_interaction_feature(merged_df)
    dm.plot_3d_eda_feature_space(merged_df)
    dm.plot_strip_hrv_by_task(merged_df)

    dm.plot_subject_rolling_hrv(smoke_data_dir, smoke_manifest)
    subject_paths = dmc.list_subject_paths(smoke_data_dir)
    dm.run_batched_group_plots(subject_paths, smoke_manifest, dm.plot_group_rolling_hrv)

    dm.plot_subject_rolling_hr(smoke_data_dir, smoke_manifest)
    dm.run_batched_group_plots(subject_paths, smoke_manifest, dm.plot_group_rolling_hr)

    dm.plot_subject_bvp_profile(smoke_data_dir, smoke_manifest)
    dm.run_batched_group_plots(subject_paths, smoke_manifest, dm.plot_group_bvp_profile)

    dm.plot_subject_rolling_hr_rppg(smoke_data_dir, smoke_manifest)
    dm.run_batched_group_plots(subject_paths, smoke_manifest, dm.plot_group_rolling_hr_rppg)

    dm.analyze_rppg_vs_ppg_hr_correlation(smoke_manifest)

    return {
        "eda_records": len(eda_features),
        "ppg_records": len(ppg_features),
        "rppg_records": len(rppg_features),
        "merged_rows": int(len(merged_df)),
        "merged_columns": list(merged_df.columns),
    }


def collect_output_summary(output_dir: Path) -> dict:
    """Collect smoke output summary for baseline comparison."""
    files = sorted([p for p in output_dir.rglob("*") if p.is_file()])
    relative_files = [str(p.relative_to(output_dir)) for p in files]

    suffix_counts: dict[str, int] = {}
    for p in files:
        suffix = p.suffix.lower() or "<no_suffix>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    csv_hashes = {
        str(p.relative_to(output_dir)): sha256_of_file(p)
        for p in files
        if p.suffix.lower() == ".csv"
    }

    return {
        "output_dir": str(output_dir),
        "file_count": len(files),
        "suffix_counts": suffix_counts,
        "files": relative_files,
        "csv_sha256": csv_hashes,
    }


def initialize_smoke_output_dir(output_dir: Path) -> None:
    """Reset smoke output directory for a clean run."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[SMOKE] Cleared previous output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SMOKE] Initialized output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data_mining smoke test on a small subject subset."
    )
    parser.add_argument(
        "--denominator",
        type=int,
        default=12,
        help="Sample about total_subjects/denominator subjects (default: 12).",
    )
    parser.add_argument(
        "--min-subjects",
        type=int,
        default=2,
        help="Minimum number of subjects for smoke run (default: 2).",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="data_mining_smoke",
        help="Output directory name under scripts/dataset_visualization.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="smoke_summary.json",
        help="Summary JSON filename inside the smoke output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="EDA signal is sampled at very low frequency")

    dataset_path = dmc.read_pathfile_or_ask_for_path()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    subject_group_map = dmc.read_master_manifest(dataset_path)
    subject_paths = dmc.list_subject_paths(dataset_path)
    sorted_subject_ids = [path.name for path in subject_paths]

    selected_subject_ids = select_subject_ids(
        subject_ids=sorted_subject_ids,
        subject_group_map=subject_group_map,
        denominator=max(1, args.denominator),
        min_subjects=max(1, args.min_subjects),
    )
    if not selected_subject_ids:
        raise ValueError("No subjects selected for smoke test.")

    print(f"[SMOKE] Selected subjects ({len(selected_subject_ids)}): {selected_subject_ids}")

    script_dir = Path(__file__).resolve().parent
    smoke_output_dir = script_dir / args.output_dir_name
    initialize_smoke_output_dir(smoke_output_dir)

    original_output_dir = dmc.DATA_MINING_OUTPUT_DIR
    dmc.DATA_MINING_OUTPUT_DIR = smoke_output_dir

    try:
        with tempfile.TemporaryDirectory(prefix="ubfc_smoke_") as temp_dir:
            smoke_data_dir, smoke_manifest = build_smoke_dataset(
                dataset_path=dataset_path,
                subject_group_map=subject_group_map,
                selected_subject_ids=selected_subject_ids,
                temp_root=Path(temp_dir),
            )
            run_stats = run_pipeline_on_subset(smoke_data_dir, smoke_manifest)

        output_summary = collect_output_summary(smoke_output_dir)
        summary = {
            "selected_subject_ids": selected_subject_ids,
            "dataset_path": str(dataset_path),
            "smoke_data_note": "Temporary symlink dataset generated during runtime",
            "run_stats": run_stats,
            "output_summary": output_summary,
        }

        summary_path = smoke_output_dir / args.summary_name
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SMOKE] Summary saved to {summary_path}")
    finally:
        dmc.DATA_MINING_OUTPUT_DIR = original_output_dir


if __name__ == "__main__":
    main()
