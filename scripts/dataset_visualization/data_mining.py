"""EDA Data Mining - Explore relationships between EDA tonic and HR/HRV metrics.

Sprint #1 Goals:
- Extract EDA tonic and HR/HRV features for each (subject, task) pair from UBFC-Phys
- Plot scatter plots with tonic_mean on X-axis and HR/HRV metrics on Y-axis
- (Future) Apply clustering algorithms to find data patterns

Data Pipeline:
1. Iterate over all subjects (s1~s56) and tasks (T1/T2/T3)
2. For each (subject, task) extract:
   - EDA: tonic_mean, tonic_median (from eda_sX_TY.csv, 4Hz)
   - HR/HRV: hr_mean, hrv_sdnn, hrv_rmssd (from bvp_sX_TY.csv, 64Hz)
3. Merge into single DataFrame keyed by (subject, task)
4. Generate visualizations and save to data_mining/ directory
5. Export merged feature table as CSV for downstream analysis

Output Directory: scripts/dataset_visualization/data_mining/
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.stats import linregress

import data_mining_common as dmc

# =============================================================================
# Output Filenames
# =============================================================================

OUT_CSV_MERGED = "merged_features.csv"
OUT_SCATTER_TONIC_VS_HR = "scatter_tonic_mean_vs_hr_mean.jpg"
OUT_SCATTER_TONIC_VS_SDNN = "scatter_tonic_mean_vs_hrv_sdnn.jpg"
OUT_SCATTER_TONIC_VS_RMSSD = "scatter_tonic_mean_vs_hrv_rmssd.jpg"
OUT_STRIP_PHASIC = "strip_task_vs_phasic_var.jpg"
OUT_STRIP_SDNN = "strip_task_vs_hrv_sdnn.jpg"
OUT_STRIP_RMSSD = "strip_task_vs_hrv_rmssd.jpg"
OUT_BOX_PHASIC_STD = "box_task_vs_phasic_std.jpg"
OUT_BOX_RMSSD = "box_task_vs_hrv_rmssd.jpg"
OUT_BOX_RPPG_RMSSD = "box_task_vs_rppg_hrv_rmssd.jpg"
OUT_BOX_HR_MEDIAN = "box_task_vs_hr_median.jpg"
OUT_BOX_INTERACTION = "box_Tonic_slop_x_Phasic_std.jpg"
OUT_3D_EDA = "scatter3d_tonic_phasic_slope.html"

# Outlier removal threshold (percentile)
OUTLIER_PERCENTILE = 90

# Rolling profile parameters for HR/HRV line plots
ROLLING_WINDOW_SEC = 3
# Stride is defined in BVP samples to avoid mixing with "Hz" terminology.
ROLLING_STRIDE_SAMPLES = 1
ROLLING_BVP_STEP_SEC = ROLLING_STRIDE_SAMPLES / dmc.BVP_SAMPLING_RATE_HZ
# Rolling profile export DPI
ROLLING_SUBJECT_PLOT_DPI = 200
ROLLING_GROUP_PLOT_DPI = 300


def make_task_label(task_id: str, group: str) -> str:
    """Generate combined task label for 5-class classification.

    T1 stays as-is (baseline rest is same for both groups).
    T2/T3 get group suffix to distinguish easy vs hard tasks.

    Args:
        task_id: Task identifier ('T1', 'T2', or 'T3').
        group: Subject group ('ctrl' or 'test').

    Returns:
        Combined label string. 'T1' for baseline, '{task_id}-{group}' for others.

    Outputs:
        None.

    Examples:
        >>> make_task_label('T1', 'ctrl')
        'T1'
        >>> make_task_label('T2', 'ctrl')
        'T2-ctrl'
        >>> make_task_label('T3', 'test')
        'T3-test'
    """
    if task_id == 'T1':
        return 'T1'
    return f"{task_id}-{group}"


def extract_eda_features(dataset_path: Path, subject_group_map: dict[str, str]) -> list[dict]:
    """Extract EDA tonic and phasic features for each (subject, task) pair.

    Reads EDA CSV files (4Hz sampling rate), applies nk.eda_process() to decompose
    into tonic and phasic components, and computes summary statistics including
    tonic slope via linear regression.

    Args:
        dataset_path: Path to UBFC-Phys/Data folder.
        subject_group_map: Mapping of {subject_id: group} from master_manifest.csv.

    Returns:
        List of dicts, each containing:
            - subject, task, group, task_label
            - tonic_mean, tonic_median, tonic_slope
            - phasic_var, phasic_std

    Outputs:
        None (no files saved).
    """
    result: list[dict] = []

    for subject_path in dmc.list_subject_paths(dataset_path):
        subject_id = subject_path.name  # e.g. 's1'
        group = subject_group_map.get(subject_id, 'unknown')
        eda_paths = dmc.list_eda_paths(subject_path)

        if not dmc.has_expected_eda_files(eda_paths, subject_path):
            continue

        for eda_path in eda_paths:
            # task: eda_s1_T1.csv -> T1
            task_id = eda_path.stem.split('_')[-1]
            task_label = make_task_label(task_id, group)

            # check if the eda data is too short
            eda_data = dmc.read_eda_data(eda_path)
            if len(eda_data) < 10:
                print(f"[WARN] {eda_path}: EDA data too short ({len(eda_data)} samples). Skipping.")
                continue

            processed_signal, _ = nk.eda_process(eda_data, sampling_rate=dmc.EDA_SAMPLING_RATE_HZ)
            tonic = processed_signal['EDA_Tonic']
            phasic = processed_signal['EDA_Phasic']

            # Compute tonic slope via linear regression (trend over time)
            x = np.arange(len(tonic))
            slope, _, _, _, _ = linregress(x, tonic)

            result.append({
                'subject': subject_id,
                'task': task_id,
                'group': group,
                'task_label': task_label,
                'tonic_mean': tonic.mean(),
                'tonic_median': tonic.median(),
                'phasic_var': phasic.var(),
                'phasic_std': phasic.std(),
                'tonic_slope': slope,
            })

    print(f"[INFO] Extracted EDA tonic features from {len(result)} records.")
    return result


def clean_rr_intervals_from_peaks(
    peaks: list[int] | np.ndarray,
    sampling_rate_hz: float,
    *,
    interval_min_ms: float = 250.0,
    interval_max_ms: float = 1300.0,
    min_valid_ratio: float = 0.8,
    min_valid_intervals: int = 10,
    fixpeaks_method: str = "Kubios",
    iterative: bool = True,
) -> tuple[dict[str, np.ndarray] | None, dict[str, float]]:
    """Generate 'cleaned RR/NN interval series' (with interpolation) from peak positions for HRV calculation.

    Processing pipeline:
    1) Use nk.signal_fixpeaks(method="Kubios") to correct missed/extra/misaligned peaks (iterative).
    2) Calculate RR intervals (ms).
    3) Mark invalid RRs using absolute physiological gating [interval_min_ms, interval_max_ms].
    4) Replace invalid RRs with linear interpolation (avoids dropping points which causes sample shortage/gaps).
    5) If 'valid RRs' are too few (insufficient ratio or count), return None.

    Args:
        peaks: Peak positions (sample indices), monotonically increasing.
        sampling_rate_hz: Sampling rate (Hz).
        interval_min_ms: Minimum RR threshold (ms), default 250ms.
        interval_max_ms: Maximum RR threshold (ms), default 1300ms.
        min_valid_ratio: Minimum ratio of valid RRs required before gating.
        min_valid_intervals: Minimum number of valid RRs required before gating.
        fixpeaks_method: Method name for nk.signal_fixpeaks, default "Kubios".
        iterative: Whether to use iterative correction (recommended True for Kubios).

    Returns:
        (rri_dict, qc_summary)
        - rri_dict: Dictionary compatible with nk.hrv_time(), containing:
            - "RRI": RR intervals (ms)
            - "RRI_Time": RR timestamps (seconds, at the time of the second peak of each RR)
          Returns None if valid RRs are insufficient.
        - qc_summary: Quality control summary (for logging/tuning).
    """
    qc: dict[str, float] = {
        "n_peaks_in": float(len(peaks)),
        "n_peaks_fixed": 0.0,
        "n_rr_total": 0.0,
        "n_rr_valid": 0.0,
        "valid_ratio": 0.0,
    }

    if sampling_rate_hz is None or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        return None, qc

    peaks_arr = np.asarray(peaks, dtype=float)
    peaks_arr = peaks_arr[np.isfinite(peaks_arr)]
    peaks_arr = peaks_arr.astype(int, copy=False)
    if peaks_arr.size < 5:
        qc["n_peaks_fixed"] = float(peaks_arr.size)
        return None, qc

    peaks_arr = np.unique(peaks_arr)
    peaks_arr.sort(kind="mergesort")
    if peaks_arr.size < 5:
        qc["n_peaks_fixed"] = float(peaks_arr.size)
        return None, qc

    try:
        if fixpeaks_method.lower() == "kubios":
            artifacts, peaks_fixed = nk.signal_fixpeaks(
                peaks_arr,
                sampling_rate=float(sampling_rate_hz),
                iterative=iterative,
                method="Kubios",
            )
            _ = artifacts  # artifacts can be used for further debugging/visualization
        else:
            peaks_fixed = nk.signal_fixpeaks(
                peaks_arr,
                sampling_rate=float(sampling_rate_hz),
                iterative=iterative,
                method=fixpeaks_method,
            )
    except Exception:
        return None, qc

    peaks_fixed = np.asarray(peaks_fixed, dtype=float)
    peaks_fixed = peaks_fixed[np.isfinite(peaks_fixed)].astype(int, copy=False)
    peaks_fixed = np.unique(peaks_fixed)
    peaks_fixed.sort(kind="mergesort")
    qc["n_peaks_fixed"] = float(peaks_fixed.size)
    if peaks_fixed.size < 5:
        return None, qc

    # RR intervals (ms) and timestamps (s at 2nd peak)
    rri_ms = np.diff(peaks_fixed) / float(sampling_rate_hz) * 1000.0
    rri_time_s = peaks_fixed[1:] / float(sampling_rate_hz)

    qc["n_rr_total"] = float(rri_ms.size)
    if rri_ms.size < 2:
        return None, qc

    valid = np.isfinite(rri_ms) & (rri_ms >= interval_min_ms) & (rri_ms <= interval_max_ms)
    n_valid = int(np.sum(valid))
    qc["n_rr_valid"] = float(n_valid)
    qc["valid_ratio"] = float(n_valid / rri_ms.size)

    # Too few valid RRs: discard this record
    if n_valid < min_valid_intervals or (n_valid / rri_ms.size) < min_valid_ratio:
        return None, qc

    # Replace invalid RRs with linear interpolation (avoid dropping points causing sample shortage)
    rri_interp = rri_ms.astype(float, copy=True)
    invalid_idx = np.where(~valid)[0]
    if invalid_idx.size > 0:
        valid_idx = np.where(valid)[0]
        if valid_idx.size < 2:
            return None, qc
        interp_values = np.interp(np.arange(rri_ms.size), valid_idx, rri_ms[valid_idx])
        rri_interp[~valid] = interp_values[~valid]
        # Clip as a fallback to ensure interpolated values are within bounds
        rri_interp = np.clip(rri_interp, interval_min_ms, interval_max_ms)

    rri_dict = {
        "RRI": rri_interp,
        "RRI_Time": rri_time_s.astype(float, copy=False),
    }
    return rri_dict, qc


def extract_ppg_features(dataset_path: Path, subject_group_map: dict[str, str]) -> list[dict]:
    """Extract HR and HRV features from PPG/BVP signals for each (subject, task) pair.

    Reads BVP CSV files (64Hz sampling rate), applies nk.ppg_process() to extract
    heart rate, and computes HRV time-domain metrics from detected peaks.

    Args:
        dataset_path: Path to UBFC-Phys/Data folder.
        subject_group_map: Mapping of {subject_id: group} from master_manifest.csv.

    Returns:
        List of dicts, each containing:
            - subject, task, group, task_label
            - hr_mean, hr_median, hrv_sdnn, hrv_rmssd

    Outputs:
        None (no files saved).
    """
    result: list[dict] = []

    # process each subject
    for subject_path in dmc.list_subject_paths(dataset_path):
        subject_id = subject_path.name  # e.g. 's1'
        group = subject_group_map.get(subject_id, 'unknown')
        bvp_paths = dmc.list_bvp_paths(subject_path)

        if not dmc.has_expected_bvp_files(bvp_paths, subject_path):
            continue

        # process each bvp file
        for bvp_path in bvp_paths:
            # bvp_s1_T1.csv -> T1
            task_id = bvp_path.stem.split('_')[-1]
            task_label = make_task_label(task_id, group)

            bvp_data = dmc.read_bvp_data(bvp_path)
            if len(bvp_data) < 320:  # at least 5 seconds of data
                print(f"[WARN] {bvp_path}: BVP data too short ({len(bvp_data)} samples). Skipping.")
                continue

            try:
                # process the bvp data
                signals, info = nk.ppg_process(bvp_data, sampling_rate=dmc.BVP_SAMPLING_RATE_HZ)

                # get the peaks for hrv calculation
                peaks = info['PPG_Peaks']
                if len(peaks) < 5:
                    print(f"[WARN] {bvp_path}: Not enough peaks ({len(peaks)}) for HRV. Skipping.")
                    continue

                rri_dict, qc = clean_rr_intervals_from_peaks(
                    peaks,
                    dmc.BVP_SAMPLING_RATE_HZ,
                    interval_min_ms=250.0,
                    interval_max_ms=1300.0,
                    min_valid_ratio=0.8,
                    min_valid_intervals=10,
                    fixpeaks_method="Kubios",
                    iterative=True,
                )
                if rri_dict is None:
                    print(
                        f"[WARN] {bvp_path}: RR QC failed "
                        f"(valid_ratio={qc['valid_ratio']:.2f}, valid_rr={int(qc['n_rr_valid'])}/{int(qc['n_rr_total'])}). Skipping."
                    )
                    continue

                # hr mean from cleaned RR (bpm)
                hr_bpm_series = 60000.0 / np.asarray(rri_dict["RRI"], dtype=float)
                hr_mean = float(np.nanmean(hr_bpm_series))
                hr_median = float(np.nanmedian(hr_bpm_series))

                # calculate the hrv
                hrv_df = nk.hrv_time(rri_dict, sampling_rate=dmc.BVP_SAMPLING_RATE_HZ)
                hrv_sdnn = hrv_df['HRV_SDNN'].iloc[0]
                hrv_rmssd = hrv_df['HRV_RMSSD'].iloc[0]

                result.append({
                    'subject': subject_id,
                    'task': task_id,
                    'group': group,
                    'task_label': task_label,
                    'hr_mean': hr_mean,
                    'hr_median': hr_median,
                    'hrv_sdnn': hrv_sdnn,
                    'hrv_rmssd': hrv_rmssd,
                })
            except Exception as e:
                print(f"[WARN] {bvp_path}: Failed to process BVP: {e}. Skipping.")
                continue

    print(f"[INFO] Extracted HR/HRV features from {len(result)} records.")
    return result


def extract_rppg_features(dataset_path: Path, subject_group_map: dict[str, str]) -> list[dict]:
    """Extract HRV RMSSD features from rPPG signals for each (subject, task) pair.

    Reads rPPG JSON files from each subject's rppg_signals directory and computes
    HRV RMSSD per window, then aggregates by mean for the task.

    Args:
        dataset_path: Path to UBFC-Phys/Data folder.
        subject_group_map: Mapping of {subject_id: group} from master_manifest.csv.

    Returns:
        List of dicts, each containing:
            - subject, task, group, task_label
            - rppg_hrv_rmssd
    """
    result: list[dict] = []

    for subject_path in dmc.list_subject_paths(dataset_path):
        subject_id = subject_path.name
        group = subject_group_map.get(subject_id, 'unknown')
        rppg_paths = dmc.list_rppg_paths(subject_path)

        if len(rppg_paths) != 3:
            print(
                f"[WARN] {subject_path}: expected 3 rPPG files (T1/T2/T3), got {len(rppg_paths)}. Skipping."
            )
            continue

        for rppg_path in rppg_paths:
            # vid_s1_T1_rppg.json -> T1
            task_id = rppg_path.stem.split('_')[2]
            task_label = make_task_label(task_id, group)

            try:
                with rppg_path.open("r", encoding="utf-8") as f:
                    rppg_data = json.load(f)
            except Exception as e:
                print(f"[WARN] {rppg_path}: Failed to read rPPG JSON: {e}. Skipping.")
                continue

            window_rmssd: list[float] = []
            for window in rppg_data:
                bvp_signal = window.get('bvp_signal', [])
                fps = window.get('fps=sampling_rate', window.get('fps'))

                if not bvp_signal or fps is None:
                    continue

                try:
                    fs = float(fps)
                    if not np.isfinite(fs) or fs <= 0:
                        continue
                    fs_int = int(round(fs))
                    signals, info = nk.ppg_process(bvp_signal, sampling_rate=fs_int)
                    peaks = info['PPG_Peaks']
                    if len(peaks) < 5:
                        continue

                    rri_dict, _qc = clean_rr_intervals_from_peaks(
                        peaks,
                        fs_int,
                        interval_min_ms=250.0,
                        interval_max_ms=1300.0,
                        min_valid_ratio=0.8,
                        min_valid_intervals=10,
                        fixpeaks_method="Kubios",
                        iterative=True,
                    )
                    if rri_dict is None:
                        continue

                    hrv_df = nk.hrv_time(rri_dict, sampling_rate=fs_int)
                    rmssd = hrv_df['HRV_RMSSD'].iloc[0]
                    if np.isfinite(rmssd):
                        window_rmssd.append(rmssd)
                except Exception:
                    continue

            if not window_rmssd:
                print(f"[WARN] {rppg_path}: No valid rPPG windows for HRV. Skipping.")
                continue

            result.append({
                'subject': subject_id,
                'task': task_id,
                'group': group,
                'task_label': task_label,
                'rppg_hrv_rmssd': float(np.mean(window_rmssd)),
            })

    print(f"[INFO] Extracted rPPG HRV features from {len(result)} records.")
    return result


def merge_eda_and_ppg_features(
    eda_features: list[dict],
    ppg_features: list[dict],
    rppg_features: list[dict] | None = None,
) -> pd.DataFrame:
    """Merge EDA, PPG, and rPPG feature lists into a single DataFrame.

    Performs inner join on (subject, task, group, task_label) keys and saves
    the merged DataFrame to CSV for downstream analysis.

    Args:
        eda_features: List of dicts from extract_eda_features().
        ppg_features: List of dicts from extract_ppg_features().
        rppg_features: Optional list of dicts from extract_rppg_features().

    Returns:
        Merged DataFrame with columns:
            [subject, task, group, task_label, tonic_mean, tonic_median, tonic_slope,
             phasic_var, phasic_std, hr_mean, hr_median, hrv_sdnn, hrv_rmssd, rppg_hrv_rmssd]

    Outputs:
        Saves to DATA_MINING_OUTPUT_DIR:
            - merged_features.csv
    """
    eda_df = pd.DataFrame(eda_features)
    hr_df = pd.DataFrame(ppg_features)

    merged_df = pd.merge(
        eda_df, hr_df,
        on=['subject', 'task', 'group', 'task_label'],
        how='inner'
    )
    print(f"[INFO] Merged {len(merged_df)} records (EDA: {len(eda_df)}, HR: {len(hr_df)}).")

    if rppg_features is not None:
        rppg_df = pd.DataFrame(rppg_features)
        if not rppg_df.empty:
            merged_df = pd.merge(
                merged_df, rppg_df,
                on=['subject', 'task', 'group', 'task_label'],
                how='left'
            )
            print(f"[INFO] Added rPPG HRV features (rPPG: {len(rppg_df)}).")

    # Save merged CSV
    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_CSV_MERGED
    merged_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved {csv_path}")

    return merged_df


def plot_scatter_tonic_vs_hr_hrv(merged_df: pd.DataFrame) -> None:
    """Plot scatter plots of EDA tonic mean vs HR/HRV metrics with 5-class coloring.

    Creates three scatter plots showing relationship between EDA tonic mean (X-axis)
    and cardiovascular metrics (HR, HRV SDNN, HRV RMSSD) on Y-axis. Points are
    colored and shaped by task_label. Outliers above 98th percentile are excluded
    and annotated below the plot.

    Color scheme:
        - T1: green (baseline rest)
        - T2-ctrl/T3-ctrl: light blue/red (easy task, control group)
        - T2-test/T3-test: dark blue/red (hard task, test group)

    Args:
        merged_df: DataFrame from merge_eda_and_ppg_features().

    Returns:
        None.

    Outputs:
        Saves to DATA_MINING_OUTPUT_DIR:
            - scatter_tonic_mean_vs_hr_mean.jpg
            - scatter_tonic_mean_vs_hrv_sdnn.jpg
            - scatter_tonic_mean_vs_hrv_rmssd.jpg
    """
    if merged_df.empty:
        print("[WARN] No data available. Skipping plots.")
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5-class color palette: same hue family, different lightness for ctrl/test
    task_palette = {
        'T1':      '#4CAF50',  # green - baseline
        'T2-ctrl': '#90CAF9',  # light blue - easy task
        'T2-test': '#1565C0',  # dark blue - hard task
        'T3-ctrl': '#EF9A9A',  # light red - easy task
        'T3-test': '#C62828',  # dark red - hard task
    }

    # Marker mapping: ctrl=circle, test=triangle, T1=circle
    marker_map = {
        'T1':      'o',
        'T2-ctrl': 'o',
        'T2-test': '^',
        'T3-ctrl': 'o',
        'T3-test': '^',
    }

    # Legend order for logical grouping
    hue_order = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']

    # Plot configs: (y_column, y_label, output_filename)
    plot_configs = [
        ('hr_mean', 'HR Mean (bpm)', OUT_SCATTER_TONIC_VS_HR),
        ('hrv_sdnn', 'HRV SDNN (ms)', OUT_SCATTER_TONIC_VS_SDNN),
        ('hrv_rmssd', 'HRV RMSSD (ms)', OUT_SCATTER_TONIC_VS_RMSSD),
    ]

    for y_col, y_label, filename in plot_configs:
        # Calculate Y-axis upper limit at 98 percentile
        y_upper_limit = merged_df[y_col].quantile(OUTLIER_PERCENTILE / 100)

        # Split data into normal points and outliers
        plot_df = merged_df[merged_df[y_col] <= y_upper_limit]
        outliers = merged_df[merged_df[y_col] > y_upper_limit]

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.scatterplot(
            data=plot_df,
            x='tonic_mean',
            y=y_col,
            hue='task_label',
            style='task_label',
            palette=task_palette,
            markers=marker_map,
            hue_order=hue_order,
            style_order=hue_order,
            s=100,
            alpha=0.75,
            edgecolor='white',
            linewidth=0.5,
            ax=ax,
        )
        ax.set_xlabel('EDA Tonic Mean (µS)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'EDA Tonic vs {y_label}', fontsize=14, fontweight='bold')
        ax.legend(title='Task-Group', bbox_to_anchor=(1.02, 1), loc='upper left')

        # Add outlier annotation below the plot if any outliers exist
        if not outliers.empty:
            outlier_lines = [f"Outliers (>{OUTLIER_PERCENTILE}%ile, Y>{y_upper_limit:.0f}):"]
            for _, row in outliers.iterrows():
                outlier_lines.append(
                    f"  {row['subject']} ({row['task_label']}): {row[y_col]:.1f}"
                )
            outlier_text = "\n".join(outlier_lines)

            # Place annotation below the plot (outside axes)
            fig.text(
                0.12, -0.02,
                outlier_text,
                fontsize=9,
                fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='#FFB300', alpha=0.9),
            )

        plt.tight_layout()

        output_path = dmc.DATA_MINING_OUTPUT_DIR / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved {output_path}")


def plot_strip_phasic_by_task(merged_df: pd.DataFrame) -> None:
    """Plot strip plot showing EDA phasic variance distribution by task_label.

    Creates a horizontal strip plot with task_label on Y-axis and phasic_var
    on X-axis. Outliers above 98th percentile are excluded and annotated below
    the plot.

    Args:
        merged_df: DataFrame from merge_eda_and_ppg_features(), must contain
            'phasic_var' column.

    Returns:
        None.

    Outputs:
        Saves to DATA_MINING_OUTPUT_DIR:
            - strip_task_vs_phasic_var.jpg
    """
    if merged_df.empty or 'phasic_var' not in merged_df.columns:
        print("[WARN] No phasic_var data available. Skipping strip plot.")
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5-class color palette (same as scatter plots)
    task_palette = {
        'T1':      '#4CAF50',  # green - baseline
        'T2-ctrl': '#90CAF9',  # light blue - easy task
        'T2-test': '#1565C0',  # dark blue - hard task
        'T3-ctrl': '#EF9A9A',  # light red - easy task
        'T3-test': '#C62828',  # dark red - hard task
    }

    # Legend order for logical grouping
    hue_order = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']

    x_upper_limit = merged_df['phasic_var'].quantile(OUTLIER_PERCENTILE / 100)

    # Split data into normal points and outliers
    plot_df = merged_df[merged_df['phasic_var'] <= x_upper_limit]
    outliers = merged_df[merged_df['phasic_var'] > x_upper_limit]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.stripplot(
        data=plot_df,
        x='phasic_var',
        y='task_label',
        hue='task_label',
        palette=task_palette,
        order=hue_order,
        hue_order=hue_order,
        size=8,
        alpha=0.7,
        jitter=0.25,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel('EDA Phasic Variance (µS²)', fontsize=12)
    ax.set_ylabel('Task-Group', fontsize=12)
    ax.set_title('EDA Phasic Variance Distribution by Task', fontsize=14, fontweight='bold')

    # Add outlier annotation below the plot if any outliers exist
    if not outliers.empty:
        outlier_lines = [f"Outliers (>{OUTLIER_PERCENTILE}%ile, X>{x_upper_limit:.4f}):"]
        for _, row in outliers.iterrows():
            outlier_lines.append(
                f"  {row['subject']} ({row['task_label']}): {row['phasic_var']:.4f}"
            )
        outlier_text = "\n".join(outlier_lines)

        fig.text(
            0.12, -0.02,
            outlier_text,
            fontsize=9,
            fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='#FFB300', alpha=0.9),
        )

    plt.tight_layout()

    output_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_STRIP_PHASIC
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def plot_box_phasic_std_by_task(merged_df: pd.DataFrame) -> None:
    """Plot box plot showing EDA phasic standard deviation distribution by task_label.

    Creates a box plot with task_label on X-axis and phasic_std on Y-axis.
    Includes individual data points as a strip plot overlay.
    Excludes outliers above OUTLIER_PERCENTILE.

    Args:
        merged_df: DataFrame from merge_eda_and_ppg_features(), must contain
            'phasic_std' column.

    Returns:
        None.

    Outputs:
        Saves to DATA_MINING_OUTPUT_DIR:
            - box_task_vs_phasic_std.jpg
    """
    if merged_df.empty or 'phasic_std' not in merged_df.columns:
        print("[WARN] No phasic_std data available. Skipping box plot.")
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter outliers
    y_upper_limit = merged_df['phasic_std'].quantile(OUTLIER_PERCENTILE / 100)
    filtered_df = merged_df[merged_df['phasic_std'] <= y_upper_limit].copy()
    outlier_count = len(merged_df) - len(filtered_df)
    if outlier_count > 0:
        print(f"[INFO] Excluded {outlier_count} outliers (>{OUTLIER_PERCENTILE}th percentile, >{y_upper_limit:.2f}) for Phasic Std box plot.")

    # 5-class color palette
    task_palette = {
        'T1':      '#4CAF50',  # green - baseline
        'T2-ctrl': '#90CAF9',  # light blue - easy task
        'T2-test': '#1565C0',  # dark blue - hard task
        'T3-ctrl': '#EF9A9A',  # light red - easy task
        'T3-test': '#C62828',  # dark red - hard task
    }

    hue_order = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Box plot for distribution statistics
    sns.boxplot(
        data=filtered_df,
        x='task_label',
        hue='task_label',
        y='phasic_std',
        palette=task_palette,
        order=hue_order,
        hue_order=hue_order,
        showfliers=False,  # Hide outliers to avoid duplication with strip plot
        ax=ax,
        legend=False,
    )
    
    # Strip plot overlay for individual points
    sns.stripplot(
        data=filtered_df,
        x='task_label',
        y='phasic_std',
        color='black',
        order=hue_order,
        size=4,
        alpha=0.3,
        jitter=True,
        ax=ax
    )

    ax.set_xlabel('Task-Group', fontsize=12)
    ax.set_ylabel('EDA Phasic Standard Deviation (µS)', fontsize=12)
    ax.set_title(f'EDA Phasic Std Distribution by Task (Excl. top {100-OUTLIER_PERCENTILE}%)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_BOX_PHASIC_STD
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def plot_box_rmssd_by_task(merged_df: pd.DataFrame) -> None:
    """Plot box plot showing HRV RMSSD distribution by task_label.

    Creates a box plot with task_label on X-axis and hrv_rmssd on Y-axis.
    Includes individual data points as a strip plot overlay.
    Excludes outliers above OUTLIER_PERCENTILE.

    Args:
        merged_df: DataFrame from merge_eda_and_ppg_features(), must contain
            'hrv_rmssd' column.

    Returns:
        None.

    Outputs:
        Saves to DATA_MINING_OUTPUT_DIR:
            - box_task_vs_hrv_rmssd.jpg
    """
    if merged_df.empty or 'hrv_rmssd' not in merged_df.columns:
        print("[WARN] No hrv_rmssd data available. Skipping box plot.")
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter outliers
    y_upper_limit = merged_df['hrv_rmssd'].quantile(OUTLIER_PERCENTILE / 100)
    filtered_df = merged_df[merged_df['hrv_rmssd'] <= y_upper_limit].copy()
    outlier_count = len(merged_df) - len(filtered_df)
    if outlier_count > 0:
        print(f"[INFO] Excluded {outlier_count} outliers (>{OUTLIER_PERCENTILE}th percentile, >{y_upper_limit:.2f}) for RMSSD box plot.")

    # 5-class color palette
    task_palette = {
        'T1':      '#4CAF50',  # green - baseline
        'T2-ctrl': '#90CAF9',  # light blue - easy task
        'T2-test': '#1565C0',  # dark blue - hard task
        'T3-ctrl': '#EF9A9A',  # light red - easy task
        'T3-test': '#C62828',  # dark red - hard task
    }

    hue_order = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Box plot for distribution statistics
    sns.boxplot(
        data=filtered_df,
        x='task_label',
        hue='task_label',
        y='hrv_rmssd',
        palette=task_palette,
        order=hue_order,
        hue_order=hue_order,
        showfliers=False,
        ax=ax,
        legend=False,
    )
    
    # Strip plot overlay for individual points
    sns.stripplot(
        data=filtered_df,
        x='task_label',
        y='hrv_rmssd',
        color='black',
        order=hue_order,
        size=4,
        alpha=0.3,
        jitter=True,
        ax=ax
    )

    ax.set_xlabel('Task-Group', fontsize=12)
    ax.set_ylabel('HRV RMSSD (ms)', fontsize=12)
    ax.set_title(f'HRV RMSSD Distribution by Task (Excl. top {100-OUTLIER_PERCENTILE}%)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_BOX_RMSSD
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def plot_box_rppg_rmssd_by_task(merged_df: pd.DataFrame) -> None:
    """Plot box plot showing rPPG HRV RMSSD distribution by task_label.

    Creates a box plot with task_label on X-axis and rppg_hrv_rmssd on Y-axis.
    Includes individual data points as a strip plot overlay.
    Excludes outliers above OUTLIER_PERCENTILE.

    Args:
        merged_df: DataFrame from merge_eda_and_ppg_features(), must contain
            'rppg_hrv_rmssd' column.

    Returns:
        None.

    Outputs:
        Saves to DATA_MINING_OUTPUT_DIR:
            - box_task_vs_rppg_hrv_rmssd.jpg
    """
    if merged_df.empty or 'rppg_hrv_rmssd' not in merged_df.columns:
        print("[WARN] No rppg_hrv_rmssd data available. Skipping box plot.")
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter outliers
    y_upper_limit = merged_df['rppg_hrv_rmssd'].quantile(OUTLIER_PERCENTILE / 100)
    filtered_df = merged_df[merged_df['rppg_hrv_rmssd'] <= y_upper_limit].copy()
    outlier_count = len(merged_df) - len(filtered_df)
    if outlier_count > 0:
        print(
            f"[INFO] Excluded {outlier_count} outliers (>{OUTLIER_PERCENTILE}th percentile, "
            f">{y_upper_limit:.2f}) for rPPG RMSSD box plot."
        )

    # 5-class color palette
    task_palette = {
        'T1':      '#4CAF50',  # green - baseline
        'T2-ctrl': '#90CAF9',  # light blue - easy task
        'T2-test': '#1565C0',  # dark blue - hard task
        'T3-ctrl': '#EF9A9A',  # light red - easy task
        'T3-test': '#C62828',  # dark red - hard task
    }

    hue_order = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']

    fig, ax = plt.subplots(figsize=(10, 7))

    # Box plot for distribution statistics
    sns.boxplot(
        data=filtered_df,
        x='task_label',
        hue='task_label',
        y='rppg_hrv_rmssd',
        palette=task_palette,
        order=hue_order,
        hue_order=hue_order,
        showfliers=False,
        ax=ax,
        legend=False,
    )

    # Strip plot overlay for individual points
    sns.stripplot(
        data=filtered_df,
        x='task_label',
        y='rppg_hrv_rmssd',
        color='black',
        order=hue_order,
        size=4,
        alpha=0.3,
        jitter=True,
        ax=ax
    )

    ax.set_xlabel('Task-Group', fontsize=12)
    ax.set_ylabel('rPPG HRV RMSSD (ms)', fontsize=12)
    ax.set_title(
        f'rPPG HRV RMSSD Distribution by Task (Excl. top {100-OUTLIER_PERCENTILE}%)',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    output_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_BOX_RPPG_RMSSD
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def plot_box_hr_median_by_task(merged_df: pd.DataFrame) -> None:
    """Plot box plot of per-subject-task median heartbeat rate by task_label.

    X-axis follows fixed order:
        T1, T2-ctrl, T2-test, T3-ctrl, T3-test.
    Y-axis is hr_median (median of beat-wise HR in bpm) for each subject-task pair.
    """
    if merged_df.empty or 'hr_median' not in merged_df.columns:
        print("[WARN] No hr_median data available. Skipping HR median box plot.")
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task_palette = {
        'T1':      '#4CAF50',  # green - baseline
        'T2-ctrl': '#90CAF9',  # light blue - easy task
        'T2-test': '#1565C0',  # dark blue - hard task
        'T3-ctrl': '#EF9A9A',  # light red - easy task
        'T3-test': '#C62828',  # dark red - hard task
    }
    hue_order = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.boxplot(
        data=merged_df,
        x='task_label',
        hue='task_label',
        y='hr_median',
        palette=task_palette,
        order=hue_order,
        hue_order=hue_order,
        ax=ax,
        legend=False,
    )

    sns.stripplot(
        data=merged_df,
        x='task_label',
        y='hr_median',
        color='black',
        order=hue_order,
        size=4,
        alpha=0.3,
        jitter=True,
        ax=ax,
    )

    ax.set_xlabel('Task-Group', fontsize=12)
    ax.set_ylabel('Median Heart Rate (bpm)', fontsize=12)
    ax.set_title('Median Heart Rate Distribution by Task', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_BOX_HR_MEDIAN
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def plot_strip_hrv_by_task(merged_df: pd.DataFrame) -> None:
    """Plot strip plots showing HRV metrics distribution by task_label.

    Creates two vertical strip plots with task_label on X-axis and HRV metric
    values on Y-axis. Outliers above 98th percentile are excluded and annotated
    below each plot.

    Args:
        merged_df: DataFrame from merge_eda_and_ppg_features(), must contain
            'hrv_sdnn' and 'hrv_rmssd' columns.

    Returns:
        None.

    Outputs:
        Saves to DATA_MINING_OUTPUT_DIR:
            - strip_task_vs_hrv_sdnn.jpg
            - strip_task_vs_hrv_rmssd.jpg
    """
    if merged_df.empty:
        print("[WARN] No data available. Skipping HRV strip plots.")
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5-class color palette (same as other plots)
    task_palette = {
        'T1':      '#4CAF50',  # green - baseline
        'T2-ctrl': '#90CAF9',  # light blue - easy task
        'T2-test': '#1565C0',  # dark blue - hard task
        'T3-ctrl': '#EF9A9A',  # light red - easy task
        'T3-test': '#C62828',  # dark red - hard task
    }

    hue_order = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']

    # Plot configs: (y_column, y_label, output_filename)
    hrv_configs = [
        ('hrv_sdnn', 'HRV SDNN (ms)', OUT_STRIP_SDNN),
        ('hrv_rmssd', 'HRV RMSSD (ms)', OUT_STRIP_RMSSD),
    ]

    for y_col, y_label, filename in hrv_configs:
        if y_col not in merged_df.columns:
            print(f"[WARN] Column {y_col} not found. Skipping.")
            continue

        # Calculate Y-axis upper limit at 98 percentile
        y_upper_limit = merged_df[y_col].quantile(OUTLIER_PERCENTILE / 100)

        # Split data into normal points and outliers
        plot_df = merged_df[merged_df[y_col] <= y_upper_limit]
        outliers = merged_df[merged_df[y_col] > y_upper_limit]

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.stripplot(
            data=plot_df,
            x='task_label',
            y=y_col,
            hue='task_label',
            palette=task_palette,
            order=hue_order,
            hue_order=hue_order,
            size=8,
            alpha=0.7,
            jitter=0.25,
            ax=ax,
            legend=False,
        )

        ax.set_xlabel('Task-Group', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{y_label} Distribution by Task', fontsize=14, fontweight='bold')

        # Add outlier annotation below the plot if any outliers exist
        if not outliers.empty:
            outlier_lines = [f"Outliers (>{OUTLIER_PERCENTILE}%ile, Y>{y_upper_limit:.0f}):"]
            for _, row in outliers.iterrows():
                outlier_lines.append(
                    f"  {row['subject']} ({row['task_label']}): {row[y_col]:.1f}"
                )
            outlier_text = "\n".join(outlier_lines)

            fig.text(
                0.12, -0.02,
                outlier_text,
                fontsize=9,
                fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='#FFB300', alpha=0.9),
            )

        plt.tight_layout()

        output_path = dmc.DATA_MINING_OUTPUT_DIR / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved {output_path}")


def plot_box_interaction_feature(merged_df: pd.DataFrame) -> None:
    """Plot box plot of Tonic Slope * Phasic Std interaction feature.

    Calculates a new feature (tonic_slope * phasic_std) representing the
    interaction between stress trend and reactivity volatility.

    Args:
        merged_df: DataFrame from merge_eda_and_ppg_features().

    Returns:
        None.

    Outputs:
        Saves to DATA_MINING_OUTPUT_DIR:
            - box_interaction_slope_phasic.jpg
    """
    required_cols = ['tonic_slope', 'phasic_std', 'task_label']
    if merged_df.empty or not all(col in merged_df.columns for col in required_cols):
        print("[WARN] Missing columns for interaction plot. Skipping.")
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate the interaction feature
    # Feature = Trend (Slope) * Volatility (Std)
    merged_df = merged_df.copy()
    merged_df['interaction_feature'] = merged_df['tonic_slope'] * merged_df['phasic_std']

    # 5-class color palette
    task_palette = {
        'T1':      '#4CAF50',  # green
        'T2-ctrl': '#90CAF9',  # light blue
        'T2-test': '#1565C0',  # dark blue
        'T3-ctrl': '#EF9A9A',  # light red
        'T3-test': '#C62828',  # dark red
    }

    hue_order = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']

    fig, ax = plt.subplots(figsize=(10, 7))

    # Box plot
    sns.boxplot(
        data=merged_df,
        x='task_label',
        hue='task_label',
        y='interaction_feature',
        palette=task_palette,
        order=hue_order,
        hue_order=hue_order,
        showfliers=False,
        ax=ax,
        legend=False,
    )

    # Strip plot overlay
    sns.stripplot(
        data=merged_df,
        x='task_label',
        y='interaction_feature',
        color='black',
        order=hue_order,
        size=4,
        alpha=0.3,
        jitter=True,
        ax=ax
    )

    # Add a horizontal line at 0 to distinguish increasing vs decreasing trend interactions
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Fixed Y-axis range for better visibility of T1/T3
    ax.set_ylim(-0.00001, 0.00001)

    ax.set_xlabel('Task-Group', fontsize=12)
    ax.set_ylabel('Tonic Slope x Phasic Std', fontsize=12)
    ax.set_title('Tonic Slope x Phasic Std', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_BOX_INTERACTION
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def plot_3d_eda_feature_space(merged_df: pd.DataFrame) -> None:
    """Plot interactive 3D scatter plot of EDA features using Plotly.

    Creates an interactive 3D scatter plot showing EDA feature space with
    tonic_mean on X-axis, phasic_std on Y-axis, and tonic_slope on Z-axis.
    Points are colored and shaped by task_label. Can be rotated/zoomed in browser.

    Args:
        merged_df: DataFrame from merge_eda_and_ppg_features(), must contain
            'tonic_mean', 'phasic_std', 'tonic_slope', and 'task_label' columns.

    Returns:
        None.

    Outputs:
        Saves to DATA_MINING_OUTPUT_DIR:
            - scatter3d_tonic_phasic_slope.html
    """
    required_cols = ['tonic_mean', 'phasic_std', 'tonic_slope', 'task_label']
    if merged_df.empty or not all(col in merged_df.columns for col in required_cols):
        print("[WARN] Missing required columns for 3D plot. Skipping.")
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5-class color palette (same as other plots)
    task_palette = {
        'T1':      '#4CAF50',  # green - baseline
        'T2-ctrl': '#90CAF9',  # light blue - easy task
        'T2-test': '#1565C0',  # dark blue - hard task
        'T3-ctrl': '#EF9A9A',  # light red - easy task
        'T3-test': '#C62828',  # dark red - hard task
    }

    # Symbol mapping: ctrl=circle, test=diamond, T1=circle
    symbol_map = {
        'T1':      'circle',
        'T2-ctrl': 'circle',
        'T2-test': 'diamond',
        'T3-ctrl': 'circle',
        'T3-test': 'diamond',
    }

    hue_order = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']

    fig = px.scatter_3d(
        merged_df,
        x='tonic_mean',
        y='phasic_std',
        z='tonic_slope',
        color='task_label',
        symbol='task_label',
        color_discrete_map=task_palette,
        symbol_map=symbol_map,
        category_orders={'task_label': hue_order},
        hover_data=['subject', 'task', 'group'],
        labels={
            'tonic_mean': 'Tonic Mean (µS)',
            'phasic_std': 'Phasic Std (µS)',
            'tonic_slope': 'Tonic Slope (µS/sample)',
            'task_label': 'Task-Group',
        },
        title='3D EDA Features by Task-Group (Interactive)',
    )

    fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color='white')))
    fig.update_layout(
        legend_title_text='Task-Group',
        scene=dict(
            xaxis_title='Tonic Mean (µS)',
            yaxis_title='Phasic Std (µS)',
            zaxis_title='Tonic Slope (µS/sample)',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    output_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_3D_EDA
    fig.write_html(str(output_path))
    print(f"[INFO] Saved {output_path}")


def calculate_rolling_rmssd(
    bvp_signal: list[float],
    fs: int,
    window_sec: int = ROLLING_WINDOW_SEC,
    step_sec: float = ROLLING_BVP_STEP_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate rolling RMSSD from BVP signal.

    1. Cleans BVP signal and finds peaks.
    2. Fixes peaks (Kubios) and cleans RR intervals (250-1300ms, interpolate).
    3. Computes RMSSD in sliding windows.

    Args:
        bvp_signal: Raw BVP signal data.
        fs: Sampling rate (Hz).
        window_sec: Size of sliding window in seconds.
        step_sec: Step size for sliding window in seconds.

    Returns:
        Tuple of (time_points, rmssd_values).
    """
    try:
        # Clean signal and find peaks
        cleaned = nk.ppg_clean(bvp_signal, sampling_rate=fs)
        peaks_dict = nk.ppg_findpeaks(cleaned, sampling_rate=fs)
        peaks = peaks_dict['PPG_Peaks']
    except Exception as e:
        # print(f"[DEBUG] Error processing BVP for rolling RMSSD: {e}")
        return np.array([]), np.array([])

    if len(peaks) < 5:
        return np.array([]), np.array([])

    rri_dict, _qc = clean_rr_intervals_from_peaks(
        peaks,
        fs,
        interval_min_ms=250.0,
        interval_max_ms=1300.0,
        min_valid_ratio=0.8,
        min_valid_intervals=10,
        fixpeaks_method="Kubios",
        iterative=True,
    )
    if rri_dict is None:
        return np.array([]), np.array([])

    # Cleaned NN intervals in ms + timestamps (s at end of interval)
    rr_intervals = np.asarray(rri_dict["RRI"], dtype=float)
    rr_times_ms = np.asarray(rri_dict["RRI_Time"], dtype=float) * 1000.0

    # Rolling window — only generate complete windows
    # n = floor((duration - window) / step) + 1
    signal_duration_sec = len(bvp_signal) / fs
    max_start = signal_duration_sec - window_sec
    if max_start < 0:
        return np.array([]), np.array([])
    n_windows = int(np.floor(max_start / step_sec)) + 1
    time_points = np.arange(n_windows) * step_sec
    rmssd_values = []

    window_ms = window_sec * 1000

    for t_sec in time_points:
        t_ms = t_sec * 1000
        # Find RR intervals that fall within [t_ms, t_ms + window_ms]
        mask = (rr_times_ms >= t_ms) & (rr_times_ms < t_ms + window_ms)
        window_rrs = rr_intervals[mask]

        if len(window_rrs) < 2:
            rmssd_values.append(np.nan)
        else:
            # Calculate RMSSD: sqrt(mean(diff(RR)^2))
            diff_rrs = np.diff(window_rrs)
            rmssd = np.sqrt(np.mean(diff_rrs**2))
            rmssd_values.append(rmssd)

    return time_points, np.array(rmssd_values)


def calculate_rolling_hr(
    bvp_signal: list[float],
    fs: int,
    window_sec: int = ROLLING_WINDOW_SEC,
    step_sec: float = ROLLING_BVP_STEP_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate rolling heart rate (bpm) from BVP signal.

    1. Cleans BVP signal and finds peaks.
    2. Fixes peaks (Kubios) and cleans RR intervals (250-1300ms, interpolate).
    3. Computes HR in sliding windows.

    Args:
        bvp_signal: Raw BVP signal data.
        fs: Sampling rate (Hz).
        window_sec: Size of sliding window in seconds.
        step_sec: Step size for sliding window in seconds.

    Returns:
        Tuple of (time_points, hr_values).
    """
    try:
        cleaned = nk.ppg_clean(bvp_signal, sampling_rate=fs)
        peaks_dict = nk.ppg_findpeaks(cleaned, sampling_rate=fs)
        peaks = peaks_dict['PPG_Peaks']
    except Exception:
        return np.array([]), np.array([])

    if len(peaks) < 5:
        return np.array([]), np.array([])

    rri_dict, _qc = clean_rr_intervals_from_peaks(
        peaks,
        fs,
        interval_min_ms=250.0,
        interval_max_ms=1300.0,
        min_valid_ratio=0.8,
        min_valid_intervals=10,
        fixpeaks_method="Kubios",
        iterative=True,
    )
    if rri_dict is None:
        return np.array([]), np.array([])

    rr_intervals = np.asarray(rri_dict["RRI"], dtype=float)
    rr_times_ms = np.asarray(rri_dict["RRI_Time"], dtype=float) * 1000.0

    # Only generate complete windows: n = floor((duration - window) / step) + 1
    signal_duration_sec = len(bvp_signal) / fs
    max_start = signal_duration_sec - window_sec
    if max_start < 0:
        return np.array([]), np.array([])
    n_windows = int(np.floor(max_start / step_sec)) + 1
    time_points = np.arange(n_windows) * step_sec
    hr_values = []

    window_ms = window_sec * 1000

    for t_sec in time_points:
        t_ms = t_sec * 1000
        mask = (rr_times_ms >= t_ms) & (rr_times_ms < t_ms + window_ms)
        window_rrs = rr_intervals[mask]

        if len(window_rrs) < 2:
            hr_values.append(np.nan)
        else:
            hr_bpm = np.nanmean(60000.0 / window_rrs)
            hr_values.append(hr_bpm)

    return time_points, np.array(hr_values)


def get_subject_hrv_profile(subject_path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Calculate concatenated rolling RMSSD for a subject across T1-T3.

    Returns:
        Tuple of (full_time, full_rmssd, task_boundaries).
    """
    bvp_paths = dmc.list_bvp_paths(subject_path)
    if not dmc.has_expected_bvp_files(bvp_paths, subject_path):
        return np.array([]), np.array([]), []

    all_times = []
    all_rmssd = []
    boundaries = []
    current_offset = 0

    # Process T1, T2, T3 in order
    for bvp_path in bvp_paths:
        bvp_data = dmc.read_bvp_data(bvp_path)
        duration = len(bvp_data) / dmc.BVP_SAMPLING_RATE_HZ

        # Calculate rolling RMSSD (10s window, 1Hz stride)
        times, rmssd = calculate_rolling_rmssd(
            bvp_data,
            dmc.BVP_SAMPLING_RATE_HZ,
            window_sec=ROLLING_WINDOW_SEC,
            step_sec=ROLLING_BVP_STEP_SEC,
        )

        if len(times) == 0:
            # Fallback for empty/failed processing (strict window count)
            max_start_fb = duration - ROLLING_WINDOW_SEC
            n_fb = max(int(np.floor(max_start_fb / ROLLING_BVP_STEP_SEC)) + 1, 0) if max_start_fb >= 0 else 0
            times = np.arange(n_fb) * ROLLING_BVP_STEP_SEC
            rmssd = np.full(len(times), np.nan)

        # Shift times by current cumulative offset
        all_times.append(times + current_offset)
        all_rmssd.append(rmssd)

        current_offset += duration
        boundaries.append(current_offset)

    full_time = np.concatenate(all_times)
    full_rmssd = np.concatenate(all_rmssd)

    return full_time, full_rmssd, boundaries


def get_subject_hr_profile(subject_path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Calculate concatenated rolling HR for a subject across T1-T3.

    Returns:
        Tuple of (full_time, full_hr, task_boundaries).
    """
    bvp_paths = dmc.list_bvp_paths(subject_path)
    if not dmc.has_expected_bvp_files(bvp_paths, subject_path):
        return np.array([]), np.array([]), []

    all_times = []
    all_hr = []
    boundaries = []
    current_offset = 0

    # Process T1, T2, T3 in order
    for bvp_path in bvp_paths:
        bvp_data = dmc.read_bvp_data(bvp_path)
        duration = len(bvp_data) / dmc.BVP_SAMPLING_RATE_HZ

        # Calculate rolling HR (10s window, 1Hz stride)
        times, hr = calculate_rolling_hr(
            bvp_data,
            dmc.BVP_SAMPLING_RATE_HZ,
            window_sec=ROLLING_WINDOW_SEC,
            step_sec=ROLLING_BVP_STEP_SEC,
        )

        if len(times) == 0:
            # Fallback for empty/failed processing (strict window count)
            max_start_fb = duration - ROLLING_WINDOW_SEC
            n_fb = max(int(np.floor(max_start_fb / ROLLING_BVP_STEP_SEC)) + 1, 0) if max_start_fb >= 0 else 0
            times = np.arange(n_fb) * ROLLING_BVP_STEP_SEC
            hr = np.full(len(times), np.nan)

        # Shift times by current cumulative offset
        all_times.append(times + current_offset)
        all_hr.append(hr)

        current_offset += duration
        boundaries.append(current_offset)

    full_time = np.concatenate(all_times)
    full_hr = np.concatenate(all_hr)

    return full_time, full_hr, boundaries


def plot_subject_rolling_hrv(dataset_path: Path, subject_group_map: dict[str, str]) -> None:
    """Generate per-subject plots of rolling RMSSD across T1-T3.

    Calculates RMSSD in a sliding window (10s) for each task and concatenates
    them on a single timeline to show HRV evolution.

    Args:
        dataset_path: Path to UBFC-Phys/Data folder.
        subject_group_map: Mapping of {subject_id: group}.

    Outputs:
        Saves to data_mining/subject_hrv_profiles/hrv_profile_{subject_id}.jpg
    """
    output_dir = dmc.DATA_MINING_OUTPUT_DIR / "subject_hrv_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_path in dmc.list_subject_paths(dataset_path):
        subject_id = subject_path.name
        group = subject_group_map.get(subject_id, 'unknown')
        
        full_time, full_rmssd, boundaries = get_subject_hrv_profile(subject_path)
        if len(full_time) == 0:
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(
            full_time,
            full_rmssd,
            color='#E91E63',
            linewidth=2,
            label=(
                f'RMSSD ({ROLLING_WINDOW_SEC}s window, '
                f'stride={ROLLING_STRIDE_SAMPLES} sample '
                f'({ROLLING_BVP_STEP_SEC:.5f}s))'
            ),
        )

        # Add vertical lines for task boundaries
        for b in boundaries[:-1]:
            ax.axvline(x=b, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

        # Add task labels centered in each segment
        segment_starts = [0] + boundaries[:-1]
        segment_ends = boundaries
        labels = ['T1', 'T2', 'T3']
        
        y_lims = ax.get_ylim()
        for start, end, lbl in zip(segment_starts, segment_ends, labels):
            mid_point = (start + end) / 2
            ax.text(mid_point, y_lims[1], lbl, ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('HRV RMSSD (ms)', fontsize=12)
        ax.set_title(f'Subject {subject_id} ({group}) - HRV RMSSD Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        out_path = output_dir / f"hrv_profile_{subject_id}.jpg"
        fig.savefig(out_path, dpi=ROLLING_SUBJECT_PLOT_DPI, bbox_inches='tight')
        plt.close(fig)

    print(f"[INFO] Saved per-subject HRV profiles to {output_dir}")


def plot_subject_rolling_hr(dataset_path: Path, subject_group_map: dict[str, str]) -> None:
    """Generate per-subject plots of rolling HR across T1-T3.

    Calculates HR in a sliding window (10s) for each task and concatenates
    them on a single timeline to show HR evolution.

    Args:
        dataset_path: Path to UBFC-Phys/Data folder.
        subject_group_map: Mapping of {subject_id: group}.

    Outputs:
        Saves to data_mining/subject_hr_profiles/hr_profile_{subject_id}.jpg
    """
    output_dir = dmc.DATA_MINING_OUTPUT_DIR / "subject_hr_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_path in dmc.list_subject_paths(dataset_path):
        subject_id = subject_path.name
        group = subject_group_map.get(subject_id, 'unknown')

        full_time, full_hr, boundaries = get_subject_hr_profile(subject_path)
        if len(full_time) == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            full_time,
            full_hr,
            color='#1976D2',
            linewidth=2,
            label=(
                f'HR ({ROLLING_WINDOW_SEC}s window, '
                f'stride={ROLLING_STRIDE_SAMPLES} sample '
                f'({ROLLING_BVP_STEP_SEC:.5f}s))'
            ),
        )

        for b in boundaries[:-1]:
            ax.axvline(x=b, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

        segment_starts = [0] + boundaries[:-1]
        segment_ends = boundaries
        labels = ['T1', 'T2', 'T3']

        y_lims = ax.get_ylim()
        for start, end, lbl in zip(segment_starts, segment_ends, labels):
            mid_point = (start + end) / 2
            ax.text(mid_point, y_lims[1], lbl, ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Heart Rate (bpm)', fontsize=12)
        ax.set_title(f'Subject {subject_id} ({group}) - HR Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        out_path = output_dir / f"hr_profile_{subject_id}.jpg"
        fig.savefig(out_path, dpi=ROLLING_SUBJECT_PLOT_DPI, bbox_inches='tight')
        plt.close(fig)

        # Save plotted HR curve data alongside JPG (one row per data point)
        csv_path = out_path.with_suffix(".csv")
        hr_curve_df = pd.DataFrame(
            {
                "subject": subject_id,
                "group": group,
                "data_point_idx": np.arange(len(full_time)),
                "window_start_sec": full_time,
                "window_end_sec": full_time + ROLLING_WINDOW_SEC,
                "hr_bpm": full_hr,
            }
        )
        hr_curve_df.to_csv(csv_path, index=False)

    print(f"[INFO] Saved per-subject HR profiles to {output_dir}")


def plot_group_rolling_hrv(subject_paths: list[Path], group_name: str, subject_group_map: dict[str, str]) -> None:
    """Plot combined rolling RMSSD profiles for a group of subjects.
    
    Args:
        subject_paths: List of subject paths to include in the plot.
        group_name: Name of the group (used for filename and title).
        subject_group_map: Mapping of {subject_id: group}.
    """
    output_dir = dmc.DATA_MINING_OUTPUT_DIR / "group_hrv_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use a colormap
    num_subjects = len(subject_paths)
    # Use tab20 if <= 20, else jet
    if num_subjects <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_subjects))
    else:
        colors = plt.cm.jet(np.linspace(0, 1, num_subjects))

    has_data = False
    for i, subject_path in enumerate(subject_paths):
        subject_id = subject_path.name
        full_time, full_rmssd, _ = get_subject_hrv_profile(subject_path)
        
        if len(full_time) > 0:
            has_data = True
            ax.plot(full_time, full_rmssd, label=subject_id, color=colors[i], alpha=0.7, linewidth=1.5)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('HRV RMSSD (ms)', fontsize=12)
    ax.set_title(f'Group HRV RMSSD Profile - {group_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend only if not too many subjects
    if num_subjects <= 20:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
    
    plt.tight_layout()
    out_path = output_dir / f"hrv_profile_{group_name}.jpg"
    fig.savefig(out_path, dpi=ROLLING_GROUP_PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_group_rolling_hr(subject_paths: list[Path], group_name: str, subject_group_map: dict[str, str]) -> None:
    """Plot combined rolling HR profiles for a group of subjects.

    Args:
        subject_paths: List of subject paths to include in the plot.
        group_name: Name of the group (used for filename and title).
        subject_group_map: Mapping of {subject_id: group}.
    """
    output_dir = dmc.DATA_MINING_OUTPUT_DIR / "group_hr_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    num_subjects = len(subject_paths)
    if num_subjects <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_subjects))
    else:
        colors = plt.cm.jet(np.linspace(0, 1, num_subjects))

    has_data = False
    hr_curve_rows: list[pd.DataFrame] = []
    for i, subject_path in enumerate(subject_paths):
        subject_id = subject_path.name
        subject_group = subject_group_map.get(subject_id, "unknown")
        full_time, full_hr, _ = get_subject_hr_profile(subject_path)

        if len(full_time) > 0:
            has_data = True
            ax.plot(full_time, full_hr, label=subject_id, color=colors[i], alpha=0.7, linewidth=1.5)
            hr_curve_rows.append(
                pd.DataFrame(
                    {
                        "group_name": group_name,
                        "subject": subject_id,
                        "group": subject_group,
                        "data_point_idx": np.arange(len(full_time)),
                        "window_start_sec": full_time,
                        "window_end_sec": full_time + ROLLING_WINDOW_SEC,
                        "hr_bpm": full_hr,
                    }
                )
            )

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Heart Rate (bpm)', fontsize=12)
    ax.set_title(f'Group HR Profile - {group_name}', fontsize=14, fontweight='bold')

    if num_subjects <= 20:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')

    plt.tight_layout()
    out_path = output_dir / f"hr_profile_{group_name}.jpg"
    fig.savefig(out_path, dpi=ROLLING_GROUP_PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")

    # Save plotted group HR curve data alongside JPG
    if hr_curve_rows:
        csv_path = out_path.with_suffix(".csv")
        group_hr_curve_df = pd.concat(hr_curve_rows, ignore_index=True)
        group_hr_curve_df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved {csv_path}")


def get_subject_bvp_profile(subject_path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Calculate concatenated cleaned BVP profile for a subject across T1-T3.

    Returns:
        Tuple of (full_time, full_bvp, task_boundaries).
    """
    bvp_paths = dmc.list_bvp_paths(subject_path)
    if not dmc.has_expected_bvp_files(bvp_paths, subject_path):
        return np.array([]), np.array([]), []

    all_times = []
    all_bvp = []
    boundaries = []
    current_offset = 0.0

    # Process T1, T2, T3 in order
    for bvp_path in bvp_paths:
        bvp_data = np.asarray(dmc.read_bvp_data(bvp_path), dtype=float)
        duration = len(bvp_data) / dmc.BVP_SAMPLING_RATE_HZ

        # No windowing: use full cleaned BVP timeline directly
        try:
            bvp_clean = np.asarray(
                nk.ppg_clean(bvp_data, sampling_rate=dmc.BVP_SAMPLING_RATE_HZ),
                dtype=float
            )
        except Exception as e:
            print(f"[WARN] {bvp_path}: ppg_clean failed ({e}). Using raw BVP.")
            bvp_clean = bvp_data

        times = np.arange(len(bvp_clean), dtype=float) / dmc.BVP_SAMPLING_RATE_HZ

        # Shift times by current cumulative offset
        all_times.append(times + current_offset)
        all_bvp.append(bvp_clean)

        current_offset += duration
        boundaries.append(current_offset)

    full_time = np.concatenate(all_times)
    full_bvp = np.concatenate(all_bvp)
    return full_time, full_bvp, boundaries


def plot_subject_bvp_profile(dataset_path: Path, subject_group_map: dict[str, str]) -> None:
    """Generate per-subject plots of cleaned BVP across T1-T3.

    Args:
        dataset_path: Path to UBFC-Phys/Data folder.
        subject_group_map: Mapping of {subject_id: group}.

    Outputs:
        Saves to data_mining/subject_bvp_profiles/bvp_profile_{subject_id}.jpg
    """
    output_dir = dmc.DATA_MINING_OUTPUT_DIR / "subject_bvp_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_path in dmc.list_subject_paths(dataset_path):
        subject_id = subject_path.name
        group = subject_group_map.get(subject_id, 'unknown')

        full_time, full_bvp, boundaries = get_subject_bvp_profile(subject_path)
        if len(full_time) == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(full_time, full_bvp, color='#7B1FA2', linewidth=1.2, label='Cleaned BVP')

        for b in boundaries[:-1]:
            ax.axvline(x=b, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

        segment_starts = [0] + boundaries[:-1]
        segment_ends = boundaries
        labels = ['T1', 'T2', 'T3']

        y_lims = ax.get_ylim()
        for start, end, lbl in zip(segment_starts, segment_ends, labels):
            mid_point = (start + end) / 2
            ax.text(mid_point, y_lims[1], lbl, ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('BVP Amplitude', fontsize=12)
        ax.set_title(f'Subject {subject_id} ({group}) - Cleaned BVP Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        out_path = output_dir / f"bvp_profile_{subject_id}.jpg"
        fig.savefig(out_path, dpi=ROLLING_SUBJECT_PLOT_DPI, bbox_inches='tight')
        plt.close(fig)

    print(f"[INFO] Saved per-subject BVP profiles to {output_dir}")


def plot_group_bvp_profile(subject_paths: list[Path], group_name: str, subject_group_map: dict[str, str]) -> None:
    """Plot combined cleaned BVP profiles for a group of subjects.

    Args:
        subject_paths: List of subject paths to include in the plot.
        group_name: Name of the group (used for filename and title).
        subject_group_map: Mapping of {subject_id: group}.
    """
    output_dir = dmc.DATA_MINING_OUTPUT_DIR / "group_bvp_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    num_subjects = len(subject_paths)
    if num_subjects <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_subjects))
    else:
        colors = plt.cm.jet(np.linspace(0, 1, num_subjects))

    has_data = False
    for i, subject_path in enumerate(subject_paths):
        subject_id = subject_path.name
        full_time, full_bvp, _ = get_subject_bvp_profile(subject_path)

        if len(full_time) > 0:
            has_data = True
            ax.plot(full_time, full_bvp, label=subject_id, color=colors[i], alpha=0.55, linewidth=1.0)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('BVP Amplitude', fontsize=12)
    ax.set_title(f'Group Cleaned BVP Profile - {group_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if num_subjects <= 20:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')

    plt.tight_layout()
    out_path = output_dir / f"bvp_profile_{group_name}.jpg"
    fig.savefig(out_path, dpi=ROLLING_GROUP_PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def cluster_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Apply clustering algorithms to analyze feature distribution (Sprint #2).

    NOT IMPLEMENTED YET.

    Planned approach:
        1. Standardize feature columns with StandardScaler
        2. Apply DBSCAN or KMeans clustering
        3. Return DataFrame with cluster labels appended

    Args:
        merged_df: DataFrame from merge_eda_and_ppg_features().

    Returns:
        DataFrame with additional 'cluster' column.

    Outputs:
        None (no files saved).
    """
    pass


def plot_cluster_results(clustered_df: pd.DataFrame) -> None:
    """Visualize clustering results on scatter plots (Sprint #2).

    NOT IMPLEMENTED YET.

    Planned approach:
        1. Color points by cluster label
        2. Optionally draw cluster boundaries or convex hulls

    Args:
        clustered_df: DataFrame from cluster_features() with 'cluster' column.

    Returns:
        None.

    Outputs:
        TBD - cluster visualization plots.
    """
    pass


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", message="EDA signal is sampled at very low frequency"
    )

    # Step 0: Read dataset path and manifest (read once, pass to functions)
    dataset_path = dmc.read_pathfile_or_ask_for_path()
    subject_group_map = dmc.read_master_manifest(dataset_path)

    # Step 1: Extract EDA features for all (subject, task) pairs
    eda_features = extract_eda_features(dataset_path, subject_group_map)

    # Step 2: Extract PPG features (HR/HRV) for all (subject, task) pairs
    ppg_features = extract_ppg_features(dataset_path, subject_group_map)

    # Step 2b: Extract rPPG HRV features (from rppg_signals JSON)
    rppg_features = extract_rppg_features(dataset_path, subject_group_map)

    # Step 3: Merge EDA, PPG, and rPPG features into single DataFrame
    merged_df = merge_eda_and_ppg_features(eda_features, ppg_features, rppg_features)

    # Step 4: Plot scatter plots (tonic vs HR/HRV)
    plot_scatter_tonic_vs_hr_hrv(merged_df)

    # Step 5: Plot phasic variance distribution by task
    plot_strip_phasic_by_task(merged_df)

    # Step 5b: Plot phasic standard deviation distribution by task (New Request)
    plot_box_phasic_std_by_task(merged_df)

    # Step 5c: Plot HRV RMSSD distribution by task (New Request)
    plot_box_rmssd_by_task(merged_df)

    # Step 5c1: Plot median heartbeat rate distribution by task
    plot_box_hr_median_by_task(merged_df)

    # Step 5c2: Plot rPPG HRV RMSSD distribution by task (New Request)
    plot_box_rppg_rmssd_by_task(merged_df)

    # Step 5d: Plot interaction feature (Tonic Slope * Phasic Std) - Professor Request
    plot_box_interaction_feature(merged_df)

    # Step 6: Plot interactive 3D EDA feature space (Plotly)
    plot_3d_eda_feature_space(merged_df)

    # Step 7: Plot HRV distribution by task
    plot_strip_hrv_by_task(merged_df)

    # Step 7b: Plot Subject HRV Profile (Rolling RMSSD)
    print("[INFO] Generating per-subject HRV profiles...")
    plot_subject_rolling_hrv(dataset_path, subject_group_map)

    # Step 7c: Plot Group HRV Profiles (Batched)
    print("[INFO] Generating group HRV profiles...")
    subject_paths = dmc.list_subject_paths(dataset_path)
    
    # Batch of 8
    chunk_size_8 = 8
    for i in range(0, len(subject_paths), chunk_size_8):
        group = subject_paths[i:i + chunk_size_8]
        group_name = f"Group_8_Batch_{i // chunk_size_8 + 1}"
        plot_group_rolling_hrv(group, group_name, subject_group_map)
        
    # Batch of 14
    chunk_size_14 = 14
    for i in range(0, len(subject_paths), chunk_size_14):
        group = subject_paths[i:i + chunk_size_14]
        group_name = f"Group_14_Batch_{i // chunk_size_14 + 1}"
        plot_group_rolling_hrv(group, group_name, subject_group_map)

    # All subjects
    plot_group_rolling_hrv(subject_paths, "All_Subjects", subject_group_map)

    # Step 7d: Plot Subject HR Profile (Rolling HR)
    print("[INFO] Generating per-subject HR profiles...")
    plot_subject_rolling_hr(dataset_path, subject_group_map)

    # Step 7e: Plot Group HR Profiles (Batched)
    print("[INFO] Generating group HR profiles...")

    # Batch of 8
    for i in range(0, len(subject_paths), chunk_size_8):
        group = subject_paths[i:i + chunk_size_8]
        group_name = f"Group_8_Batch_{i // chunk_size_8 + 1}"
        plot_group_rolling_hr(group, group_name, subject_group_map)

    # Batch of 14
    for i in range(0, len(subject_paths), chunk_size_14):
        group = subject_paths[i:i + chunk_size_14]
        group_name = f"Group_14_Batch_{i // chunk_size_14 + 1}"
        plot_group_rolling_hr(group, group_name, subject_group_map)

    # All subjects
    plot_group_rolling_hr(subject_paths, "All_Subjects", subject_group_map)

    # Step 7f: Plot Subject Raw BVP Profile
    print("[INFO] Generating per-subject BVP profiles...")
    plot_subject_bvp_profile(dataset_path, subject_group_map)

    # Step 7g: Plot Group Raw BVP Profiles (Batched)
    print("[INFO] Generating group BVP profiles...")

    # Batch of 8
    for i in range(0, len(subject_paths), chunk_size_8):
        group = subject_paths[i:i + chunk_size_8]
        group_name = f"Group_8_Batch_{i // chunk_size_8 + 1}"
        plot_group_bvp_profile(group, group_name, subject_group_map)

    # Batch of 14
    for i in range(0, len(subject_paths), chunk_size_14):
        group = subject_paths[i:i + chunk_size_14]
        group_name = f"Group_14_Batch_{i // chunk_size_14 + 1}"
        plot_group_bvp_profile(group, group_name, subject_group_map)

    # All subjects
    plot_group_bvp_profile(subject_paths, "All_Subjects", subject_group_map)

    # --- Sprint #2 (Not implemented yet) ---
    # Step 8: Clustering analysis
    # clustered_df = cluster_features(merged_df)
    # Step 9: Plot clustering results
    # plot_cluster_results(clustered_df)
