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
from scipy.stats import linregress, pearsonr

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
OUT_CSV_RPPG_PPG_CORRELATION = "rppg_vs_ppg_hr_correlation.csv"
OUT_SCATTER_RPPG_PPG_PEARSON = "scatter_rppg_vs_ppg_pearson_r.jpg"
OUT_SCATTER_RPPG_PPG_MAE = "scatter_rppg_vs_ppg_mae.jpg"

# Outlier removal threshold (percentile)
OUTLIER_PERCENTILE = 90

# Shared plotting style for 5-class task labels
TASK_PALETTE = {
    'T1':      '#4CAF50',  # green - baseline
    'T2-ctrl': '#90CAF9',  # light blue - easy task
    'T2-test': '#1565C0',  # dark blue - hard task
    'T3-ctrl': '#EF9A9A',  # light red - easy task
    'T3-test': '#C62828',  # dark red - hard task
}
TASK_HUE_ORDER = ['T1', 'T2-ctrl', 'T2-test', 'T3-ctrl', 'T3-test']
TASK_MARKER_MAP = {
    'T1':      'o',
    'T2-ctrl': 'o',
    'T2-test': '^',
    'T3-ctrl': 'o',
    'T3-test': '^',
}

# Rolling profile parameters for HR/HRV line plots
ROLLING_WINDOW_SEC = 3
# Stride is defined in BVP samples to avoid mixing with "Hz" terminology.
ROLLING_STRIDE_SAMPLES = 1
ROLLING_BVP_STEP_SEC = ROLLING_STRIDE_SAMPLES / dmc.BVP_SAMPLING_RATE_HZ
# Rolling profile export DPI
ROLLING_SUBJECT_PLOT_DPI = 300
ROLLING_GROUP_PLOT_DPI = 500


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


def annotate_outliers(
    fig: plt.Figure,
    outliers: pd.DataFrame,
    *,
    threshold_text: str,
    value_col: str,
    value_format: str,
) -> None:
    """Render outlier details below a plot."""
    if outliers.empty:
        return

    outlier_lines = [threshold_text]
    for _, row in outliers.iterrows():
        formatted_value = format(row[value_col], value_format)
        outlier_lines.append(f"  {row['subject']} ({row['task_label']}): {formatted_value}")
    outlier_text = "\n".join(outlier_lines)

    fig.text(
        0.12, -0.02,
        outlier_text,
        fontsize=9,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='#FFB300', alpha=0.9),
    )


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
            palette=TASK_PALETTE,
            markers=TASK_MARKER_MAP,
            hue_order=TASK_HUE_ORDER,
            style_order=TASK_HUE_ORDER,
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

        annotate_outliers(
            fig,
            outliers,
            threshold_text=f"Outliers (>{OUTLIER_PERCENTILE}%ile, Y>{y_upper_limit:.0f}):",
            value_col=y_col,
            value_format=".1f",
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

    x_upper_limit = merged_df['phasic_var'].quantile(OUTLIER_PERCENTILE / 100)

    # Split data into normal points and outliers
    plot_df = merged_df[merged_df['phasic_var'] <= x_upper_limit]
    outliers = merged_df[merged_df['phasic_var'] > x_upper_limit]

    fig, ax = plt.subplots(figsize=(24, 6))
    sns.stripplot(
        data=plot_df,
        x='phasic_var',
        y='task_label',
        hue='task_label',
        palette=TASK_PALETTE,
        order=TASK_HUE_ORDER,
        hue_order=TASK_HUE_ORDER,
        size=8,
        alpha=0.7,
        jitter=0.25,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel('EDA Phasic Variance (µS²)', fontsize=12)
    ax.set_ylabel('Task-Group', fontsize=12)
    ax.set_title('EDA Phasic Variance Distribution by Task', fontsize=14, fontweight='bold')

    annotate_outliers(
        fig,
        outliers,
        threshold_text=f"Outliers (>{OUTLIER_PERCENTILE}%ile, X>{x_upper_limit:.4f}):",
        value_col='phasic_var',
        value_format=".4f",
    )

    plt.tight_layout()

    output_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_STRIP_PHASIC
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def plot_box_metric_by_task(
    merged_df: pd.DataFrame,
    *,
    y_col: str,
    y_label: str,
    title: str,
    output_filename: str,
    missing_warn: str,
    outlier_label: str,
    filter_outliers: bool = True,
    showfliers: bool = False,
) -> None:
    """Plot a generic task-wise box plot with strip overlay."""
    if merged_df.empty or y_col not in merged_df.columns:
        print(missing_warn)
        return

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = merged_df
    if filter_outliers:
        y_upper_limit = merged_df[y_col].quantile(OUTLIER_PERCENTILE / 100)
        plot_df = merged_df[merged_df[y_col] <= y_upper_limit].copy()
        outlier_count = len(merged_df) - len(plot_df)
        if outlier_count > 0:
            print(
                f"[INFO] Excluded {outlier_count} outliers "
                f"(>{OUTLIER_PERCENTILE}th percentile, >{y_upper_limit:.2f}) "
                f"for {outlier_label} box plot."
            )

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.boxplot(
        data=plot_df,
        x='task_label',
        hue='task_label',
        y=y_col,
        palette=TASK_PALETTE,
        order=TASK_HUE_ORDER,
        hue_order=TASK_HUE_ORDER,
        showfliers=showfliers,
        ax=ax,
        legend=False,
    )

    sns.stripplot(
        data=plot_df,
        x='task_label',
        y=y_col,
        color='black',
        order=TASK_HUE_ORDER,
        size=4,
        alpha=0.3,
        jitter=True,
        ax=ax,
    )

    ax.set_xlabel('Task-Group', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = dmc.DATA_MINING_OUTPUT_DIR / output_filename
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def plot_box_phasic_std_by_task(merged_df: pd.DataFrame) -> None:
    """Plot box plot showing EDA phasic standard deviation distribution by task_label."""
    plot_box_metric_by_task(
        merged_df,
        y_col='phasic_std',
        y_label='EDA Phasic Standard Deviation (µS)',
        title=f'EDA Phasic Std Distribution by Task (Excl. top {100-OUTLIER_PERCENTILE}%)',
        output_filename=OUT_BOX_PHASIC_STD,
        missing_warn='[WARN] No phasic_std data available. Skipping box plot.',
        outlier_label='Phasic Std',
        filter_outliers=True,
        showfliers=False,
    )


def plot_box_rmssd_by_task(merged_df: pd.DataFrame) -> None:
    """Plot box plot showing HRV RMSSD distribution by task_label."""
    plot_box_metric_by_task(
        merged_df,
        y_col='hrv_rmssd',
        y_label='HRV RMSSD (ms)',
        title=f'HRV RMSSD Distribution by Task (Excl. top {100-OUTLIER_PERCENTILE}%)',
        output_filename=OUT_BOX_RMSSD,
        missing_warn='[WARN] No hrv_rmssd data available. Skipping box plot.',
        outlier_label='RMSSD',
        filter_outliers=True,
        showfliers=False,
    )


def plot_box_rppg_rmssd_by_task(merged_df: pd.DataFrame) -> None:
    """Plot box plot showing rPPG HRV RMSSD distribution by task_label."""
    plot_box_metric_by_task(
        merged_df,
        y_col='rppg_hrv_rmssd',
        y_label='rPPG HRV RMSSD (ms)',
        title=f'rPPG HRV RMSSD Distribution by Task (Excl. top {100-OUTLIER_PERCENTILE}%)',
        output_filename=OUT_BOX_RPPG_RMSSD,
        missing_warn='[WARN] No rppg_hrv_rmssd data available. Skipping box plot.',
        outlier_label='rPPG RMSSD',
        filter_outliers=True,
        showfliers=False,
    )


def plot_box_hr_median_by_task(merged_df: pd.DataFrame) -> None:
    """Plot box plot of per-subject-task median heartbeat rate by task_label."""
    plot_box_metric_by_task(
        merged_df,
        y_col='hr_median',
        y_label='Median Heart Rate (bpm)',
        title='Median Heart Rate Distribution by Task',
        output_filename=OUT_BOX_HR_MEDIAN,
        missing_warn='[WARN] No hr_median data available. Skipping HR median box plot.',
        outlier_label='HR median',
        filter_outliers=False,
        showfliers=True,
    )


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
            palette=TASK_PALETTE,
            order=TASK_HUE_ORDER,
            hue_order=TASK_HUE_ORDER,
            size=8,
            alpha=0.7,
            jitter=0.25,
            ax=ax,
            legend=False,
        )

        ax.set_xlabel('Task-Group', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{y_label} Distribution by Task', fontsize=14, fontweight='bold')

        annotate_outliers(
            fig,
            outliers,
            threshold_text=f"Outliers (>{OUTLIER_PERCENTILE}%ile, Y>{y_upper_limit:.0f}):",
            value_col=y_col,
            value_format=".1f",
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

    fig, ax = plt.subplots(figsize=(10, 7))

    # Box plot
    sns.boxplot(
        data=merged_df,
        x='task_label',
        hue='task_label',
        y='interaction_feature',
        palette=TASK_PALETTE,
        order=TASK_HUE_ORDER,
        hue_order=TASK_HUE_ORDER,
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
        order=TASK_HUE_ORDER,
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

    # Symbol mapping: ctrl=circle, test=diamond, T1=circle
    symbol_map = {
        'T1':      'circle',
        'T2-ctrl': 'circle',
        'T2-test': 'diamond',
        'T3-ctrl': 'circle',
        'T3-test': 'diamond',
    }

    fig = px.scatter_3d(
        merged_df,
        x='tonic_mean',
        y='phasic_std',
        z='tonic_slope',
        color='task_label',
        symbol='task_label',
        color_discrete_map=TASK_PALETTE,
        symbol_map=symbol_map,
        category_orders={'task_label': TASK_HUE_ORDER},
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


def _prepare_rr_series_and_windows(
    bvp_signal: list[float],
    fs: int,
    window_sec: int,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare cleaned RR intervals and rolling window start points."""
    try:
        cleaned = nk.ppg_clean(bvp_signal, sampling_rate=fs)
        peaks_dict = nk.ppg_findpeaks(cleaned, sampling_rate=fs)
        peaks = peaks_dict['PPG_Peaks']
    except Exception:
        return np.array([]), np.array([]), np.array([])

    if len(peaks) < 5:
        return np.array([]), np.array([]), np.array([])

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
        return np.array([]), np.array([]), np.array([])

    rr_intervals = np.asarray(rri_dict["RRI"], dtype=float)
    rr_times_ms = np.asarray(rri_dict["RRI_Time"], dtype=float) * 1000.0

    # Only generate complete windows: n = floor((duration - window) / step) + 1
    signal_duration_sec = len(bvp_signal) / fs
    max_start = signal_duration_sec - window_sec
    if max_start < 0:
        return np.array([]), np.array([]), np.array([])
    n_windows = int(np.floor(max_start / step_sec)) + 1
    time_points = np.arange(n_windows) * step_sec
    return rr_intervals, rr_times_ms, time_points


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
    rr_intervals, rr_times_ms, time_points = _prepare_rr_series_and_windows(
        bvp_signal,
        fs,
        window_sec,
        step_sec,
    )
    if len(time_points) == 0:
        return np.array([]), np.array([])

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
    rr_intervals, rr_times_ms, time_points = _prepare_rr_series_and_windows(
        bvp_signal,
        fs,
        window_sec,
        step_sec,
    )
    if len(time_points) == 0:
        return np.array([]), np.array([])
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


def _get_subject_rolling_profile(
    subject_path: Path,
    metric_fn,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Calculate a concatenated rolling metric profile for a subject across T1-T3."""
    bvp_paths = dmc.list_bvp_paths(subject_path)
    if not dmc.has_expected_bvp_files(bvp_paths, subject_path):
        return np.array([]), np.array([]), []

    all_times = []
    all_values = []
    boundaries = []
    current_offset = 0

    # Process T1, T2, T3 in order
    for bvp_path in bvp_paths:
        bvp_data = dmc.read_bvp_data(bvp_path)
        duration = len(bvp_data) / dmc.BVP_SAMPLING_RATE_HZ

        times, metric_values = metric_fn(
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
            metric_values = np.full(len(times), np.nan)

        # Shift times by current cumulative offset
        all_times.append(times + current_offset)
        all_values.append(metric_values)

        current_offset += duration
        boundaries.append(current_offset)

    full_time = np.concatenate(all_times)
    full_values = np.concatenate(all_values)
    return full_time, full_values, boundaries


def get_subject_hrv_profile(subject_path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Calculate concatenated rolling RMSSD for a subject across T1-T3."""
    return _get_subject_rolling_profile(subject_path, calculate_rolling_rmssd)


def get_subject_hr_profile(subject_path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Calculate concatenated rolling HR for a subject across T1-T3."""
    return _get_subject_rolling_profile(subject_path, calculate_rolling_hr)


def _reconstruct_rppg_bvp_overlap_average(
    rppg_path: Path,
) -> tuple[np.ndarray, int, float] | tuple[None, None, None]:
    """Reconstruct a continuous BVP signal from overlapping rPPG windows via overlap-average.

    Background
    ----------
    pyVHR extracts rPPG BVP using a sliding window over the video. For example,
    a typical configuration is a 60-second window with a 5-second stride, which
    means ~91% of the signal overlaps between consecutive windows.  The output
    JSON stores each window as a separate ``bvp_signal`` array together with
    ``start_frame`` / ``end_frame`` / ``fps`` metadata.

    Naively concatenating these windows produces a signal ~8x longer than the
    real video — which is wrong.  Naively picking only one window per stride
    wastes most of the extracted information.

    Algorithm (Overlap-Average)
    ---------------------------
    1. Determine the total timeline length in **samples** from the last window's
       ``end_frame`` (all windows share the same ``fps``).
    2. Allocate two float arrays of that length: ``sum_buf`` (accumulator) and
       ``count_buf`` (per-sample contribution counter).
    3. For each window, add its BVP samples into ``sum_buf`` at the correct
       position (``start_frame``).  Increment ``count_buf`` at those positions.
    4. Divide ``sum_buf / count_buf`` element-wise to get the overlap-averaged
       BVP signal.  Positions with zero contributions stay as NaN.

    This produces a single continuous BVP waveform at the original video fps,
    with the same duration as the real video.  The dense signal can then be
    processed identically to PPG BVP data (ppg_clean → peaks → rolling HR),
    giving a comparable number of data points for visual comparison.

    Args:
        rppg_path: Path to a single rPPG JSON file (one task, e.g. vid_s1_T1_rppg.json).

    Returns:
        (bvp_signal, sampling_rate_int, task_duration_sec) on success.
        (None, None, None) if the file cannot be processed.
    """
    try:
        with rppg_path.open("r", encoding="utf-8") as f:
            rppg_data = json.load(f)
    except Exception as e:
        print(f"[WARN] {rppg_path}: Failed to read rPPG JSON: {e}. Skipping.")
        return None, None, None

    if not isinstance(rppg_data, list) or len(rppg_data) == 0:
        return None, None, None

    # First pass: determine consistent fps and total length
    fs_int: int | None = None
    total_samples = 0

    for window in rppg_data:
        if not isinstance(window, dict):
            continue
        fps = window.get('fps=sampling_rate', window.get('fps'))
        end_frame = window.get('end_frame')
        if fps is None or end_frame is None:
            continue
        fs = float(fps)
        if not np.isfinite(fs) or fs <= 0:
            continue
        window_fs = int(round(fs))
        if fs_int is None:
            fs_int = window_fs
        elif window_fs != fs_int:
            continue
        # end_frame is the last frame index; +1 for total sample count
        total_samples = max(total_samples, int(end_frame) + 1)

    if fs_int is None or total_samples == 0:
        print(f"[WARN] {rppg_path}: No valid rPPG windows found. Skipping.")
        return None, None, None

    # Overlap-average accumulation
    sum_buf = np.zeros(total_samples, dtype=np.float64)
    count_buf = np.zeros(total_samples, dtype=np.float64)

    for window in rppg_data:
        if not isinstance(window, dict):
            continue
        bvp_signal = window.get('bvp_signal', [])
        fps = window.get('fps=sampling_rate', window.get('fps'))
        start_frame = window.get('start_frame')

        if not bvp_signal or fps is None or start_frame is None:
            continue
        if int(round(float(fps))) != fs_int:
            continue

        sf = int(start_frame)
        n = len(bvp_signal)
        end_idx = min(sf + n, total_samples)
        usable = end_idx - sf
        if usable <= 0:
            continue

        arr = np.asarray(bvp_signal[:usable], dtype=np.float64)
        sum_buf[sf:end_idx] += arr
        count_buf[sf:end_idx] += 1.0

    # Average where we have contributions; leave NaN where we don't
    with np.errstate(invalid='ignore', divide='ignore'):
        reconstructed = np.where(count_buf > 0, sum_buf / count_buf, np.nan)

    task_duration_sec = total_samples / fs_int
    return reconstructed, fs_int, task_duration_sec


def get_subject_rppg_hr_profile(subject_path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Calculate concatenated rolling HR from overlap-averaged rPPG BVP across T1-T3.

    Uses _reconstruct_rppg_bvp_overlap_average to produce a dense continuous BVP
    signal, then feeds it through calculate_rolling_hr (same pipeline as PPG).
    """
    rppg_paths = dmc.list_rppg_paths(subject_path)
    if len(rppg_paths) != 3:
        print(
            f"[WARN] {subject_path}: expected 3 rPPG files (T1/T2/T3), got {len(rppg_paths)}. Skipping."
        )
        return np.array([]), np.array([]), []

    all_times: list[np.ndarray] = []
    all_hr: list[np.ndarray] = []
    boundaries: list[float] = []
    current_offset = 0.0

    for rppg_path in rppg_paths:
        bvp_signal, fs_int, task_duration = _reconstruct_rppg_bvp_overlap_average(rppg_path)
        if bvp_signal is None or fs_int is None or task_duration is None:
            return np.array([]), np.array([]), []

        times, hr = calculate_rolling_hr(
            bvp_signal.tolist(),
            fs_int,
            window_sec=ROLLING_WINDOW_SEC,
            step_sec=ROLLING_BVP_STEP_SEC,
        )

        if len(times) == 0:
            max_start_fb = task_duration - ROLLING_WINDOW_SEC
            n_fb = max(int(np.floor(max_start_fb / ROLLING_BVP_STEP_SEC)) + 1, 0) if max_start_fb >= 0 else 0
            times = np.arange(n_fb) * ROLLING_BVP_STEP_SEC
            hr = np.full(len(times), np.nan)

        all_times.append(times + current_offset)
        all_hr.append(hr)
        current_offset += task_duration
        boundaries.append(current_offset)

    if not all_times:
        return np.array([]), np.array([]), []

    full_time = np.concatenate(all_times)
    full_hr = np.concatenate(all_hr)
    return full_time, full_hr, boundaries


def _add_task_boundary_annotations(ax: plt.Axes, boundaries: list[float]) -> None:
    """Add vertical separators and T1/T2/T3 labels."""
    for boundary in boundaries[:-1]:
        ax.axvline(x=boundary, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

    segment_starts = [0] + boundaries[:-1]
    segment_ends = boundaries
    labels = ['T1', 'T2', 'T3']

    y_lims = ax.get_ylim()
    for start, end, label in zip(segment_starts, segment_ends, labels):
        mid_point = (start + end) / 2
        ax.text(mid_point, y_lims[1], label, ha='center', va='bottom', fontweight='bold', fontsize=12)


def _plot_subject_metric_profiles(
    dataset_path: Path,
    subject_group_map: dict[str, str],
    *,
    profile_fn,
    output_subdir: str,
    output_prefix: str,
    line_color: str,
    line_width: float,
    legend_label: str,
    y_label: str,
    title_suffix: str,
    save_csv: bool = False,
    csv_value_col: str | None = None,
) -> None:
    """Generate per-subject profile plots using a shared pipeline."""
    output_dir = dmc.DATA_MINING_OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_path in dmc.list_subject_paths(dataset_path):
        subject_id = subject_path.name
        group = subject_group_map.get(subject_id, 'unknown')

        full_time, full_values, boundaries = profile_fn(subject_path)
        if len(full_time) == 0:
            continue

        fig, ax = plt.subplots(figsize=(24, 6))
        ax.plot(full_time, full_values, color=line_color, linewidth=line_width, label=legend_label)
        _add_task_boundary_annotations(ax, boundaries)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'Subject {subject_id} ({group}) - {title_suffix}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        out_path = output_dir / f"{output_prefix}_{subject_id}.jpg"
        fig.savefig(out_path, dpi=ROLLING_SUBJECT_PLOT_DPI, bbox_inches='tight')
        plt.close(fig)

        if save_csv and csv_value_col is not None:
            csv_path = out_path.with_suffix(".csv")
            profile_df = pd.DataFrame(
                {
                    "subject": subject_id,
                    "group": group,
                    "data_point_idx": np.arange(len(full_time)),
                    "window_start_sec": full_time,
                    "window_end_sec": full_time + ROLLING_WINDOW_SEC,
                    csv_value_col: full_values,
                }
            )
            profile_df.to_csv(csv_path, index=False)

    print(f"[INFO] Saved per-subject profiles to {output_dir}")


def _plot_group_metric_profiles(
    subject_paths: list[Path],
    group_name: str,
    subject_group_map: dict[str, str],
    *,
    profile_fn,
    output_subdir: str,
    output_prefix: str,
    y_label: str,
    title_prefix: str,
    line_alpha: float,
    line_width: float,
    enable_grid: bool = True,
    save_csv: bool = False,
    csv_value_col: str | None = None,
) -> None:
    """Generate grouped profile plots using a shared pipeline."""
    output_dir = dmc.DATA_MINING_OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    num_subjects = len(subject_paths)
    colors = (
        plt.cm.tab20(np.linspace(0, 1, num_subjects))
        if num_subjects <= 20 else plt.cm.jet(np.linspace(0, 1, num_subjects))
    )

    has_data = False
    curve_rows: list[pd.DataFrame] = []

    for i, subject_path in enumerate(subject_paths):
        subject_id = subject_path.name
        subject_group = subject_group_map.get(subject_id, "unknown")
        full_time, full_values, _ = profile_fn(subject_path)

        if len(full_time) == 0:
            continue

        has_data = True
        ax.plot(full_time, full_values, label=subject_id, color=colors[i], alpha=line_alpha, linewidth=line_width)

        if save_csv and csv_value_col is not None:
            curve_rows.append(
                pd.DataFrame(
                    {
                        "group_name": group_name,
                        "subject": subject_id,
                        "group": subject_group,
                        "data_point_idx": np.arange(len(full_time)),
                        "window_start_sec": full_time,
                        "window_end_sec": full_time + ROLLING_WINDOW_SEC,
                        csv_value_col: full_values,
                    }
                )
            )

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{title_prefix} - {group_name}', fontsize=14, fontweight='bold')
    if enable_grid:
        ax.grid(True, alpha=0.3)
    if num_subjects <= 20:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')

    plt.tight_layout()
    out_path = output_dir / f"{output_prefix}_{group_name}.jpg"
    fig.savefig(out_path, dpi=ROLLING_GROUP_PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")

    if curve_rows:
        csv_path = out_path.with_suffix(".csv")
        group_curve_df = pd.concat(curve_rows, ignore_index=True)
        group_curve_df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved {csv_path}")


def run_batched_group_plots(
    subject_paths: list[Path],
    subject_group_map: dict[str, str],
    plot_fn,
    *,
    chunk_sizes: tuple[int, ...] = (8, 14),
) -> None:
    """Run group plots for several batch sizes and all-subject view."""
    for chunk_size in chunk_sizes:
        for i in range(0, len(subject_paths), chunk_size):
            group = subject_paths[i:i + chunk_size]
            group_name = f"Group_{chunk_size}_Batch_{i // chunk_size + 1}"
            plot_fn(group, group_name, subject_group_map)
    plot_fn(subject_paths, "All_Subjects", subject_group_map)


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
    _plot_subject_metric_profiles(
        dataset_path,
        subject_group_map,
        profile_fn=get_subject_hrv_profile,
        output_subdir="subject_hrv_profiles",
        output_prefix="hrv_profile",
        line_color='#E91E63',
        line_width=2.0,
        legend_label=(
            f'RMSSD ({ROLLING_WINDOW_SEC}s window, '
            f'stride={ROLLING_STRIDE_SAMPLES} sample '
            f'({ROLLING_BVP_STEP_SEC:.5f}s))'
        ),
        y_label='HRV RMSSD (ms)',
        title_suffix='HRV RMSSD Profile',
        save_csv=False,
    )


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
    _plot_subject_metric_profiles(
        dataset_path,
        subject_group_map,
        profile_fn=get_subject_hr_profile,
        output_subdir="subject_hr_profiles",
        output_prefix="hr_profile",
        line_color='#1976D2',
        line_width=2.0,
        legend_label=(
            f'HR ({ROLLING_WINDOW_SEC}s window, '
            f'stride={ROLLING_STRIDE_SAMPLES} sample '
            f'({ROLLING_BVP_STEP_SEC:.5f}s))'
        ),
        y_label='Heart Rate (bpm)',
        title_suffix='HR Profile',
        save_csv=True,
        csv_value_col='hr_bpm',
    )


def plot_group_rolling_hrv(subject_paths: list[Path], group_name: str, subject_group_map: dict[str, str]) -> None:
    """Plot combined rolling RMSSD profiles for a group of subjects.
    
    Args:
        subject_paths: List of subject paths to include in the plot.
        group_name: Name of the group (used for filename and title).
        subject_group_map: Mapping of {subject_id: group}.
    """
    _plot_group_metric_profiles(
        subject_paths,
        group_name,
        subject_group_map,
        profile_fn=get_subject_hrv_profile,
        output_subdir="group_hrv_profiles",
        output_prefix="hrv_profile",
        y_label='HRV RMSSD (ms)',
        title_prefix='Group HRV RMSSD Profile',
        line_alpha=0.7,
        line_width=1.5,
        enable_grid=True,
        save_csv=False,
    )


def plot_group_rolling_hr(subject_paths: list[Path], group_name: str, subject_group_map: dict[str, str]) -> None:
    """Plot combined rolling HR profiles for a group of subjects.

    Args:
        subject_paths: List of subject paths to include in the plot.
        group_name: Name of the group (used for filename and title).
        subject_group_map: Mapping of {subject_id: group}.
    """
    _plot_group_metric_profiles(
        subject_paths,
        group_name,
        subject_group_map,
        profile_fn=get_subject_hr_profile,
        output_subdir="group_hr_profiles",
        output_prefix="hr_profile",
        y_label='Heart Rate (bpm)',
        title_prefix='Group HR Profile',
        line_alpha=0.7,
        line_width=1.5,
        enable_grid=False,
        save_csv=True,
        csv_value_col='hr_bpm',
    )


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
    _plot_subject_metric_profiles(
        dataset_path,
        subject_group_map,
        profile_fn=get_subject_bvp_profile,
        output_subdir="subject_bvp_profiles",
        output_prefix="bvp_profile",
        line_color='#7B1FA2',
        line_width=1.2,
        legend_label='Cleaned BVP',
        y_label='BVP Amplitude',
        title_suffix='Cleaned BVP Profile',
        save_csv=False,
    )


def plot_group_bvp_profile(subject_paths: list[Path], group_name: str, subject_group_map: dict[str, str]) -> None:
    """Plot combined cleaned BVP profiles for a group of subjects.

    Args:
        subject_paths: List of subject paths to include in the plot.
        group_name: Name of the group (used for filename and title).
        subject_group_map: Mapping of {subject_id: group}.
    """
    _plot_group_metric_profiles(
        subject_paths,
        group_name,
        subject_group_map,
        profile_fn=get_subject_bvp_profile,
        output_subdir="group_bvp_profiles",
        output_prefix="bvp_profile",
        y_label='BVP Amplitude',
        title_prefix='Group Cleaned BVP Profile',
        line_alpha=0.55,
        line_width=1.0,
        enable_grid=True,
        save_csv=False,
    )


def plot_subject_rolling_hr_rppg(dataset_path: Path, subject_group_map: dict[str, str]) -> None:
    """Generate per-subject rolling HR plots from overlap-averaged rPPG BVP."""
    _plot_subject_metric_profiles(
        dataset_path,
        subject_group_map,
        profile_fn=get_subject_rppg_hr_profile,
        output_subdir="subject_rppg_hr_profiles",
        output_prefix="rppg_hr_profile",
        line_color='#00897B',
        line_width=2.0,
        legend_label=(
            f'rPPG HR ({ROLLING_WINDOW_SEC}s window, '
            f'stride={ROLLING_STRIDE_SAMPLES} sample '
            f'({ROLLING_BVP_STEP_SEC:.5f}s))'
        ),
        y_label='Heart Rate (bpm)',
        title_suffix='rPPG HR Profile (overlap-avg)',
        save_csv=True,
        csv_value_col='hr_bpm',
    )


def plot_group_rolling_hr_rppg(subject_paths: list[Path], group_name: str, subject_group_map: dict[str, str]) -> None:
    """Plot combined rolling rPPG-based HR profiles for a group of subjects."""
    _plot_group_metric_profiles(
        subject_paths,
        group_name,
        subject_group_map,
        profile_fn=get_subject_rppg_hr_profile,
        output_subdir="group_rppg_hr_profiles",
        output_prefix="rppg_hr_profile",
        y_label='Heart Rate (bpm)',
        title_prefix='Group rPPG HR Profile',
        line_alpha=0.7,
        line_width=1.5,
        enable_grid=False,
        save_csv=True,
        csv_value_col='hr_bpm',
    )


def analyze_rppg_vs_ppg_hr_correlation(
    subject_group_map: dict[str, str],
) -> None:
    """Compute per-subject Pearson r and MAE between rPPG and PPG HR.

    Reads the pre-generated dense rPPG HR profile CSVs and PPG HR profile CSVs,
    aligns them by ``window_start_sec``, and computes agreement metrics for each
    subject.  Results are saved to a summary CSV (per-subject rows + overall
    mean) and visualised as scatter correlation plots (X=rPPG HR, Y=PPG HR).
    """
    rppg_dir = dmc.DATA_MINING_OUTPUT_DIR / "subject_rppg_hr_profiles"
    ppg_dir = dmc.DATA_MINING_OUTPUT_DIR / "subject_hr_profiles"

    if not rppg_dir.exists() or not ppg_dir.exists():
        print(
            "[WARN] rPPG or PPG HR profile directory not found. "
            "[WARN] Skipping rPPG-vs-PPG correlation analysis."
        )
        return

    results: list[dict] = []
    all_matched_rows: list[pd.DataFrame] = []

    for subject_id in sorted(subject_group_map.keys(), key=lambda s: int(s[1:])):
        group = subject_group_map[subject_id]

        rppg_csv = rppg_dir / f"rppg_hr_dense_profile_{subject_id}.csv"
        ppg_csv = ppg_dir / f"hr_profile_{subject_id}.csv"

        if not rppg_csv.exists() or not ppg_csv.exists():
            print(f"[WARN] Missing CSV for {subject_id}. Skipping.")
            continue

        rppg_df = pd.read_csv(rppg_csv)
        ppg_df = pd.read_csv(ppg_csv)

        rppg_df["t_key"] = rppg_df["window_start_sec"].round(6)
        ppg_df["t_key"] = ppg_df["window_start_sec"].round(6)

        merged = pd.merge(
            rppg_df[["t_key", "hr_bpm"]],
            ppg_df[["t_key", "hr_bpm"]],
            on="t_key",
            suffixes=("_rppg", "_ppg"),
        )
        merged = merged.dropna(subset=["hr_bpm_rppg", "hr_bpm_ppg"])

        if len(merged) < 10:
            print(f"[WARN] {subject_id}: only {len(merged)} matched points. Skipping.")
            continue

        rppg_hr = merged["hr_bpm_rppg"].values
        ppg_hr = merged["hr_bpm_ppg"].values

        r_val, _ = pearsonr(rppg_hr, ppg_hr)
        mae = float(np.mean(np.abs(rppg_hr - ppg_hr)))

        results.append({
            "subject": subject_id,
            "group": group,
            "pearson_r": round(r_val, 6),
            "mae_bpm": round(mae, 4),
            "n_matched_points": len(merged),
        })

        merged["subject"] = subject_id
        merged["group"] = group
        all_matched_rows.append(merged)

    if not results:
        print("[WARN] No valid subjects for rPPG vs PPG correlation. Skipping.")
        return

    results_df = pd.DataFrame(results)

    mean_row = pd.DataFrame([{
        "subject": "MEAN",
        "group": "",
        "pearson_r": round(results_df["pearson_r"].mean(), 6),
        "mae_bpm": round(results_df["mae_bpm"].mean(), 4),
        "n_matched_points": int(results_df["n_matched_points"].mean()),
    }])
    output_df = pd.concat([results_df, mean_row], ignore_index=True)

    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_CSV_RPPG_PPG_CORRELATION
    output_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved rPPG vs PPG HR correlation → {csv_path}")

    all_matched_df = pd.concat(all_matched_rows, ignore_index=True)
    _plot_rppg_ppg_correlation_scatter(all_matched_df, results_df)


def _plot_rppg_ppg_correlation_scatter(
    all_matched_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> None:
    """Draw scatter correlation plots for rPPG HR vs PPG HR.

    Creates two plots:
    1. Pearson r scatter: regression line + identity line, annotated with r.
    2. MAE scatter: identity line, annotated with MAE.

    Args:
        all_matched_df: DataFrame of all time-aligned (hr_bpm_rppg, hr_bpm_ppg,
            subject, group) rows pooled across subjects.
        results_df: Per-subject summary DataFrame (used for overall stats).
    """
    dmc.DATA_MINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    group_palette = {"ctrl": "#90CAF9", "test": "#EF9A9A"}
    group_order = ["ctrl", "test"]

    rppg_hr = all_matched_df["hr_bpm_rppg"].values
    ppg_hr = all_matched_df["hr_bpm_ppg"].values

    r_val, p_val = pearsonr(rppg_hr, ppg_hr)
    mae = float(np.mean(np.abs(rppg_hr - ppg_hr)))
    slope, intercept, _, _, _ = linregress(rppg_hr, ppg_hr)

    all_hr = np.concatenate([rppg_hr, ppg_hr])
    hr_min = float(np.nanmin(all_hr)) - 5
    hr_max = float(np.nanmax(all_hr)) + 5
    axis_range = np.array([hr_min, hr_max])

    # --- Plot 1: Pearson r scatter with regression line ---
    fig, ax = plt.subplots(figsize=(8, 8))

    for grp in group_order:
        mask = all_matched_df["group"] == grp
        if mask.any():
            ax.scatter(
                all_matched_df.loc[mask, "hr_bpm_rppg"],
                all_matched_df.loc[mask, "hr_bpm_ppg"],
                c=group_palette[grp],
                label=grp,
                s=8,
                alpha=0.15,
                edgecolors="none",
            )

    ax.plot(
        axis_range, axis_range,
        color="gray", linestyle="--", linewidth=1, alpha=0.6, label="y = x",
    )
    ax.plot(
        axis_range, slope * axis_range + intercept,
        color="#D32F2F", linewidth=2,
        label=f"y = {slope:.2f}x + {intercept:.1f}",
    )

    p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
    ax.text(
        0.05, 0.95,
        f"Pearson r = {r_val:.4f}\n{p_text}\nn = {len(rppg_hr):,}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("rPPG HR (bpm)", fontsize=12)
    ax.set_ylabel("PPG HR (bpm)", fontsize=12)
    ax.set_title("rPPG vs PPG HR — Pearson Correlation", fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_SCATTER_RPPG_PPG_PEARSON
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")

    # --- Plot 2: MAE scatter with identity line ---
    fig, ax = plt.subplots(figsize=(8, 8))

    for grp in group_order:
        mask = all_matched_df["group"] == grp
        if mask.any():
            ax.scatter(
                all_matched_df.loc[mask, "hr_bpm_rppg"],
                all_matched_df.loc[mask, "hr_bpm_ppg"],
                c=group_palette[grp],
                label=grp,
                s=8,
                alpha=0.15,
                edgecolors="none",
            )

    ax.plot(
        axis_range, axis_range,
        color="#D32F2F", linewidth=2, label="y = x (perfect agreement)",
    )

    ax.text(
        0.05, 0.95,
        f"MAE = {mae:.2f} bpm\nn = {len(rppg_hr):,}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("rPPG HR (bpm)", fontsize=12)
    ax.set_ylabel("PPG HR (bpm)", fontsize=12)
    ax.set_title("rPPG vs PPG HR — Mean Absolute Error", fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = dmc.DATA_MINING_OUTPUT_DIR / OUT_SCATTER_RPPG_PPG_MAE
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
    run_batched_group_plots(subject_paths, subject_group_map, plot_group_rolling_hrv)

    # Step 7d: Plot Subject HR Profile (Rolling HR)
    print("[INFO] Generating per-subject HR profiles...")
    plot_subject_rolling_hr(dataset_path, subject_group_map)

    # Step 7e: Plot Group HR Profiles (Batched)
    print("[INFO] Generating group HR profiles...")
    run_batched_group_plots(subject_paths, subject_group_map, plot_group_rolling_hr)

    # Step 7f: Plot Subject Raw BVP Profile
    print("[INFO] Generating per-subject BVP profiles...")
    plot_subject_bvp_profile(dataset_path, subject_group_map)

    # Step 7g: Plot Group Raw BVP Profiles (Batched)
    print("[INFO] Generating group BVP profiles...")
    run_batched_group_plots(subject_paths, subject_group_map, plot_group_bvp_profile)

    # Step 7h: Plot Subject rPPG-based HR Profile (overlap-averaged, dense)
    print("[INFO] Generating per-subject rPPG HR profiles...")
    plot_subject_rolling_hr_rppg(dataset_path, subject_group_map)

    # Step 7i: Plot Group rPPG-based HR Profiles (Batched)
    print("[INFO] Generating group rPPG HR profiles...")
    run_batched_group_plots(subject_paths, subject_group_map, plot_group_rolling_hr_rppg)

    # Step 8: rPPG vs PPG HR agreement analysis (Pearson r, MAE)
    print("[INFO] Analysing rPPG vs PPG HR correlation...")
    analyze_rppg_vs_ppg_hr_correlation(subject_group_map)

    # --- Sprint #2 (Not implemented yet) ---
    # Step 9: Clustering analysis
    # clustered_df = cluster_features(merged_df)
    # Step 10: Plot clustering results
    # plot_cluster_results(clustered_df)
