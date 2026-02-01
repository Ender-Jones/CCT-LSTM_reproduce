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
OUT_BOX_INTERACTION = "box_Tonic_slop_x_Phasic_std.jpg"
OUT_3D_EDA = "scatter3d_tonic_phasic_slope.html"

# Outlier removal threshold (percentile)
OUTLIER_PERCENTILE = 85


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
            - hr_mean, hrv_sdnn, hrv_rmssd

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

                # get the hr mean
                hr_mean = signals['PPG_Rate'].mean()

                # get the peaks for hrv calculation
                peaks = info['PPG_Peaks']
                if len(peaks) < 5:
                    print(f"[WARN] {bvp_path}: Not enough peaks ({len(peaks)}) for HRV. Skipping.")
                    continue

                # calculate the hrv
                hrv_df = nk.hrv_time(peaks, sampling_rate=dmc.BVP_SAMPLING_RATE_HZ)
                hrv_sdnn = hrv_df['HRV_SDNN'].iloc[0]
                hrv_rmssd = hrv_df['HRV_RMSSD'].iloc[0]

                result.append({
                    'subject': subject_id,
                    'task': task_id,
                    'group': group,
                    'task_label': task_label,
                    'hr_mean': hr_mean,
                    'hrv_sdnn': hrv_sdnn,
                    'hrv_rmssd': hrv_rmssd,
                })
            except Exception as e:
                print(f"[WARN] {bvp_path}: Failed to process BVP: {e}. Skipping.")
                continue

    print(f"[INFO] Extracted HR/HRV features from {len(result)} records.")
    return result


def merge_eda_and_ppg_features(eda_features: list[dict], ppg_features: list[dict]) -> pd.DataFrame:
    """Merge EDA and PPG feature lists into a single DataFrame.

    Performs inner join on (subject, task, group, task_label) keys and saves
    the merged DataFrame to CSV for downstream analysis.

    Args:
        eda_features: List of dicts from extract_eda_features().
        ppg_features: List of dicts from extract_ppg_features().

    Returns:
        Merged DataFrame with columns:
            [subject, task, group, task_label, tonic_mean, tonic_median, tonic_slope,
             phasic_var, phasic_std, hr_mean, hrv_sdnn, hrv_rmssd]

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
        data=merged_df,
        x='task_label',
        y='phasic_std',
        palette=task_palette,
        order=hue_order,
        showfliers=False,  # Hide outliers to avoid duplication with strip plot
        ax=ax
    )
    
    # Strip plot overlay for individual points
    sns.stripplot(
        data=merged_df,
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
    ax.set_title('EDA Phasic Std Distribution by Task', fontsize=14, fontweight='bold')

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
        y='hrv_rmssd',
        palette=task_palette,
        order=hue_order,
        showfliers=False,
        ax=ax
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
        y='interaction_feature',
        palette=task_palette,
        order=hue_order,
        showfliers=False,
        ax=ax
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


def calculate_rolling_rmssd(bvp_signal: list[float], fs: int, window_sec: int = 30, step_sec: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Calculate rolling RMSSD from BVP signal.

    1. Cleans BVP signal and finds peaks.
    2. Calculates RR intervals.
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

    if len(peaks) < 2:
        return np.array([]), np.array([])

    # Calculate NN intervals in ms
    r_peaks_ms = peaks / fs * 1000
    rr_intervals = np.diff(r_peaks_ms)
    # Time of each RR interval (assigned to end of interval)
    rr_times_ms = r_peaks_ms[1:]

    # Rolling window
    signal_duration_sec = len(bvp_signal) / fs
    time_points = np.arange(0, signal_duration_sec, step_sec)
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

        # Calculate rolling RMSSD (30s window)
        times, rmssd = calculate_rolling_rmssd(
            bvp_data, dmc.BVP_SAMPLING_RATE_HZ, window_sec=30, step_sec=1
        )

        if len(times) == 0:
            # Fallback for empty/failed processing
            times = np.arange(0, duration, 1)
            rmssd = np.full(len(times), np.nan)

        # Shift times by current cumulative offset
        all_times.append(times + current_offset)
        all_rmssd.append(rmssd)

        current_offset += duration
        boundaries.append(current_offset)

    full_time = np.concatenate(all_times)
    full_rmssd = np.concatenate(all_rmssd)

    return full_time, full_rmssd, boundaries


def plot_subject_rolling_hrv(dataset_path: Path, subject_group_map: dict[str, str]) -> None:
    """Generate per-subject plots of rolling RMSSD across T1-T3.

    Calculates RMSSD in a sliding window (30s) for each task and concatenates
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
        
        ax.plot(full_time, full_rmssd, color='#E91E63', linewidth=2, label='RMSSD (30s window)')

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
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    print(f"[INFO] Saved per-subject HRV profiles to {output_dir}")


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
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
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

    # Step 3: Merge EDA and PPG features into single DataFrame
    merged_df = merge_eda_and_ppg_features(eda_features, ppg_features)

    # Step 4: Plot scatter plots (tonic vs HR/HRV)
    plot_scatter_tonic_vs_hr_hrv(merged_df)

    # Step 5: Plot phasic variance distribution by task
    plot_strip_phasic_by_task(merged_df)

    # Step 5b: Plot phasic standard deviation distribution by task (New Request)
    plot_box_phasic_std_by_task(merged_df)

    # Step 5c: Plot HRV RMSSD distribution by task (New Request)
    plot_box_rmssd_by_task(merged_df)

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

    # --- Sprint #2 (Not implemented yet) ---
    # Step 8: Clustering analysis
    # clustered_df = cluster_features(merged_df)
    # Step 9: Plot clustering results
    # plot_cluster_results(clustered_df)
