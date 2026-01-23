import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk

import data_mining_common as dmc

# T2: per-task scatter plots across subjects using `nk.eda_analyze` features.
# Goal: Verify whether SCR features differ significantly between tasks.
# my professor asking me to use sklearn to directly load and make data points for each eda file.

TASK_COLORS = {'T1': '#4CAF50', 'T2': '#FF9800', 'T3': '#F44336'}
TASK_LABELS = {'T1': 'T1: Rest', 'T2': 'T2: Speech', 'T3': 'T3: Math'}


def extract_eda_features(eda_data: list[float]) -> dict:
    """
    Extract EDA features from raw data using NeuroKit2's eda_analyze.
    Returns a dictionary of feature name -> value.
    """
    signals, info = nk.eda_process(eda_data, sampling_rate=dmc.EDA_SAMPLING_RATE_HZ)
    features_df = nk.eda_analyze(signals, sampling_rate=dmc.EDA_SAMPLING_RATE_HZ)
    return features_df.iloc[0].to_dict()


def collect_all_features(dataset_path: Path) -> pd.DataFrame:
    """
    Collect EDA features from all subjects and all tasks.
    Returns a DataFrame with columns: subject_id, task, and all feature columns.
    """
    all_data: list[dict] = []

    subject_paths = dmc.list_subject_paths(dataset_path)
    for subject_path in subject_paths:
        subject_id = subject_path.name
        eda_paths = dmc.list_eda_paths(subject_path)

        if not dmc.has_expected_eda_files(eda_paths, subject_path):
            continue

        for eda_path in eda_paths:
            # Extract task ID (T1, T2, T3) from filename like "eda_s1_T1.csv"
            task_id = eda_path.stem.split('_')[-1]

            eda_data = dmc.read_eda_data(eda_path)
            features = extract_eda_features(eda_data)

            # Add metadata
            features['subject_id'] = subject_id
            features['task'] = task_id

            all_data.append(features)

    return pd.DataFrame(all_data)


def create_scatter_plot(feature_name: str):
    """
    Create the scatter plot figure and axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    return fig, ax


def plot_scatter_with_jitter(ax, df: pd.DataFrame, feature_name: str) -> None:
    """
    Plot scatter points with jitter for each task.
    Jitter prevents overlapping points when X-axis is categorical.
    """
    task_to_num = {'T1': 0, 'T2': 1, 'T3': 2}

    for task in ['T1', 'T2', 'T3']:
        subset = df[df['task'] == task]
        x_base = task_to_num[task]

        # Add jitter to prevent overlapping
        jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
        x_jittered = x_base + jitter

        ax.scatter(
            x_jittered,
            subset[feature_name],
            c=TASK_COLORS[task],
            s=80,
            alpha=0.6,
            edgecolors='white',
            linewidths=0.5,
            label=TASK_LABELS[task],
        )

    # Set X-axis ticks to task labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([TASK_LABELS['T1'], TASK_LABELS['T2'], TASK_LABELS['T3']])


def add_scatter_labels(ax, feature_name: str) -> None:
    """
    Add labels and title to the scatter plot.
    """
    ax.set_ylabel(feature_name)
    ax.set_title(f'{feature_name} by Task (all subjects)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')


def save_scatter_graph(fig, feature_name: str) -> Path:
    """
    Save the scatter plot to file.
    """
    dmc.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    output_path = dmc.OUTPUT_DIR / f"scatter_{feature_name}.jpg"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def visualize_task_scatter(df: pd.DataFrame, feature_name: str) -> Path:
    """
    Create and save a scatter plot for the given feature.
    """
    # Create the plot
    fig, ax = create_scatter_plot(feature_name)

    # Plot the scatter points
    plot_scatter_with_jitter(ax, df, feature_name)

    # Add labels and title
    add_scatter_labels(ax, feature_name)

    # Save the graph
    return save_scatter_graph(fig, feature_name)


def get_available_features(df: pd.DataFrame) -> list[str]:
    """
    Get list of numeric feature columns (exclude metadata columns).
    """
    exclude_cols = ['subject_id', 'task']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", message="EDA signal is sampled at very low frequency"
    )

    # Read (or ask for) dataset path.
    dataset_path = dmc.read_pathfile_or_ask_for_path()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Collect all features from all subjects
    print("Collecting EDA features from all subjects...")
    features_df = collect_all_features(dataset_path)

    if features_df.empty:
        raise ValueError("No features collected. Check dataset path and files.")

    print(f"Collected {len(features_df)} records from {features_df['subject_id'].nunique()} subjects.")
    print(f"Available features: {get_available_features(features_df)}")

    # Plot key SCR features
    key_features = [
        'SCR_Peaks_N',              # Number of SCR peaks
        'SCR_Peaks_Amplitude_Mean', # Mean amplitude of peaks
    ]

    for feature in key_features:
        if feature in features_df.columns:
            visualize_task_scatter(features_df, feature)
        else:
            print(f"[WARN] Feature '{feature}' not found in data. Skipping.")
