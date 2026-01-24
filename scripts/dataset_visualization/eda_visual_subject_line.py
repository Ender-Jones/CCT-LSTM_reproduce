import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
import numpy as np

import data_mining_common as dmc

# TODO: (per-subject combined plot) is DONE in `visualize_subject_combined`.
# TODO: per-task scatter plots across subjects using `nk.eda_analyze` features.


def calculate_graph_task_boundary(task_length_list: list[int]) -> tuple[float, float]:
    """
    Input: length list of the signals.
    Output: task boundaries in seconds on the concatenated timeline.
    """
    task1_task2_div_sec = task_length_list[0] / dmc.EDA_SAMPLING_RATE_HZ
    task2_task3_div_sec = (
        task_length_list[0] + task_length_list[1]) / dmc.EDA_SAMPLING_RATE_HZ
    return task1_task2_div_sec, task2_task3_div_sec


def create_graph_plot(all_signals: pd.DataFrame):
    """
    Create the graph plot of the signals.
    """
    time_axis = all_signals.index / dmc.EDA_SAMPLING_RATE_HZ

    # create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

    return time_axis, fig, (ax1, ax2, ax3)


def plot_row_1(time_axis, ax1, all_signals: pd.DataFrame) -> None:
    """
    Row 1: Raw and Cleaned Signal
    """
    ax1.plot(
        time_axis,
        all_signals["EDA_Raw"],
        color="gray",
        alpha=0.5,
        label="Raw",
        linewidth=0.8,
    )
    ax1.plot(
        time_axis,
        all_signals["EDA_Clean"],
        color="purple",
        label="Cleaned",
        linewidth=1.5,
    )
    ax1.set_title("Raw and Cleaned EDA Signal", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Conductance (µS)")
    ax1.legend(loc='upper right')


def plot_row_2(time_axis, ax2, all_signals: pd.DataFrame) -> None:
    """
    Row 2: SCR (Phasic), skin conductance response
    """
    ax2.plot(
        time_axis,
        all_signals["EDA_Phasic"],
        color="#D81B60",
        label="SCR (Phasic)",
        linewidth=1,
    )
    ax2.set_title("Skin Conductance Response (Phasic)",
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel("Amplitude (µS)")
    ax2.axhline(y=0, color='gray', linestyle='-',
                linewidth=0.5, alpha=0.5)
    ax2.legend(loc='upper right')


def plot_row_3(time_axis, ax3, all_signals: pd.DataFrame) -> None:
    """
    Row 3: SCL (Tonic), baseline skin conductance level
    """
    ax3.plot(
        time_axis,
        all_signals["EDA_Tonic"],
        color="#1E88E5",
        label="SCL (Tonic)",
        linewidth=2,
    )
    ax3.set_title("Skin Conductance Level (Tonic)",
                  fontsize=14, fontweight='bold')
    ax3.set_ylabel("Conductance (µS)")
    ax3.set_xlabel("Time (seconds)")
    ax3.legend(loc='upper right')


def add_boundary_lines(row1, row2, row3, task1_task2_div_sec: float, task2_task3_div_sec: float) -> None:
    """
    Add the boundary lines to the graph.
    """
    for row in [row1, row2, row3]:
        row.axvline(x=task1_task2_div_sec, color='black',
                    linestyle='--', linewidth=2, alpha=0.7)
        row.axvline(x=task2_task3_div_sec, color='black',
                    linestyle='--', linewidth=2, alpha=0.7)


def save_graph(fig, subject_id: str) -> Path:
    dmc.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.suptitle(f"EDA {subject_id}", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path = dmc.OUTPUT_DIR / f"eda_analysis_{subject_id}.jpg"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def process_subject_data(subject_path: Path) -> tuple[pd.DataFrame, list[int]] | None:
    """
    Process EDA data for a single subject.
    Returns the combined signals DataFrame and the list of task lengths.
    """
    eda_paths = dmc.list_eda_paths(subject_path)
    if not dmc.has_expected_eda_files(eda_paths, subject_path):
        return None

    signal_list: list[pd.DataFrame] = []
    length_list: list[int] = []

    for eda_path in eda_paths:
        eda_data = dmc.read_eda_data(eda_path)
        processed_eda_signal, info = nk.eda_process(
            eda_data, sampling_rate=dmc.EDA_SAMPLING_RATE_HZ)
        signal_list.append(processed_eda_signal)
        length_list.append(len(processed_eda_signal))

    # combine the signals into a single dataframe
    all_signals = pd.concat(signal_list, axis=0, ignore_index=True)
    return all_signals, length_list


def visualize_subject_combined(subject_path: Path) -> Path | None:
    """
    Make a plot for each subject with all the tasks combined (T1+T2+T3).
    """
    result = process_subject_data(subject_path)
    if result is None:
        return None
    all_signals, length_list = result

    # calculate boundary of the signals
    task1_task2_div_sec, task2_task3_div_sec = calculate_graph_task_boundary(
        length_list)

    # create the graph plot
    time_axis, fig, (ax1, ax2, ax3) = create_graph_plot(all_signals)

    # plot the rows
    plot_row_1(time_axis, ax1, all_signals)
    plot_row_2(time_axis, ax2, all_signals)
    plot_row_3(time_axis, ax3, all_signals)

    # add the boundary lines
    add_boundary_lines(ax1, ax2, ax3, task1_task2_div_sec, task2_task3_div_sec)

    # add the title to the graph
    subject_id = subject_path.name

    # save the graph
    return save_graph(fig, subject_id)


def visualize_group_combined(subject_paths: list[Path], group_name: str) -> Path | None:
    """
    Make a combined plot for a group of subjects.
    """
    group_data = []
    for subject_path in subject_paths:
        result = process_subject_data(subject_path)
        if result:
            all_signals, length_list = result
            group_data.append((subject_path.name, all_signals))

    if not group_data:
        return None

    # create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

    # Generate colors
    num_subjects = len(group_data)
    if num_subjects <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_subjects))
    else:
        colors = plt.cm.jet(np.linspace(0, 1, num_subjects))

    for i, (subject_id, all_signals) in enumerate(group_data):
        time_axis = all_signals.index / dmc.EDA_SAMPLING_RATE_HZ
        color = colors[i]

        # Row 1: Cleaned
        ax1.plot(
            time_axis,
            all_signals["EDA_Clean"],
            color=color,
            label=subject_id,
            linewidth=1.2,
            alpha=0.7,
        )

        # Row 2: SCR (Phasic)
        ax2.plot(
            time_axis,
            all_signals["EDA_Phasic"],
            color=color,
            label=subject_id,
            linewidth=1,
            alpha=0.7,
        )

        # Row 3: SCL (Tonic)
        ax3.plot(
            time_axis,
            all_signals["EDA_Tonic"],
            color=color,
            label=subject_id,
            linewidth=1.5,
            alpha=0.7,
        )

    # Set titles and labels
    ax1.set_title(f"Cleaned EDA Signal - {group_name}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Conductance (µS)")

    ax2.set_title(f"Skin Conductance Response (Phasic) - {group_name}",
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel("Amplitude (µS)")
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax3.set_title(f"Skin Conductance Level (Tonic) - {group_name}",
                  fontsize=14, fontweight='bold')
    ax3.set_ylabel("Conductance (µS)")
    ax3.set_xlabel("Time (seconds)")

    # Legend (only if reasonable number)
    if num_subjects <= 20:
        ax1.legend(loc='upper right', ncol=2, fontsize='small')

    return save_graph(fig, group_name)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", message="EDA signal is sampled at very low frequency"
    )

    # Read (or ask for) dataset path.
    dataset_path = dmc.read_pathfile_or_ask_for_path()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    subject_paths = dmc.list_subject_paths(dataset_path)
    if not subject_paths:
        raise ValueError(f"No subject directories found under: {dataset_path}")

    # 1. Original per-subject visualization
    print("Generating per-subject plots...")
    for subject_path in subject_paths:
        visualize_subject_combined(subject_path)

    # 2. Group by 8
    print("Generating 8-subject group plots...")
    chunk_size_8 = 8
    for i in range(0, len(subject_paths), chunk_size_8):
        group = subject_paths[i:i + chunk_size_8]
        group_name = f"Group_8_Batch_{i // chunk_size_8 + 1}"
        visualize_group_combined(group, group_name)

    # 3. Group by 14
    print("Generating 14-subject group plots...")
    chunk_size_14 = 14
    for i in range(0, len(subject_paths), chunk_size_14):
        group = subject_paths[i:i + chunk_size_14]
        group_name = f"Group_14_Batch_{i // chunk_size_14 + 1}"
        visualize_group_combined(group, group_name)

    # 4. All subjects
    print("Generating all-subjects plot...")
    visualize_group_combined(subject_paths, "All_Subjects")
