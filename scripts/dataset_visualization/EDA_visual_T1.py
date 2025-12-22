import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd

# TODO: (per-subject combined plot) is DONE in `visualize_subject_combined`.
# TODO: per-task scatter plots across subjects using `nk.eda_analyze` features.

SAMPLING_RATE_HZ = 4
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH_FILE = SCRIPT_DIR / "UBFC_path.txt"
OUTPUT_DIR = SCRIPT_DIR / "eda_results"


def ensure_dataset_path_file_exists() -> None:
    if not DATASET_PATH_FILE.exists():
        raise FileNotFoundError(f"The file {DATASET_PATH_FILE} does not exist")


def prompt_dataset_path() -> Path:
    dataset_path = input(
        "Enter the path to the dataset (to Data folder): ").strip()
    return Path(dataset_path)


def save_dataset_path(dataset_path: Path) -> None:
    DATASET_PATH_FILE.write_text(str(dataset_path), encoding="utf-8")


def read_dataset_path() -> Path:
    return Path(DATASET_PATH_FILE.read_text(encoding="utf-8").splitlines()[0].strip())


def list_subject_paths(dataset_path: Path) -> list[Path]:
    subject_paths: list[Path] = []
    for entry in dataset_path.iterdir():
        if entry.is_dir() and entry.name.startswith("s"):
            subject_paths.append(entry)
    return sorted(subject_paths, key=lambda p: p.name)


def list_eda_paths(subject_path: Path) -> list[Path]:
    eda_paths: list[Path] = []
    for entry in subject_path.iterdir():
        if entry.is_file() and entry.name.startswith("eda_"):
            eda_paths.append(entry)
    return sorted(eda_paths, key=lambda p: p.name)


def read_eda_data(eda_path: Path) -> list[float]:
    eda_data: list[float] = []
    with eda_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eda_data.append(float(line))
    return eda_data


def process_eda_data(eda_data: list[float]) -> tuple[pd.DataFrame, dict]:
    eda_processed_signal, info = nk.eda_process(
        eda_data,
        sampling_rate=SAMPLING_RATE_HZ,
    )
    return eda_processed_signal, info


def has_expected_eda_files(eda_paths: list[Path], subject_path: Path) -> bool:
    """
    UBFC-Phys EDA should have exactly 3 files per subject: T1/T2/T3.
    We skip subjects that don't match to avoid breaking batch processing.
    """
    if len(eda_paths) != 3:
        print(
            f"[WARN] {subject_path}: expected 3 EDA files (T1/T2/T3), got {len(eda_paths)}. Skipping."
        )
        return False
    return True


def calculate_graph_task_boundary(task_length_list: list[int]) -> tuple[float, float]:
    """
    Input: length list of the signals.
    Output: task boundaries in seconds on the concatenated timeline.
    """
    task1_task2_div_sec = task_length_list[0] / SAMPLING_RATE_HZ
    task2_task3_div_sec = (
        task_length_list[0] + task_length_list[1]) / SAMPLING_RATE_HZ
    return task1_task2_div_sec, task2_task3_div_sec


def create_graph_plot(all_signals: pd.DataFrame):
    """
    Create the graph plot of the signals.
    """
    time_axis = all_signals.index / SAMPLING_RATE_HZ

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
                linewidth=0.5, alpha=0.5)  # 零基线
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.suptitle(f"EDA {subject_id}", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"eda_analysis_{subject_id}.jpg"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def visualize_subject_combined(subject_path: Path) -> Path | None:
    """
    Make a plot for each subject with all the tasks combined (T1+T2+T3).
    """
    eda_paths = list_eda_paths(subject_path)
    if not has_expected_eda_files(eda_paths, subject_path):
        return None

    signal_list: list[pd.DataFrame] = []
    length_list: list[int] = []

    for eda_path in eda_paths:
        eda_data = read_eda_data(eda_path)
        processed_eda_signal, info = process_eda_data(eda_data)
        signal_list.append(processed_eda_signal)
        length_list.append(len(processed_eda_signal))

    # combine the signals into a single dataframe
    all_signals = pd.concat(signal_list, axis=0, ignore_index=True)

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


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", message="EDA signal is sampled at very low frequency"
    )

    # Read (or ask for) dataset path.
    try:
        ensure_dataset_path_file_exists()
    except FileNotFoundError:
        dataset_path = prompt_dataset_path()
        save_dataset_path(dataset_path)

    dataset_path = read_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    subject_paths = list_subject_paths(dataset_path)
    if not subject_paths:
        raise ValueError(f"No subject directories found under: {dataset_path}")

    for subject_path in subject_paths:
        visualize_subject_combined(subject_path)
