import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_classification_report(path: Path) -> Tuple[float, float, float, float]:
    """
    Parse sklearn's text classification_report to extract:
    - accuracy
    - macro avg: precision, recall, f1-score

    Returns: (accuracy, macro_precision, macro_recall, macro_f1)
    """
    text = path.read_text(encoding='utf-8', errors='ignore')

    # accuracy line like: "accuracy                         0.6250        24"
    acc_match = re.search(r"accuracy\s+([0-9]*\.?[0-9]+)", text)
    accuracy = float(acc_match.group(1)) if acc_match else float('nan')

    # macro avg line like: "macro avg     0.6456    0.6250    0.5984        24"
    macro_match = re.search(r"macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)", text)
    if macro_match:
        macro_precision = float(macro_match.group(1))
        macro_recall = float(macro_match.group(2))
        macro_f1 = float(macro_match.group(3))
    else:
        macro_precision = macro_recall = macro_f1 = float('nan')

    return accuracy, macro_precision, macro_recall, macro_f1


def collect_experiment(report_root: Path, exp_name: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    exp_dir = report_root / exp_name
    if not exp_dir.exists():
        return rows

    for fold_dir in sorted(exp_dir.glob('fold_*'), key=lambda p: int(p.name.split('_')[-1])):
        report_path = fold_dir / 'classification_report.txt'
        if not report_path.exists():
            continue
        acc, mp, mr, mf1 = parse_classification_report(report_path)
        rows.append({
            'experiment': exp_name,
            'fold': fold_dir.name.split('_')[-1],
            'accuracy': f"{acc:.4f}",
            'macro_precision': f"{mp:.4f}",
            'macro_recall': f"{mr:.4f}",
            'macro_f1': f"{mf1:.4f}",
        })
    return rows


def add_summary_rows(rows: List[Dict[str, str]], exp_name: str) -> None:
    vals = [float(r['macro_f1']) for r in rows if r['experiment'] == exp_name and r['macro_f1'] != 'nan']
    if not vals:
        return
    # mean row
    import statistics as stats
    mean_f1 = stats.mean(vals)
    # compute mean for accuracy/precision/recall as well
    mean_acc = stats.mean([float(r['accuracy']) for r in rows if r['experiment'] == exp_name and r['accuracy'] != 'nan'])
    mean_mp = stats.mean([float(r['macro_precision']) for r in rows if r['experiment'] == exp_name and r['macro_precision'] != 'nan'])
    mean_mr = stats.mean([float(r['macro_recall']) for r in rows if r['experiment'] == exp_name and r['macro_recall'] != 'nan'])

    rows.append({
        'experiment': exp_name,
        'fold': 'mean',
        'accuracy': f"{mean_acc:.4f}",
        'macro_precision': f"{mean_mp:.4f}",
        'macro_recall': f"{mean_mr:.4f}",
        'macro_f1': f"{mean_f1:.4f}",
    })

    # mark best fold by macro_f1 (optional; add an extra row)
    best = max((r for r in rows if r['experiment'] == exp_name and r['fold'].isdigit()), key=lambda r: float(r['macro_f1']), default=None)
    if best:
        rows.append({
            'experiment': exp_name,
            'fold': f"best_fold_{best['fold']}",
            'accuracy': best['accuracy'],
            'macro_precision': best['macro_precision'],
            'macro_recall': best['macro_recall'],
            'macro_f1': best['macro_f1'],
        })


def main():
    repo_root = Path(__file__).parent
    reports_root = repo_root / 'reports'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = reports_root / f'summary_{ts}.csv'

    all_rows: List[Dict[str, str]] = []
    header = ['experiment', 'fold', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1']

    # collect tasks and levels
    for exp in ['tasks', 'levels']:
        rows = collect_experiment(reports_root, exp)
        all_rows.extend(rows)
        add_summary_rows(all_rows, exp)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Summary written to: {out_csv}")


if __name__ == '__main__':
    main()


