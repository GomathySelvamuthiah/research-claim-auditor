"""
fine_tuning_prep.py

Converts labeled (citing_claim, source_passage, distortion_type) pairs into
OpenAI fine-tuning JSONL format and provides evaluation metric utilities.
"""

import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass

# Make project root importable when this file is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from src.distortion_classifier import CLASSIFICATION_SYSTEM_PROMPT

load_dotenv()

# Map ground-truth "accurate" label to the classifier's "none" enum value
_GT_REMAP = {"accurate": "none"}

_SEVERITY_FOR_TYPE = {
    "none": (0, "clean"),
    "accurate": (0, "clean"),
}
_DEFAULT_SEVERITY = (2, "moderate")


# ---------------------------------------------------------------------------
# Fine-tuning dataset preparation
# ---------------------------------------------------------------------------

def prepare_finetune_dataset(
    pairs_path: str = "data/evaluation_set/labeled_pairs.json",
    output_path: str = "data/finetune_dataset.jsonl",
) -> int:
    """
    Load labeled pairs, convert to OpenAI fine-tuning JSONL format, split
    80/20 into train/val files, and return the total pair count.
    """
    with open(pairs_path) as f:
        raw = json.load(f)

    # Support both flat list and {pairs: [...]} envelope
    pairs: list[dict] = raw if isinstance(raw, list) else raw.get("pairs", [])

    records: list[dict] = []
    for pair in pairs:
        citing_claim = pair.get("citing_claim", "")
        source_passage = pair.get("source_passage", "")
        raw_dt = pair.get("distortion_type", "none")
        distortion_type = _GT_REMAP.get(raw_dt, raw_dt)

        severity, severity_label = _SEVERITY_FOR_TYPE.get(
            distortion_type, _DEFAULT_SEVERITY
        )

        explanation = pair.get("distortion_explanation") or pair.get("explanation") or ""

        assistant_payload = {
            "distortion_type": distortion_type,
            "severity": severity,
            "severity_label": severity_label,
            "confidence": 0.92,
            "explanation": explanation,
            "problematic_phrase": "",
            "what_source_actually_says": source_passage[:100],
            "adequacy_score": 0.8,
        }

        user_message = (
            f"CITING CLAIM: {citing_claim}\n\n"
            f"RETRIEVED SOURCE PASSAGES:\n"
            f"[Passage 1, similarity: 0.75]\n"
            f"{source_passage}\n\n"
            f"Classify the distortion type."
        )

        records.append({
            "messages": [
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": json.dumps(assistant_payload)},
            ]
        })

    # 80/20 split
    split_idx = math.ceil(len(records) * 0.8)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    out_dir = pathlib.Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "finetune_train.jsonl"
    val_path = out_dir / "finetune_val.jsonl"

    def _write_jsonl(path: pathlib.Path, items: list[dict]) -> None:
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)

    # Also write the combined file at output_path for reference
    _write_jsonl(pathlib.Path(output_path), records)

    print(f"Train : {len(train_records)} examples → {train_path}")
    print(f"Val   : {len(val_records)} examples → {val_path}")
    print(f"Total : {len(records)} → {output_path}")

    return len(records)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: dict          # {class_label: {precision, recall, f1, support}}
    silent_failure_rate: float
    cohen_kappa: float
    n_evaluated: int


def compute_metrics(
    ground_truths: list[str],
    predictions: list[str],
) -> EvaluationMetrics:
    """Full manual metric computation — no sklearn dependency."""
    # Normalize ground truth labels
    y_true = [_GT_REMAP.get(t, t) for t in ground_truths]
    y_pred = list(predictions)

    n = len(y_true)
    if n == 0:
        return EvaluationMetrics(
            accuracy=0.0, precision_macro=0.0, recall_macro=0.0, f1_macro=0.0,
            per_class={}, silent_failure_rate=0.0, cohen_kappa=0.0, n_evaluated=0,
        )

    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    # Per-class metrics
    classes = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # Macro averages (exclude classes with zero support in ground truth)
    active = [m for cls, m in per_class.items() if m["support"] > 0]
    precision_macro = sum(m["precision"] for m in active) / len(active) if active else 0.0
    recall_macro = sum(m["recall"] for m in active) / len(active) if active else 0.0
    f1_macro = sum(m["f1"] for m in active) / len(active) if active else 0.0

    # Silent failure rate: truth != "none" but predicted "none" or "unverifiable"
    non_none_indices = [i for i, t in enumerate(y_true) if t != "none"]
    if non_none_indices:
        silent = sum(
            1 for i in non_none_indices
            if y_pred[i] in ("none", "unverifiable")
        )
        sfr = silent / len(non_none_indices)
    else:
        sfr = 0.0

    # Cohen's kappa
    po = correct / n
    pe = sum(
        (y_true.count(cls) / n) * (y_pred.count(cls) / n)
        for cls in classes
    )
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    return EvaluationMetrics(
        accuracy=round(accuracy, 4),
        precision_macro=round(precision_macro, 4),
        recall_macro=round(recall_macro, 4),
        f1_macro=round(f1_macro, 4),
        per_class=per_class,
        silent_failure_rate=round(sfr, 4),
        cohen_kappa=round(kappa, 4),
        n_evaluated=n,
    )


def print_metrics_report(
    metrics: EvaluationMetrics,
    label: str = "Evaluation",
) -> None:
    sep = "═" * 44
    print(f"\n{sep}")
    print(f"{label.upper():^44}")
    print(sep)
    print(f"  Pairs evaluated : {metrics.n_evaluated}")
    print(f"  Accuracy        : {metrics.accuracy:.0%}")
    print(f"  Precision (macro): {metrics.precision_macro:.0%}")
    print(f"  Recall (macro)  : {metrics.recall_macro:.0%}")
    print(f"  F1 (macro)      : {metrics.f1_macro:.0%}")
    print(f"  Cohen's κ       : {metrics.cohen_kappa:.2f}  (target ≥ 0.75)")
    print(f"  Silent fail rate: {metrics.silent_failure_rate:.0%}  (target ≤ 5%)")

    if metrics.per_class:
        col_w = max(len(c) for c in metrics.per_class) + 2
        print(f"\n  {'Class':<{col_w}} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>8}")
        print(f"  {'─' * col_w} {'─' * 10} {'─' * 8} {'─' * 6} {'─' * 8}")
        for cls, m in sorted(metrics.per_class.items()):
            print(
                f"  {cls:<{col_w}} {m['precision']:>10.2f} {m['recall']:>8.2f} "
                f"{m['f1']:>6.2f} {m['support']:>8}"
            )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    total = prepare_finetune_dataset()
    train_n = math.ceil(total * 0.8)
    val_n = total - train_n
    print(f"\nDataset ready: {train_n} train / {val_n} val ({total} total)")
