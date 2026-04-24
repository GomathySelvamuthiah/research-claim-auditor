"""
Evaluation harness for the Research Claim Auditor distortion classifier.
Run with: python tests/run_evaluation.py
Requires ANTHROPIC_API_KEY in environment.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.claim_extractor import Claim
from src.distortion_classifier import DistortionClassifier
from src.source_retriever import RetrievalResult, SourceChunk

# Ground truth uses "accurate"; classifier uses "none" — normalise for comparison
_GT_REMAP = {"accurate": "none"}


def _load_pairs(path: str = "data/evaluation_set/labeled_pairs.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _build_claim(pair: dict, i: int) -> Claim:
    return Claim(
        claim_id=i,
        claim_text=pair["citing_claim"],
        citation_keys=["[ref]"],
        claim_type="descriptive",
        confidence=0.9,
        source_doi=None,
        hedging_language=[],
        claim_strength="assertive",
    )


def _build_retrieval_result(pair: dict) -> RetrievalResult:
    chunk = SourceChunk(
        chunk_id=0,
        doi="eval_doc",
        title="Eval Source",
        year=2020,
        text=pair["source_passage"],
        chunk_index=0,
    )
    return RetrievalResult(chunk=chunk, score=0.75)


# ---------------------------------------------------------------------------
# Metric helpers (sklearn-optional)
# ---------------------------------------------------------------------------

def _per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, dict]:
    classes = sorted(set(y_true) | set(y_pred))
    results: dict[str, dict] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        results[cls] = {"precision": precision, "recall": recall, "f1": f1,
                        "support": tp + fn}
    return results


def _cohen_kappa(y_true: list[str], y_pred: list[str]) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        pass
    # Manual computation
    n = len(y_true)
    if n == 0:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    pe = sum(
        (y_true.count(l) / n) * (y_pred.count(l) / n)
        for l in labels
    )
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0


# ---------------------------------------------------------------------------
# Formatted table
# ---------------------------------------------------------------------------

def _print_table(per_class: dict[str, dict]) -> None:
    col_w = max(len(c) for c in per_class) + 2
    header = (
        f"┌{'─' * col_w}┬{'─' * 11}┬{'─' * 8}┬{'─' * 6}┐\n"
        f"│ {'Class':<{col_w - 2}} │ Precision │ Recall │  F1  │\n"
        f"├{'─' * col_w}┼{'─' * 11}┼{'─' * 8}┼{'─' * 6}┤"
    )
    print(header)
    for cls, m in sorted(per_class.items()):
        p = f"{m['precision']:.2f}"
        r = f"{m['recall']:.2f}"
        f = f"{m['f1']:.2f}"
        print(
            f"│ {cls:<{col_w - 2}} │   {p:^7} │  {r:^5} │ {f:^4} │"
        )
    print(f"└{'─' * col_w}┴{'─' * 11}┴{'─' * 8}┴{'─' * 6}┘")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pairs = _load_pairs()
    classifier = DistortionClassifier()

    predictions: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    print(f"Running evaluation on {len(pairs)} pairs ...\n")

    for i, pair in enumerate(pairs, start=1):
        claim = _build_claim(pair, i)
        retrieval = _build_retrieval_result(pair)

        result = classifier.classify(claim, [retrieval])

        gt_raw = pair["distortion_type"]
        gt = _GT_REMAP.get(gt_raw, gt_raw)
        pred = result.distortion_type.value

        correct = gt == pred
        y_true.append(gt)
        y_pred.append(pred)

        predictions.append({
            "id": pair["id"],
            "ground_truth": gt_raw,
            "predicted": pred,
            "correct": correct,
            "explanation": result.explanation,
            "confidence": result.confidence,
        })
        status = "✓" if correct else "✗"
        print(f"  [{status}] Pair {i:02d}  gt={gt_raw:<22}  pred={pred}")

    # ── Metrics ─────────────────────────────────────────────────────────────
    total = len(pairs)
    correct_count = sum(1 for p in predictions if p["correct"])
    accuracy = correct_count / total if total else 0.0

    non_none_indices = [i for i, t in enumerate(y_true) if t != "none"]
    silent_failures = sum(
        1 for i in non_none_indices
        if y_pred[i] in ("none", "unverifiable")
    )
    sfr = silent_failures / len(non_none_indices) if non_none_indices else 0.0

    kappa = _cohen_kappa(y_true, y_pred)
    per_class = _per_class_metrics(y_true, y_pred)

    # ── Print report ────────────────────────────────────────────────────────
    sep = "═" * 40
    print(f"\n{sep}")
    print("DISTORTION CLASSIFIER — EVALUATION REPORT")
    print(sep)
    print(f"Total pairs evaluated: {total}")
    print(f"Overall accuracy:      {correct_count}/{total} ({accuracy:.0%})")
    print(f"Silent failure rate:   {sfr:.0%}  (target: ≤5%)")
    print(f"Cohen's κ:             {kappa:.2f} (target: ≥0.75)")

    print("\nPER-CLASS RESULTS")
    _print_table(per_class)

    silent_failure_pairs = [
        predictions[i] for i in non_none_indices
        if y_pred[i] in ("none", "unverifiable")
    ]
    if silent_failure_pairs:
        print("\nSILENT FAILURES (predicted accurate when it wasn't):")
        for p in silent_failure_pairs:
            claim_text = pairs[p["id"] - 1]["citing_claim"][:60]
            print(
                f"  [{p['id']}] Ground truth: {p['ground_truth']} | "
                f"Predicted: {p['predicted']} | Claim: {claim_text}..."
            )
    else:
        print("\nSILENT FAILURES: none")

    # ── Save JSON ────────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    output = {
        "total_pairs": total,
        "accuracy": round(accuracy, 4),
        "silent_failure_rate": round(sfr, 4),
        "cohen_kappa": round(kappa, 4),
        "per_class": {
            cls: {k: round(v, 4) for k, v in m.items()}
            for cls, m in per_class.items()
        },
        "predictions": predictions,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = "outputs/eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
