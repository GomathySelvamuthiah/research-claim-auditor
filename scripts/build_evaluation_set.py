"""
build_evaluation_set.py

Builds a real evaluation set by:
1. Sampling real paper abstracts from data/real_papers.json as source passages
2. Using GPT-4o-mini to generate realistic citing claims with controlled distortions
3. Saving to data/evaluation_set/real_eval_pairs.json

This gives ground-truth pairs grounded in real paper abstracts,
making the evaluation more credible than fully synthetic data.
"""

import json
import os
import random
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DISTORTION_TYPES = [
    "accurate",
    "certainty_inflation",
    "causal_overclaim",
    "scope_inflation",
    "cherry_picking",
]

_PROMPT_TEMPLATE = """\
Here is a real academic paper abstract:

Title: {title}
Abstract: {abstract}

Generate a realistic citing sentence that a subsequent paper might write when citing this paper.
The citing sentence must be of distortion type: {distortion_type}

Distortion definitions:
- accurate: faithfully represents what the abstract says, preserving hedging and scope
- certainty_inflation: drops hedging words ("may", "suggests") and presents finding as established fact
- causal_overclaim: abstract reports association/correlation, citing sentence asserts causation
- scope_inflation: abstract studied a specific population, citing sentence generalizes to all humans/broadly
- cherry_picking: abstract has mixed or qualified findings, citing sentence cites only the positive aspect

Return ONLY JSON:
{{"citing_claim": "...", "distortion_type": "{distortion_type}", "distortion_explanation": "one sentence explaining the specific distortion", "source_doi": "{doi}", "source_title": "{title}"}}"""


def _parse_json(raw: str) -> dict:
    """Extract JSON from model response, tolerating markdown fences."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


def build_real_eval_set(n_per_class: int = 6) -> list[dict]:
    """
    For each of 5 distortion types, sample n_per_class real abstracts and ask
    GPT-4o-mini to generate a corresponding citing claim.
    Returns a list of ground-truth pairs and saves to real_eval_pairs.json.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load and filter papers
    with open("data/real_papers.json") as f:
        raw = json.load(f)
    papers = [
        p for p in raw.get("papers", [])
        if len((p.get("abstract") or "").split()) >= 50
    ]
    print(f"  Pool: {len(papers)} papers with ≥50-word abstracts")

    # Reproducible shuffle — each class gets its own independent sample
    random.seed(42)
    shuffled = list(papers)
    random.shuffle(shuffled)

    pairs: list[dict] = []
    pair_id = 1

    for distortion_type in DISTORTION_TYPES:
        # Cycle through shuffled papers; offset each class so papers are spread out
        class_offset = DISTORTION_TYPES.index(distortion_type) * n_per_class
        class_papers = [shuffled[(class_offset + i) % len(shuffled)] for i in range(n_per_class)]

        print(f"\n  Generating {n_per_class} × '{distortion_type}' …")
        for i, paper in enumerate(class_papers, start=1):
            title   = paper.get("title", "")
            abstract = paper.get("abstract", "")
            doi     = paper.get("doi", "")

            prompt = _PROMPT_TEMPLATE.format(
                title=title,
                abstract=abstract[:800],   # cap to avoid token overrun
                distortion_type=distortion_type,
                doi=doi,
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = response.choices[0].message.content.strip()
                data = _parse_json(raw_text)
            except Exception as exc:
                print(f"    [{i}/{n_per_class}] FAILED: {exc}")
                data = {}

            if not data.get("citing_claim"):
                print(f"    [{i}/{n_per_class}] empty response — skipping")
                continue

            pairs.append({
                "id": pair_id,
                "citing_claim": data.get("citing_claim", ""),
                "source_passage": abstract,
                "distortion_type": distortion_type,
                "distortion_explanation": data.get("distortion_explanation", ""),
                "source_doi": doi,
                "source_title": title,
                "is_real_paper": True,
            })
            print(f"    [{i}/{n_per_class}] OK — {title[:55]}…")
            pair_id += 1

    # Save with metadata envelope
    os.makedirs("data/evaluation_set", exist_ok=True)
    output = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "total": len(pairs),
        "sources": "OpenAlex real paper abstracts",
        "pairs": pairs,
    }
    with open("data/evaluation_set/real_eval_pairs.json", "w") as f:
        json.dump(output, f, indent=2)

    return pairs


def merge_with_existing() -> None:
    """
    Merge real_eval_pairs.json into labeled_pairs.json, deduplicating by
    citing_claim and back-filling is_real_paper=false on legacy entries.
    """
    # Load real pairs (inside envelope)
    with open("data/evaluation_set/real_eval_pairs.json") as f:
        real_data = json.load(f)
    real_pairs: list[dict] = real_data.get("pairs", [])

    # Load existing labeled pairs (flat list)
    labeled_path = "data/evaluation_set/labeled_pairs.json"
    with open(labeled_path) as f:
        existing: list[dict] = json.load(f)

    # Back-fill is_real_paper on legacy entries
    for p in existing:
        p.setdefault("is_real_paper", False)

    # Deduplicate: build seen-set from existing, then append new real pairs
    seen: set[str] = {p["citing_claim"].strip().lower() for p in existing}
    next_id = max((p.get("id", 0) for p in existing), default=0) + 1

    added = 0
    for pair in real_pairs:
        key = pair.get("citing_claim", "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged_entry = {
            "id": next_id,
            "citing_claim": pair["citing_claim"],
            "source_passage": pair["source_passage"],
            "distortion_type": pair["distortion_type"],
            "explanation": pair.get("distortion_explanation", ""),
            "source_doi": pair.get("source_doi", ""),
            "source_title": pair.get("source_title", ""),
            "is_real_paper": True,
        }
        existing.append(merged_entry)
        next_id += 1
        added += 1

    with open(labeled_path, "w") as f:
        json.dump(existing, f, indent=2)

    n_real = sum(1 for p in existing if p.get("is_real_paper"))
    n_synthetic = len(existing) - n_real
    print(
        f"Merged eval set: {n_real} real + {n_synthetic} synthetic = {len(existing)} total pairs"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Building Real Evaluation Set ===\n")
    pairs = build_real_eval_set(n_per_class=6)   # 30 real pairs total
    print(f"\n✅ {len(pairs)} real eval pairs saved")
    merge_with_existing()

    if pairs:
        print("\n=== Sample Real Pair ===")
        print(f"Source: {pairs[0]['source_title'][:60]}...")
        print(f"Citing claim: {pairs[0]['citing_claim']}")
        print(f"Distortion: {pairs[0]['distortion_type']}")
        print(f"Explanation: {pairs[0]['distortion_explanation']}")
