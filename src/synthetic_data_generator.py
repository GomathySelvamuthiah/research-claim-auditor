"""
synthetic_data_generator.py

Generates synthetic labeled (citing_claim, source_passage, distortion_type) pairs
for training and testing the distortion classifier using the Anthropic API.

Ethical Considerations:
- All data is fully synthetic — no real patient data, no real paper text
- Generated claims do not reference real researchers by name
- Distortion examples are educational, not designed to defame any real work
- Dataset is intended for classifier training/testing only
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DISTORTION_TYPES = [
    "accurate",
    "certainty_inflation",
    "causal_overclaim",
    "scope_inflation",
    "cherry_picking",
]

DEFAULT_DOMAINS = [
    "oncology",
    "cardiology",
    "neuroscience",
    "psychiatry",
    "epidemiology",
    "pharmacology",
]

_PROMPT_TEMPLATE = """\
Generate a realistic academic citation distortion example of type: {distortion_type}
Domain: {domain}

Return ONLY JSON with this structure:
{{
  "citing_claim": "the sentence in the citing paper (with subtle distortion if applicable)",
  "source_passage": "the actual text from the cited source",
  "distortion_type": "{distortion_type}",
  "domain": "{domain}",
  "distortion_explanation": "one sentence explaining the distortion"
}}

Rules:
- Make it realistic and subtle, not obvious
- For 'accurate': citing_claim should faithfully represent source_passage
- For 'certainty_inflation': source uses hedging ("may", "suggests"), claim drops it
- For 'causal_overclaim': source reports correlation, claim asserts causation
- For 'scope_inflation': source studied a narrow population, claim generalizes broadly
- For 'cherry_picking': source had mixed results, claim cites only positive arm"""


class SyntheticDataGenerator:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate_pair(self, distortion_type: str, domain: str) -> dict:
        """Call Claude to produce one labeled pair. Returns a raw dict."""
        prompt = _PROMPT_TEMPLATE.format(
            distortion_type=distortion_type, domain=domain
        )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_json(raw, distortion_type, domain)

    @staticmethod
    def _parse_json(raw: str, distortion_type: str, domain: str) -> dict:
        """Extract and validate the JSON payload from the model response."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to pull the first {...} block
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        # Enforce required keys with safe defaults
        data.setdefault("citing_claim", "")
        data.setdefault("source_passage", "")
        data.setdefault("distortion_type", distortion_type)
        data.setdefault("domain", domain)
        data.setdefault("distortion_explanation", "")
        return data

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_per_class: int = 10,
        domains: list[str] | None = None,
    ) -> list[dict]:
        """
        Generate n_per_class examples for each of the 5 distortion types.
        Rotates through domains for variety.
        """
        if domains is None:
            domains = DEFAULT_DOMAINS

        dataset: list[dict] = []
        pair_id = 1

        for distortion_type in DISTORTION_TYPES:
            log.info("  Generating %d × '%s' ...", n_per_class, distortion_type)
            for i in range(n_per_class):
                domain = domains[i % len(domains)]
                try:
                    pair = self.generate_pair(distortion_type, domain)
                    pair["id"] = pair_id
                    dataset.append(pair)
                    log.info(
                        "    [%d/%d] %s / %s — OK",
                        i + 1,
                        n_per_class,
                        distortion_type,
                        domain,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "    [%d/%d] %s / %s — FAILED: %s",
                        i + 1,
                        n_per_class,
                        distortion_type,
                        domain,
                        exc,
                    )
                pair_id += 1

        return dataset

    # ------------------------------------------------------------------
    # Diversity metrics
    # ------------------------------------------------------------------

    def validate_diversity(self, dataset: list[dict]) -> dict:
        """Compute class/domain distribution and average text lengths."""
        class_dist: dict[str, int] = {}
        domain_dist: dict[str, int] = {}
        claim_lengths: list[int] = []
        passage_lengths: list[int] = []

        for pair in dataset:
            dt = pair.get("distortion_type", "unknown")
            dm = pair.get("domain", "unknown")
            class_dist[dt] = class_dist.get(dt, 0) + 1
            domain_dist[dm] = domain_dist.get(dm, 0) + 1
            claim_lengths.append(len(pair.get("citing_claim", "").split()))
            passage_lengths.append(len(pair.get("source_passage", "").split()))

        return {
            "total_pairs": len(dataset),
            "class_distribution": class_dist,
            "domain_distribution": domain_dist,
            "avg_citing_claim_words": (
                round(sum(claim_lengths) / len(claim_lengths), 1)
                if claim_lengths
                else 0
            ),
            "avg_source_passage_words": (
                round(sum(passage_lengths) / len(passage_lengths), 1)
                if passage_lengths
                else 0
            ),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_dataset(
        self,
        dataset: list[dict],
        path: str = "data/synthetic_pairs.json",
    ) -> None:
        """Save with a metadata header."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        stats = self.validate_diversity(dataset)
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_pairs": len(dataset),
            "diversity_stats": stats,
            "pairs": dataset,
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        log.info("Saved %d pairs → %s", len(dataset), path)


# ---------------------------------------------------------------------------
# Evaluation-set merge helper
# ---------------------------------------------------------------------------

def _merge_into_eval_set(
    new_pairs: list[dict],
    eval_path: str = "data/evaluation_set/labeled_pairs.json",
) -> None:
    """
    Merge new_pairs into the existing labeled_pairs.json.
    Keeps originals, appends new ones, deduplicates by citing_claim.
    """
    # Load existing
    existing: list[dict] = []
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            existing = json.load(f)

    seen: set[str] = {p["citing_claim"].strip().lower() for p in existing}

    # Determine next id
    next_id = max((p.get("id", 0) for p in existing), default=0) + 1

    added = 0
    for pair in new_pairs:
        key = pair.get("citing_claim", "").strip().lower()
        if key in seen or not key:
            continue
        seen.add(key)
        entry = {
            "id": next_id,
            "citing_claim": pair.get("citing_claim", ""),
            "source_passage": pair.get("source_passage", ""),
            "distortion_type": pair.get("distortion_type", ""),
            "explanation": pair.get("distortion_explanation", ""),
        }
        # Preserve optional fields present in synthetic pairs
        if pair.get("domain"):
            entry["domain"] = pair["domain"]
        existing.append(entry)
        next_id += 1
        added += 1

    with open(eval_path, "w") as f:
        json.dump(existing, f, indent=2)
    log.info("Merged %d new pairs into %s (total: %d)", added, eval_path, len(existing))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Synthetic Data Generator ===\n")

    generator = SyntheticDataGenerator()

    log.info("Generating dataset (10 pairs × 5 classes = 50 total)...\n")
    dataset = generator.generate_dataset(n_per_class=10)

    log.info("\nSaving to data/synthetic_pairs.json ...")
    generator.save_dataset(dataset, path="data/synthetic_pairs.json")

    stats = generator.validate_diversity(dataset)

    print("\n" + "=" * 50)
    print("DIVERSITY STATS")
    print("=" * 50)
    print(f"Total pairs generated : {stats['total_pairs']}")
    print(f"Avg citing_claim len  : {stats['avg_citing_claim_words']} words")
    print(f"Avg source_passage len: {stats['avg_source_passage_words']} words")

    print("\nClass distribution:")
    for dt, count in sorted(stats["class_distribution"].items()):
        bar = "#" * count
        print(f"  {dt:<25} {count:>3}  {bar}")

    print("\nDomain distribution:")
    for dm, count in sorted(stats["domain_distribution"].items()):
        bar = "#" * count
        print(f"  {dm:<20} {count:>3}  {bar}")

    print("\nMerging into data/evaluation_set/labeled_pairs.json ...")
    _merge_into_eval_set(dataset, eval_path="data/evaluation_set/labeled_pairs.json")

    print("\nDone.")
