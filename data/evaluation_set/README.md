# Evaluation Set

## Files
- `real_eval_pairs.json` — 30 pairs grounded in real paper abstracts (OpenAlex)
- `labeled_pairs.json` — full merged set (real + synthetic)

## Composition
| Type | Count | Source |
|------|-------|--------|
| Real paper abstracts | 30 | OpenAlex via fetch_real_data.py |
| Synthetic (GPT-4o-mini) | 50 | synthetic_data_generator.py |
| Original hand-labeled | 10 | Manual annotation |
| Total | 90 | — |

## Distortion Type Distribution
Balanced: 6 real pairs per class × 5 classes = 30 real pairs

## Ground Truth Reliability
Real pairs: source passage is the actual paper abstract from OpenAlex.
The distortion is introduced at the citing_claim level by controlled generation.
This makes the ground truth more credible than fully synthetic pairs.
