"""
Compute Cohen's Kappa (v2) from validation_results_v2.jsonl.
Run this after validate_labels_v2.py has completed.
"""

import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

RESULTS_FILE = Path("data/generated/validation_results_v2.jsonl")
SUMMARY_FILE = Path("data/generated/validation_summary_v2.json")

def interpret_kappa(kappa: float) -> str:
    if kappa < 0.2:
        return "Slight agreement"
    elif kappa < 0.4:
        return "Fair agreement"
    elif kappa < 0.6:
        return "Moderate agreement"
    elif kappa < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def main():
    results = []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    primary   = [r["primary_label"]   for r in results]
    validator = [r["validator_label"] for r in results]

    kappa = cohen_kappa_score(primary, validator)
    interpretation = interpret_kappa(kappa)

    print(f"Cohen's Kappa")
    print(f"Total examples: {len(results)}")
    print(f"Cohen's Kappa:  {kappa:.3f}")
    print(f"Interpretation: {interpretation}")
    print()
    print("Reference scale:")
    print("  < 0.20 : Slight")
    print("  0.20-0.40 : Fair")
    print("  0.40-0.60 : Moderate")
    print("  0.60-0.80 : Substantial")
    print("  > 0.80 : Almost perfect")

    labels = sorted(set(primary))
    print(f"\nPer-label agreement:")
    for label in labels:
        indices = [i for i, l in enumerate(primary) if l == label]
        label_primary   = [primary[i] for i in indices]
        label_validator = [validator[i] for i in indices]
        matches = sum(1 for p, v in zip(label_primary, label_validator) if p == v)
        print(f"  {label}: {matches}/{len(indices)} = {matches/len(indices):.1%}")

    if SUMMARY_FILE.exists():
        summary = json.loads(SUMMARY_FILE.read_text())
        summary["cohen_kappa"] = round(kappa, 3)
        summary["kappa_interpretation"] = interpretation
        SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
        print(f"\nKappa added to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()