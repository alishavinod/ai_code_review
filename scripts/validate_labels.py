"""
Step 3b (v2): Second LLM validator — re-evaluates a random subset of v2 noise labels
using Claude Sonnet as an independent annotator to measure inter-annotator agreement.

Input:  data/generated/noise_labels_v2.jsonl
Output: data/generated/validation_results_v2.jsonl
        data/generated/validation_summary_v2.json
"""

import os
import json
import time
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import anthropic

load_dotenv()

INPUT_FILE  = Path("data/generated/noise_labels_v2.jsonl")
OUTPUT_FILE = Path("data/generated/validation_results_v2.jsonl")
SUMMARY_FILE = Path("data/generated/validation_summary_v2.json")

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 64
SLEEP = 0.5

SAMPLES_PER_LABEL = 10

PROMPT = """\
You are evaluating an AI code review suggestion that does not match what a human reviewer said.
Classify this AI suggestion into exactly one noise category.

BEFORE the change (deleted lines):
{old_code}

AFTER the change (added lines):
{new_code}

AI suggestion: {ai_suggestion}

Categories (read carefully before choosing):
- trivial: minor style, naming, formatting, or whitespace suggestion with no functional impact.
- incorrect: the suggestion is factually wrong, misreads the code, or would break functionality
  if followed. Only use this when the advice is clearly harmful or wrong based on the visible diff.
- context-missing: the suggestion may be valid but requires knowledge of the broader codebase,
  APIs, or project conventions not visible in the provided diff.
- irrelevant: completely unrelated to the code change shown.

Reply with exactly one word: trivial, incorrect, context-missing, or irrelevant.
"""

VALID_LABELS = {"trivial", "incorrect", "context-missing", "irrelevant"}

def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def classify_noise(client: anthropic.Anthropic, old_code: str, new_code: str, ai_suggestion: str) -> str:
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{
                    "role": "user",
                    "content": PROMPT.replace("{old_code}", old_code[:1200]) \
                                     .replace("{new_code}", new_code[:1200]) \
                                     .replace("{ai_suggestion}", ai_suggestion[:500])
                }]
            )
            raw = response.content[0].text.strip().lower()
            for label in VALID_LABELS:
                if label in raw:
                    return label
            return "irrelevant"
        except Exception as e:
            if attempt == 0:
                print(f"    Retry after error: {e}")
                time.sleep(3)
            else:
                raise

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    data = load_jsonl(INPUT_FILE)

    noise_examples = [d for d in data if d.get("noise_type") != "valid"]

    by_label = defaultdict(list)
    for d in noise_examples:
        by_label[d["noise_type"]].append(d)

    sample = []
    for label, examples in by_label.items():
        n = min(SAMPLES_PER_LABEL, len(examples))
        sample.extend(random.sample(examples, n))

    random.shuffle(sample)
    print(f"Validating {len(sample)} examples ({SAMPLES_PER_LABEL} per label max)")
    print(f"Label distribution: { {k: len(v) for k, v in by_label.items()} }")

    results = []
    agree = 0
    disagree = 0
    confusion = defaultdict(lambda: defaultdict(int))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, example in enumerate(sample):
            old_code     = example.get("old_code", "").strip()
            new_code     = example.get("new_code", "").strip()
            ai_suggestion = example.get("ai_suggestion", "").strip()
            primary_label = example["noise_type"]

            try:
                validator_label = classify_noise(client, old_code, new_code, ai_suggestion)
            except Exception as e:
                print(f"  [{i}] ERROR: {e}")
                time.sleep(2)
                continue

            match = primary_label == validator_label
            if match:
                agree += 1
            else:
                disagree += 1

            confusion[primary_label][validator_label] += 1

            record = {
                "index": example["index"],
                "ai_suggestion": ai_suggestion[:200],
                "primary_label": primary_label,
                "validator_label": validator_label,
                "agreement": match,
            }
            results.append(record)
            f.write(json.dumps(record) + "\n")
            f.flush()

            if i % 5 == 0:
                total_so_far = agree + disagree
                acc = agree / total_so_far if total_so_far > 0 else 0
                print(f"  Progress: {i+1}/{len(sample)} | Agreement so far: {acc:.1%}")

            time.sleep(SLEEP)

    total = agree + disagree
    agreement_rate = agree / total if total > 0 else 0

    per_label_agreement = {}
    for label in VALID_LABELS:
        label_results = [r for r in results if r["primary_label"] == label]
        if label_results:
            label_agree = sum(1 for r in label_results if r["agreement"])
            per_label_agreement[label] = {
                "agreement": round(label_agree / len(label_results), 3),
                "total": len(label_results),
            }

    summary = {
        "total_validated": total,
        "overall_agreement": round(agreement_rate, 3),
        "agreed": agree,
        "disagreed": disagree,
        "per_label_agreement": per_label_agreement,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    print(f"\nValidation Summary")
    print(f"Total validated:    {total}")
    print(f"Overall agreement:  {agreement_rate:.1%}")
    print(f"\nPer-label agreement:")
    for label, stats in per_label_agreement.items():
        print(f"  {label}: {stats['agreement']:.1%} ({stats['total']} examples)")
    print(f"\nConfusion matrix:")
    for primary, validator_counts in confusion.items():
        print(f"  {primary}: {dict(validator_counts)}")
    print(f"\nSummary saved to {SUMMARY_FILE}")
    print(f"\nNext: run compute_kappa_v2.py to compute Cohen's Kappa from these results.")

if __name__ == "__main__":
    main()