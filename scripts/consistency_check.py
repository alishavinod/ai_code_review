"""
Consistency check (v2) — run the v2 noise classifier twice on the same examples
to measure label stability (same model, same prompt, different API calls).

Input:  data/generated/noise_labels_v2.jsonl
Output: data/generated/consistency_results_v2.json
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
OUTPUT_FILE = Path("data/generated/consistency_results_v2.json")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 64
SLEEP = 0.3
NUM_SAMPLES = 10

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

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    data = load_jsonl(INPUT_FILE)
    noise_examples = [d for d in data if d.get("noise_type") != "valid"]

    by_label = defaultdict(list)
    for d in noise_examples:
        by_label[d["noise_type"]].append(d)

    sample = []
    per_label = max(1, NUM_SAMPLES // len(by_label))
    for label, examples in by_label.items():
        sample.extend(random.sample(examples, min(per_label, len(examples))))
    sample = sample[:NUM_SAMPLES]

    print(f"Running consistency check on {len(sample)} examples (2 runs each)...")
    print(f"Labels in sample: { [d['noise_type'] for d in sample] }")

    results = []
    consistent = 0

    for i, example in enumerate(sample):
        old_code      = example.get("old_code", "").strip()
        new_code      = example.get("new_code", "").strip()
        ai_suggestion = example.get("ai_suggestion", "").strip()
        original_label = example["noise_type"]

        try:
            run1 = classify_noise(client, old_code, new_code, ai_suggestion)
            time.sleep(SLEEP)
            run2 = classify_noise(client, old_code, new_code, ai_suggestion)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            time.sleep(2)
            continue

        is_consistent = run1 == run2
        if is_consistent:
            consistent += 1

        result = {
            "index": example["index"],
            "original_label": original_label,
            "run1": run1,
            "run2": run2,
            "consistent": is_consistent,
            "matches_original_run1": run1 == original_label,
            "matches_original_run2": run2 == original_label,
        }
        results.append(result)

        status = "✓" if is_consistent else "✗"
        print(f"  [{i+1}/{len(sample)}] {status} original={original_label} | run1={run1} | run2={run2}")

        time.sleep(SLEEP)

    total = len(results)
    consistency_rate = consistent / total if total > 0 else 0
    matches_original = sum(1 for r in results if r["matches_original_run1"]) / total if total > 0 else 0

    summary = {
        "total_tested": total,
        "consistent_both_runs": consistent,
        "consistency_rate": round(consistency_rate, 3),
        "matches_original_label_rate": round(matches_original, 3),
        "results": results,
    }

    OUTPUT_FILE.write_text(json.dumps(summary, indent=2))

    print(f"\nConsistency Summary")
    print(f"Total tested:              {total}")
    print(f"Consistent (run1==run2):   {consistent}/{total} = {consistency_rate:.1%}")
    print(f"Matches original label:    {int(matches_original*total)}/{total} = {matches_original:.1%}")
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()