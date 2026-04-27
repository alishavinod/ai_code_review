"""
Prompt sensitivity test (v2) — run 10 examples through 3 prompt phrasings
to measure how much v2 labels change based on prompt wording.

Input:  data/generated/noise_labels_v2.jsonl
Output: data/generated/prompt_sensitivity_results_v2.json
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
OUTPUT_FILE = Path("data/generated/prompt_sensitivity_results_v2.json")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 64
SLEEP = 0.3
NUM_SAMPLES = 10

VALID_LABELS = {"trivial", "incorrect", "context-missing", "irrelevant"}

PROMPT_V1 = """\
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

PROMPT_V2 = """\
A developer received this AI code review suggestion but it missed the point of the actual change.
What type of noise is this suggestion?

Code before the change:
{old_code}

Code after the change:
{new_code}

AI suggestion:
{ai_suggestion}

Choose one:
- trivial: cosmetic or style issue, no functional impact
- incorrect: wrong advice, misreads the diff, or would introduce bugs
- context-missing: requires knowledge of the broader codebase not shown here
- irrelevant: unrelated to the actual code change

Answer with one word only.
"""

PROMPT_V3 = """\
Categorize this unhelpful AI code review comment into one of four noise types.

Deleted code (before):
{old_code}

Added code (after):
{new_code}

Comment: {ai_suggestion}

Noise types:
1. trivial — nitpick about style, formatting, naming with no real impact
2. incorrect — technically wrong, misleading, or would break the code based on the diff shown
3. context-missing — needs more info about the codebase to evaluate properly
4. irrelevant — off-topic or unrelated to the change being reviewed

Respond with exactly one of: trivial, incorrect, context-missing, irrelevant.
"""

PROMPTS = {
    "v1_original":        PROMPT_V1,
    "v2_developer_framing": PROMPT_V2,
    "v3_numbered_list":   PROMPT_V3,
}

def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def classify_noise(client: anthropic.Anthropic, prompt: str, old_code: str, new_code: str, ai_suggestion: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": prompt.replace("{old_code}", old_code[:1200]) \
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

    print(f"Testing {len(sample)} examples across {len(PROMPTS)} prompt variants...")

    results = []

    for i, example in enumerate(sample):
        old_code      = example.get("old_code", "").strip()
        new_code      = example.get("new_code", "").strip()
        ai_suggestion = example.get("ai_suggestion", "").strip()
        original_label = example["noise_type"]

        labels_per_prompt = {}
        for prompt_name, prompt_template in PROMPTS.items():
            try:
                label = classify_noise(client, prompt_template, old_code, new_code, ai_suggestion)
                labels_per_prompt[prompt_name] = label
            except Exception as e:
                print(f"  [{i}] ERROR on {prompt_name}: {e}")
                labels_per_prompt[prompt_name] = "error"
                time.sleep(2)
            time.sleep(SLEEP)

        unique_labels = set(labels_per_prompt.values()) - {"error"}
        all_agree = len(unique_labels) == 1
        agrees_with_original = labels_per_prompt.get("v1_original") == original_label

        result = {
            "index": example["index"],
            "original_label": original_label,
            "labels": labels_per_prompt,
            "all_prompts_agree": all_agree,
            "agrees_with_original": agrees_with_original,
            "unique_labels": list(unique_labels),
        }
        results.append(result)

        status = "✓" if all_agree else "✗"
        print(f"  [{i+1}/{len(sample)}] {status} original={original_label} | {labels_per_prompt}")

    total = len(results)
    all_agree_count = sum(1 for r in results if r["all_prompts_agree"])
    agrees_original_count = sum(1 for r in results if r["agrees_with_original"])

    prompt_agreement = {}
    for prompt_name in PROMPTS:
        if prompt_name == "v1_original":
            continue
        agree = sum(
            1 for r in results
            if r["labels"].get(prompt_name) == r["labels"].get("v1_original")
        )
        prompt_agreement[prompt_name] = round(agree / total, 3) if total > 0 else 0

    summary = {
        "total_tested": total,
        "all_three_prompts_agree": all_agree_count,
        "all_agree_rate": round(all_agree_count / total, 3) if total > 0 else 0,
        "v1_matches_original_label": agrees_original_count,
        "prompt_agreement_with_v1": prompt_agreement,
        "results": results,
    }

    OUTPUT_FILE.write_text(json.dumps(summary, indent=2))

    print(f"\nPrompt Sensitivity Summary")
    print(f"Total tested: {total}")
    print(f"All 3 prompts agree: {all_agree_count}/{total} = {all_agree_count/total:.1%}")
    print(f"V1 matches original label: {agrees_original_count}/{total} = {agrees_original_count/total:.1%}")
    print(f"\nPer-prompt agreement with V1:")
    for pname, rate in prompt_agreement.items():
        print(f"  {pname}: {rate:.1%}")
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()