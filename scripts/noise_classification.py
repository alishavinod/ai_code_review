"""
Step 3: Noise classification — classify unmatched/uncertain suggestions into noise types.
Input:  data/generated/semantic_matches.jsonl
Output: data/generated/noise_labels.jsonl

Noise types:
  - trivial         : style/formatting nitpick with no functional impact
  - incorrect       : wrong or misleading advice that could break code
  - context-missing : needs more context about the codebase to be actionable
  - irrelevant      : out of scope or unrelated to the actual change
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

INPUT_FILE = Path("data/generated/semantic_matches.jsonl")
OUTPUT_FILE = Path("data/generated/noise_labels.jsonl")
CHECKPOINT_FILE = Path("data/generated/noise_checkpoint.txt")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 64
SLEEP = 0.3

PROMPT = """\
You are evaluating an AI code review suggestion that does not match what a human reviewer said.
Classify this AI suggestion into exactly one noise category.

Code:
{code}

AI suggestion: {ai_suggestion}

Categories (read carefully before choosing):
- trivial: minor style, naming, formatting, or whitespace suggestion with no functional impact. \
This is the most common category — when in doubt between trivial and another label, pick trivial.
- incorrect: the suggestion is factually wrong, misreads the code, or would break functionality \
if followed. Only use this when the advice is clearly harmful or wrong.
- context-missing: the suggestion references specific functions, variables, APIs, or project \
conventions that are NOT visible in the provided code snippet. Do NOT use this just because \
the code is a diff or partial snippet.
- irrelevant: completely unrelated to the code change, off-topic, or about a different part \
of the codebase entirely.

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

def load_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        return int(CHECKPOINT_FILE.read_text().strip())
    return 0

def save_checkpoint(idx: int):
    CHECKPOINT_FILE.write_text(str(idx))

def classify_noise(client: anthropic.Anthropic, code: str, ai_suggestion: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": PROMPT.format(
                code=code[:2000],
                ai_suggestion=ai_suggestion[:500]
            )
        }]
    )
    raw = response.content[0].text.strip().lower()

    for label in VALID_LABELS:
        if label in raw:
            return label

    # fallback if model returns something unexpected
    return "irrelevant"

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    data = load_jsonl(INPUT_FILE)

    # only classify noise examples (unmatched + uncertain)
    noise_examples = [d for d in data if d["match_label"] in ("unmatched", "uncertain")]
    valid_examples = [d for d in data if d["match_label"] == "valid"]

    print(f"Total examples: {len(data)}")
    print(f"  Valid (skipping): {len(valid_examples)}")
    print(f"  To classify:      {len(noise_examples)}")

    start_idx = load_checkpoint()
    if start_idx > 0:
        print(f"Resuming from index {start_idx}")

    errors = []
    mode = "a" if start_idx > 0 else "w"

    label_counts = {"trivial": 0, "incorrect": 0, "context-missing": 0, "irrelevant": 0}

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:

        # write valid examples as-is (no noise type needed)
        if start_idx == 0:
            for example in valid_examples:
                record = {**example, "noise_type": "valid"}
                f.write(json.dumps(record) + "\n")

        for i, example in enumerate(noise_examples):
            if i < start_idx:
                continue

            code = example.get("old_code", "").strip()
            ai_suggestion = example.get("ai_suggestion", "").strip()

            if not code or not ai_suggestion:
                print(f"  [{i}] Skipping — missing field")
                save_checkpoint(i + 1)
                continue

            try:
                noise_type = classify_noise(client, code, ai_suggestion)
            except Exception as e:
                print(f"  [{i}] ERROR: {e}")
                errors.append({"index": example["index"], "error": str(e)})
                time.sleep(2)
                continue

            label_counts[noise_type] += 1

            record = {**example, "noise_type": noise_type}
            f.write(json.dumps(record) + "\n")
            save_checkpoint(i + 1)

            if i % 50 == 0:
                print(f"  Progress: {i+1}/{len(noise_examples)} | {label_counts}")

            time.sleep(SLEEP)

    print(f"\nDone.")
    print(f"Noise label distribution: {label_counts}")
    print(f"Output: {OUTPUT_FILE}")

    if errors:
        err_path = Path("data/generated/noise_errors.json")
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"{len(errors)} errors logged to {err_path}")

if __name__ == "__main__":
    main()