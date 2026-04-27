"""
Step 2: Semantic matching — compare AI suggestions to human reviews.
Input:  data/generated/ai_suggestions.jsonl
Output: data/generated/semantic_matches.jsonl

Labels:
  - valid     : AI suggestion addresses the same issue as human review
  - unmatched : AI suggestion does not address the same issue (goes to noise classification)
  - uncertain : borderline case (also goes to noise classification)
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

INPUT_FILE = Path("data/generated/ai_suggestions_v2.jsonl")
OUTPUT_FILE = Path("data/generated/semantic_matches_v2.jsonl")
CHECKPOINT_FILE = Path("data/generated/semantic_checkpoint_v2.txt")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 64
SLEEP = 0.3

PROMPT = """\
You are evaluating an AI code review suggestion against a human reviewer's comment.

Human review: {human_review}
AI suggestion: {ai_suggestion}

Does the AI suggestion address the same issue as the human review?
- yes: the AI caught the same problem or improvement
- no: the AI is talking about something completely different
- uncertain: there is some overlap but not a clear match

Reply with exactly one word: yes, no, or uncertain.
"""

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

def get_match_label(client: anthropic.Anthropic, human_review: str, ai_suggestion: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": PROMPT.format(
                human_review=human_review[:500],
                ai_suggestion=ai_suggestion[:500]
            )
        }]
    )
    raw = response.content[0].text.strip().lower()
    if "yes" in raw:
        return "valid"
    elif "uncertain" in raw:
        return "uncertain"
    else:
        return "unmatched"

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(data)} examples")

    start_idx = load_checkpoint()
    if start_idx > 0:
        print(f"Resuming from index {start_idx}")

    errors = []
    mode = "a" if start_idx > 0 else "w"

    label_counts = {"valid": 0, "unmatched": 0, "uncertain": 0}

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        for i, example in enumerate(data):
            if i < start_idx:
                continue

            human_review = example.get("human_review", "").strip()
            ai_suggestion = example.get("ai_suggestion", "").strip()

            if not human_review or not ai_suggestion:
                print(f"  [{i}] Skipping — missing field")
                save_checkpoint(i + 1)
                continue

            try:
                label = get_match_label(client, human_review, ai_suggestion)
            except Exception as e:
                print(f"  [{i}] ERROR: {e}")
                errors.append({"index": i, "error": str(e)})
                time.sleep(2)
                continue

            label_counts[label] += 1

            record = {
                "index": example["index"],
                "old_code": example.get("old_code", ""),
                "human_review": human_review,
                "ai_suggestion": ai_suggestion,
                "language": example.get("language", "unknown"),
                "match_label": label,
            }
            f.write(json.dumps(record) + "\n")
            save_checkpoint(i + 1)

            if i % 50 == 0:
                print(f"  Progress: {i+1}/{len(data)} | {label_counts}")

            time.sleep(SLEEP)

    print(f"\nDone. Label distribution: {label_counts}")
    print(f"Output: {OUTPUT_FILE}")

    if errors:
        err_path = Path("data/generated/semantic_errors.json")
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"{len(errors)} errors logged to {err_path}")

if __name__ == "__main__":
    main()