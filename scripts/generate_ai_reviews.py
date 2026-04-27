"""
Step 1b: Generate AI code review suggestions using Claude Haiku.
Input:  data/raw/codereviewqa_sample_300.json
Output: data/generated/ai_suggestions.jsonl
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

INPUT_FILE = Path("data/raw/codereviewqa_sample_300.json")
OUTPUT_FILE = Path("data/generated/ai_suggestions.jsonl")
CHECKPOINT_FILE = Path("data/generated/checkpoint.txt")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512
SLEEP = 0.3

PROMPT = """\
You are an experienced software engineer performing a code review.
Review the following code and provide one specific, actionable suggestion for improvement.
Be concise — one suggestion only, 1-3 sentences.

Code:
{code}
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

def generate_suggestion(client: anthropic.Anthropic, code: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": PROMPT.format(code=code[:3000])}],
    )
    return response.content[0].text.strip()

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

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        for i, example in enumerate(data):
            if i < start_idx:
                continue

            code = example.get("old", "").strip()
            if not code:
                print(f"  [{i}] Skipping — empty 'old' field")
                save_checkpoint(i + 1)
                continue

            try:
                ai_suggestion = generate_suggestion(client, code)
            except Exception as e:
                print(f"  [{i}] ERROR: {e}")
                errors.append({"index": i, "error": str(e)})
                time.sleep(2)
                continue

            record = {
                "index": i,
                "old_code": code,
                "new_code": example.get("new", ""),
                "human_review": example.get("review", ""),
                "language": example.get("lang", "unknown"),
                "ai_suggestion": ai_suggestion,
            }
            f.write(json.dumps(record) + "\n")
            save_checkpoint(i + 1)

            if i % 50 == 0:
                print(f"  Progress: {i+1}/{len(data)}")

            time.sleep(SLEEP)

    total = len(OUTPUT_FILE.read_text().strip().splitlines())
    print(f"\nDone. {total} records written to {OUTPUT_FILE}")

    if errors:
        err_path = Path("data/generated/errors.json")
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"{len(errors)} errors logged to {err_path}")

if __name__ == "__main__":
    main()