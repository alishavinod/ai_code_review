"""
Step 1a: Inspect and validate the CodeReviewQA dataset.
Input:  data/raw/codereviewqa_sample_300.json (or _50 for quick test)
"""

import json
from pathlib import Path
from collections import Counter

INPUT_FILE = Path("data/raw/codereviewqa_sample_300.json")  # swap to _50 for quick test

def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Skipping line {i}: {e}")
    return records

def main():
    print(f"Loading {INPUT_FILE}...")
    data = load_jsonl(INPUT_FILE)
    print(f"  Total examples: {len(data)}")

    # Field coverage
    print("\nField coverage:")
    for field in ["old", "new", "review", "lang"]:
        filled = sum(1 for d in data if d.get(field, "").strip())
        print(f"  {field}: {filled}/{len(data)}")

    # Language distribution
    lang_counts = Counter(d.get("lang", "unknown") for d in data)
    print("\nLanguage distribution:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")

    # Sample record
    print("\nSample record (index 0):")
    ex = data[0]
    for field in ["lang", "review"]:
        print(f"  {field}: {ex.get(field, '')[:200]}")
    old_preview = ex.get("old", "")[:200].replace("\n", "\\n")
    print(f"  old (first 200 chars): {old_preview}")

if __name__ == "__main__":
    main()