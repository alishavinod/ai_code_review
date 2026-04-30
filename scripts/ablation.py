"""
ablation.py — Ablation study for type-specific filtering

Tests each filtering intervention in isolation to measure individual contribution:
  1. Baseline         : no filtering
  2. Heuristic only   : keyword heuristic for trivial only
  3. Validator only   : two-stage LLM validation for incorrect only
  4. Relevance only   : LLM relevance verification for irrelevant only
  5. All three        : full type-specific filtering (already run)

Input:  data/generated/noise_labels_v2.jsonl  (use v1 labels - more balanced)
        data/generated/noise_labels.jsonl      (v1 labels)
Output: data/generated/ablation_results.json
"""

import os
import json
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import anthropic

load_dotenv()

INPUT_FILE   = Path("data/generated/noise_labels.jsonl")  
OUTPUT_FILE  = Path("data/generated/ablation_results.json")

MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 10
SLEEP      = 0.3

VALIDATE_PROMPT = """\
You are a code review validator. Given the code change below, determine whether \
the AI suggestion is factually correct.

BEFORE the change (deleted lines):
{old_code}

AFTER the change (added lines):
{new_code}

AI suggestion: {ai_suggestion}

Is this suggestion factually correct based on what is visible in the diff above?
Answer with exactly one word: yes or no."""

RELEVANCE_PROMPT = """\
You are a code review scope checker. Given the code change below, determine \
whether the AI suggestion is about the specific change shown.

BEFORE the change (deleted lines):
{old_code}

AFTER the change (added lines):
{new_code}

AI suggestion: {ai_suggestion}

Is this suggestion directly related to the code change shown above?
Answer with exactly one word: yes or no."""

TRIVIAL_KEYWORDS = [
    "naming", "name convention", "variable name", "function name",
    "camelcase", "snake_case", "pascal", "uppercase", "lowercase", "rename",
    "whitespace", "indentation", "indent", "spacing", "blank line", "newline",
    "trailing space", "tab", "formatting", "format",
    "comment", "documentation", "docstring", "missing comment", "add comment",
    "typo", "spelling", "grammar",
    "style", "nitpick", "nit:", "cosmetic", "readability", "consistent",
    "consistency", "code style",
]

def is_trivial(text):
    t = text.lower()
    return any(kw in t for kw in TRIVIAL_KEYWORDS)

def call(client, prompt):
    for attempt in range(2):
        try:
            r = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            return r.content[0].text.strip().lower()
        except Exception as e:
            if attempt == 0:
                print(f"    Retry: {e}")
                time.sleep(3)
            else:
                return "yes"  
    return "yes"

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def compute_metrics(examples, kept_indices):
    tp = sum(1 for e in examples
             if e["index"] in kept_indices and e["match_label"] == "valid")
    fp = sum(1 for e in examples
             if e["index"] in kept_indices and e["match_label"] != "valid")
    fn = sum(1 for e in examples
             if e["index"] not in kept_indices and e["match_label"] == "valid")
    tn = sum(1 for e in examples
             if e["index"] not in kept_indices and e["match_label"] != "valid")

    total_kept  = tp + fp
    total_noise = fp + tn

    precision       = tp / total_kept  if total_kept  > 0 else 0.0
    noise_reduction = tn / total_noise if total_noise > 0 else 0.0

    return {
        "kept":            total_kept,
        "filtered":        len(examples) - total_kept,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":       round(precision, 3),
        "noise_reduction": round(noise_reduction, 3),
    }

def run_config(client, data, name, use_heuristic, use_validator, use_relevance):
    print(f"\n  Running: {name}")
    kept = set()
    filtered_counts = defaultdict(int)

    for i, example in enumerate(data):
        noise_type    = example.get("noise_type", "unknown")
        match_label   = example.get("match_label", "unknown")
        old_code      = example.get("old_code", "").strip()
        new_code      = example.get("new_code", "").strip()
        ai_suggestion = example.get("ai_suggestion", "").strip()
        idx           = example["index"]

        if noise_type == "valid" or match_label == "valid":
            kept.add(idx)
            continue

        if noise_type == "context-missing":
            kept.add(idx)
            continue

        if noise_type == "trivial":
            if use_heuristic and is_trivial(ai_suggestion):
                filtered_counts["trivial"] += 1
            else:
                kept.add(idx)
            continue

        if noise_type == "incorrect":
            if use_validator:
                answer = call(client, VALIDATE_PROMPT.format(
                    old_code=old_code[:1200],
                    new_code=new_code[:1200],
                    ai_suggestion=ai_suggestion[:500]
                ))
                if "yes" in answer:
                    kept.add(idx)
                else:
                    filtered_counts["incorrect"] += 1
                time.sleep(SLEEP)
            else:
                kept.add(idx)
            continue

        if noise_type == "irrelevant":
            if use_relevance:
                answer = call(client, RELEVANCE_PROMPT.format(
                    old_code=old_code[:1200],
                    new_code=new_code[:1200],
                    ai_suggestion=ai_suggestion[:500]
                ))
                if "yes" in answer:
                    kept.add(idx)
                else:
                    filtered_counts["irrelevant"] += 1
                time.sleep(SLEEP)
            else:
                kept.add(idx)
            continue

        kept.add(idx)

    metrics = compute_metrics(data, kept)
    metrics["filtered_by_category"] = dict(filtered_counts)
    print(f"    Kept: {metrics['kept']} | "
          f"Precision: {metrics['precision']:.1%} | "
          f"Noise reduced: {metrics['noise_reduction']:.1%}")
    return metrics

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(data)} examples")

    noise_dist = defaultdict(int)
    match_dist = defaultdict(int)
    for d in data:
        noise_dist[d.get("noise_type", "unknown")] += 1
        match_dist[d.get("match_label", "unknown")] += 1
    print(f"Noise distribution: {dict(noise_dist)}")
    print(f"Match distribution: {dict(match_dist)}")

    configs = [
        ("Baseline",               False, False, False),
        ("Heuristic only",         True,  False, False),
        ("Validator only",         False, True,  False),
        ("Relevance only",         False, False, True),
        ("All three (full)",       True,  True,  True),
    ]

    results = {}
    for name, h, v, r in configs:
        results[name] = run_config(client, data, name, h, v, r)

    print("\n" + "=" * 65)
    print("=== ABLATION STUDY RESULTS ===")
    print(f"{'Strategy':<25} {'Kept':>6} {'Precision':>10} "
          f"{'Noise Red':>10} {'Filtered':>10}")
    print("-" * 65)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['kept']:>6} "
              f"{metrics['precision']:>9.1%} "
              f"{metrics['noise_reduction']:>9.1%} "
              f"{metrics['filtered']:>10}")
        
    output = {
        "data_summary": {
            "total": len(data),
            "noise_distribution": dict(noise_dist),
            "match_distribution": dict(match_dist),
        },
        "ablation_results": results,
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()