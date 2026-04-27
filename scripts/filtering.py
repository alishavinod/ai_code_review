"""
filtering_v2.py — Type-specific noise filtering for AI code review suggestions.

Applies three filtering strategies and measures effectiveness:
  1. Baseline     : keep all suggestions (no filtering)
  2. Binary       : keep only valid matches, remove all noise
  3. Type-specific: per-category intervention based on noise type
       - trivial        : keyword/pattern heuristic (simulates linter pre-filtering)
       - incorrect      : two-stage LLM validation (simulates BitsAI-CR ReviewFilter)
       - irrelevant     : LLM scope check (simulates fine-tuning scope signal)
       - context-missing: keep with flag (RAG/few-shot is the fix, not removal)
       - valid          : always keep

Input:  data/generated/noise_labels_v2.jsonl
Output: data/generated/filtering_results.json
        data/generated/filtering_kept_typespecific.jsonl
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
RESULTS_FILE = Path("data/generated/filtering_results.json")
KEPT_FILE    = Path("data/generated/filtering_kept_typespecific.jsonl")

MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 10
SLEEP      = 0.3

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

def is_trivial_by_heuristic(ai_suggestion: str) -> bool:
    text = ai_suggestion.lower()
    return any(kw in text for kw in TRIVIAL_KEYWORDS)

INCORRECT_VALIDATION_PROMPT = """\
You are a code review validator. Given the code change below, determine whether the AI suggestion is factually correct.

BEFORE the change (deleted lines):
{old_code}

AFTER the change (added lines):
{new_code}

AI suggestion: {ai_suggestion}

Is this suggestion factually correct based on what is visible in the diff above?
Answer with exactly one word: yes or no."""

def validate_incorrect(client, old_code, new_code, ai_suggestion):
    """Returns True if suggestion is correct (keep), False if incorrect (filter)."""
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content":
                    INCORRECT_VALIDATION_PROMPT
                        .replace("{old_code}", old_code[:1200])
                        .replace("{new_code}", new_code[:1200])
                        .replace("{ai_suggestion}", ai_suggestion[:500])
                }]
            )
            answer = response.content[0].text.strip().lower()
            return "yes" in answer
        except Exception as e:
            if attempt == 0:
                print(f"    Retry: {e}")
                time.sleep(3)
            else:
                return True  
    return True

IRRELEVANT_SCOPE_PROMPT = """\
You are a code review scope checker. Given the code change below, determine whether the AI suggestion is about the specific change shown.

BEFORE the change (deleted lines):
{old_code}

AFTER the change (added lines):
{new_code}

AI suggestion: {ai_suggestion}

Is this suggestion directly related to the code change shown above?
Answer with exactly one word: yes or no."""

def check_relevance(client, old_code, new_code, ai_suggestion):
    """Returns True if relevant (keep), False if irrelevant (filter)."""
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content":
                    IRRELEVANT_SCOPE_PROMPT
                        .replace("{old_code}", old_code[:1200])
                        .replace("{new_code}", new_code[:1200])
                        .replace("{ai_suggestion}", ai_suggestion[:500])
                }]
            )
            answer = response.content[0].text.strip().lower()
            return "yes" in answer
        except Exception as e:
            if attempt == 0:
                print(f"    Retry: {e}")
                time.sleep(3)
            else:
                return True  
    return True

def compute_metrics(examples, kept_indices):
    tp = sum(1 for e in examples if e["index"] in kept_indices and e["match_label"] == "valid")
    fp = sum(1 for e in examples if e["index"] in kept_indices and e["match_label"] != "valid")
    fn = sum(1 for e in examples if e["index"] not in kept_indices and e["match_label"] == "valid")
    tn = sum(1 for e in examples if e["index"] not in kept_indices and e["match_label"] != "valid")

    total_kept  = tp + fp
    total_valid = tp + fn
    total_noise = fp + tn

    precision       = tp / total_kept  if total_kept  > 0 else 0.0
    recall          = tp / total_valid if total_valid > 0 else 0.0
    f1              = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    noise_reduction = tn / total_noise if total_noise > 0 else 0.0

    return {
        "total_kept": total_kept, "total_filtered": len(examples) - total_kept,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(f1, 3), "noise_reduction": round(noise_reduction, 3),
    }

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def main():
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(data)} examples")

    noise_dist = defaultdict(int)
    match_dist = defaultdict(int)
    for d in data:
        noise_dist[d.get("noise_type", "unknown")] += 1
        match_dist[d.get("match_label", "unknown")] += 1

    print(f"Match label distribution: {dict(match_dist)}")
    print(f"Noise type distribution:  {dict(noise_dist)}")

    total_valid = match_dist.get("valid", 0)
    total_noise = sum(v for k, v in match_dist.items() if k != "valid")
    print(f"\nTotal valid: {total_valid} | Total noise: {total_noise} | Total: {len(data)}")

    print("Strategy 1: Baseline (keep all)")
    baseline_kept = {d["index"] for d in data}
    baseline_metrics = compute_metrics(data, baseline_kept)
    print(f"  Kept: {baseline_metrics['total_kept']} | "
          f"Precision: {baseline_metrics['precision']:.1%} | "
          f"Recall: {baseline_metrics['recall']:.1%} | "
          f"F1: {baseline_metrics['f1']:.1%} | "
          f"Noise reduced: {baseline_metrics['noise_reduction']:.1%}")

    print("Strategy 2: Binary filtering (keep valid only)")
    binary_kept = {d["index"] for d in data if d.get("match_label") == "valid"}
    binary_metrics = compute_metrics(data, binary_kept)
    print(f"  Kept: {binary_metrics['total_kept']} | "
          f"Precision: {binary_metrics['precision']:.1%} | "
          f"Recall: {binary_metrics['recall']:.1%} | "
          f"F1: {binary_metrics['f1']:.1%} | "
          f"Noise reduced: {binary_metrics['noise_reduction']:.1%}")

    print("Strategy 3: Type-specific filtering")
    print("  trivial - keyword heuristic (linter simulation)")
    print("  incorrect - two-stage LLM validation")
    print("  irrelevant - LLM scope check")
    print("  context-missing - keep with flag")
    print("  valid - always keep")
    print()

    typespecific_kept = set()
    per_category      = defaultdict(lambda: {"total": 0, "kept": 0, "filtered": 0})
    typespecific_records = []

    trivial_filtered   = 0
    incorrect_filtered = 0
    irrelevant_filtered = 0

    for i, example in enumerate(data):
        noise_type    = example.get("noise_type", "unknown")
        match_label   = example.get("match_label", "unknown")
        old_code      = example.get("old_code", "").strip()
        new_code      = example.get("new_code", "").strip()
        ai_suggestion = example.get("ai_suggestion", "").strip()
        idx           = example["index"]

        per_category[noise_type]["total"] += 1
        keep = False
        filter_reason = None

        if noise_type == "valid" or match_label == "valid":
            keep = True

        elif noise_type == "context-missing":
            keep = True
            filter_reason = "context-missing: kept with flag"

        elif noise_type == "trivial":
            if is_trivial_by_heuristic(ai_suggestion):
                keep = False
                filter_reason = "trivial: filtered by keyword heuristic"
                trivial_filtered += 1
            else:
                keep = True

        elif noise_type == "incorrect":
            is_correct = validate_incorrect(client, old_code, new_code, ai_suggestion)
            if is_correct:
                keep = True
            else:
                keep = False
                filter_reason = "incorrect: filtered by LLM validation"
                incorrect_filtered += 1
            time.sleep(SLEEP)

        elif noise_type == "irrelevant":
            is_relevant = check_relevance(client, old_code, new_code, ai_suggestion)
            if is_relevant:
                keep = True
            else:
                keep = False
                filter_reason = "irrelevant: filtered by scope check"
                irrelevant_filtered += 1
            time.sleep(SLEEP)

        else:
            keep = True

        if keep:
            typespecific_kept.add(idx)
            per_category[noise_type]["kept"] += 1
        else:
            per_category[noise_type]["filtered"] += 1

        typespecific_records.append({
            "index": idx,
            "noise_type": noise_type,
            "match_label": match_label,
            "kept": keep,
            "filter_reason": filter_reason,
            "ai_suggestion": ai_suggestion[:200],
        })

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(data)}")

    typespecific_metrics = compute_metrics(data, typespecific_kept)

    print("\nPer-category results (type-specific):")
    for cat, stats in sorted(per_category.items()):
        total    = stats["total"]
        filtered = stats["filtered"]
        rate     = filtered / total if total > 0 else 0
        print(f"  {cat:20s}: {filtered:3d}/{total:3d} filtered = {rate:.1%}")

    print(f"\n  trivial   filtered by heuristic : {trivial_filtered}")
    print(f"  incorrect filtered by LLM       : {incorrect_filtered}")
    print(f"  irrelevant filtered by LLM      : {irrelevant_filtered}")

    print("FILTERING STRATEGY COMPARISON")
    print(f"{'Strategy':<20} {'Kept':>6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Noise Red.':>11}")
    for name, metrics in [
        ("Baseline",      baseline_metrics),
        ("Binary",        binary_metrics),
        ("Type-specific", typespecific_metrics),
    ]:
        print(f"{name:<20} {metrics['total_kept']:>6} "
              f"{metrics['precision']:>9.1%} "
              f"{metrics['recall']:>7.1%} "
              f"{metrics['f1']:>7.1%} "
              f"{metrics['noise_reduction']:>10.1%}")

    results = {
        "data_summary": {
            "total": len(data),
            "total_valid": total_valid,
            "total_noise": total_noise,
            "noise_distribution": dict(noise_dist),
            "match_distribution": dict(match_dist),
        },
        "strategies": {
            "baseline":      baseline_metrics,
            "binary":        binary_metrics,
            "type_specific": typespecific_metrics,
        },
        "type_specific_detail": {
            "per_category": {
                k: {**v, "filter_rate": round(v["filtered"]/v["total"], 3) if v["total"] > 0 else 0}
                for k, v in per_category.items()
            },
            "interventions": {
                "trivial_filtered_by_heuristic": trivial_filtered,
                "incorrect_filtered_by_llm":     incorrect_filtered,
                "irrelevant_filtered_by_llm":    irrelevant_filtered,
            }
        }
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {RESULTS_FILE}")

    with open(KEPT_FILE, "w", encoding="utf-8") as f:
        for r in typespecific_records:
            if r["kept"]:
                f.write(json.dumps(r) + "\n")
    print(f"Kept suggestions saved to {KEPT_FILE}")

if __name__ == "__main__":
    main()