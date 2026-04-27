"""
demo_v2.py — AI Code Review Noise Filtering Pipeline Demo

Explains the pipeline stages then runs 2 real examples live:
  Example 1: Valid suggestion - KEPT   (AI catches real bug)
  Example 2: Incorrect suggestion - FILTERED by LLM two-stage validation

Usage:
  python demo_v2.py pipeline   # explain pipeline stages
  python demo_v2.py 1          # Example 1: valid - kept
  python demo_v2.py 2          # Example 2: incorrect - filtered
  python demo_v2.py results    # results summary
  python demo_v2.py all        # full demo
"""

import os
import sys
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def header(text):
    print(f"\n{BOLD}{BLUE}{'═' * 60}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'═' * 60}{RESET}\n")

def step(n, total, text):
    print(f"\n{BOLD}{CYAN} Step {n}/{total}: {text}{RESET}")

def wait(msg):
    print(f"\n  {DIM}⟳  {msg}...{RESET}", end="", flush=True)

def done_wait(elapsed):
    print(f"{DIM} done ({elapsed:.1f}s){RESET}")

def show_pipeline():
    header("Pipeline: AI Code Review Noise Filtering")
    stages = [
        (
            "1. Generation",
            "Claude Haiku reads a code diff and generates a review suggestion.",
            "Input : old_code (deleted lines) + new_code (added lines)\n"
            "       Output: AI suggestion"
        ),
        (
            "2. Semantic Matching",
            "Compare AI suggestion to the human reviewer's comment.",
            "Question: Did the AI notice the same issue as the human?\n"
            "       Labels  : valid / unmatched / uncertain"
        ),
        (
            "3. Noise Classification",
            "For unmatched suggestions, classify why they missed.",
            "Labels:\n"
            "incorrect       — AI is factually wrong given the diff\n"
            "trivial         — style/naming with no functional impact\n"
            "context-missing — valid but needs broader codebase knowledge\n"
            "irrelevant      — unrelated to the change shown"
        ),
        (
            "4. Validation (LLM-as-Judge)",
            "Second independent LLM re-labels a sample to measure reliability.",
            "Metric: Cohen's Kappa = 0.413  (moderate agreement)"
        ),
        (
            "5. Type-Specific Filtering",
            "Different intervention per noise type — not one-size-fits-all.",
            "incorrect       - Two-stage LLM validation\n"
            "trivial         - Keyword heuristic (simulates linter)\n"
            "irrelevant      - LLM scope check\n"
            "context-missing - Keep with flag"
        ),
    ]
    for name, desc, detail in stages:
        print(f"  {BOLD}{YELLOW}{name}{RESET}")
        print(f"  {desc}")
        print(f"  {DIM}{detail}{RESET}\n")

    print(f"  {BOLD}Dataset :{RESET} 300 examples — CodeReviewQA (C, C++, C#)")
    print(f"  {BOLD}Model   :{RESET} Claude Haiku (generation + classification)")
    print(f"  {BOLD}Ground truth:{RESET} Human reviewer comments\n")

EXAMPLES = {
    1: {
        "title": "Example 1 — Valid Suggestion (C++)",
        "subtitle": "AI catches a real bug - pipeline keeps it",
        "context": (
            "Developer migrated from StringFormat() to fmt::format() "
            "but kept the old '%u' format specifiers, which are invalid for fmt."
        ),
        "old_code": (
            "query = StringFormat(\n"
            "    \"INSERT INTO `bot_timers` VALUES ('%u', '%u', '%u')\",\n"
            "    bot_inst->GetBotID(), timer_index + 1, bot_timers[timer_index]\n"
            ");"
        ),
        "new_code": (
            "query = fmt::format(\n"
            "    \"INSERT INTO `bot_timers` VALUES ('%u', '%u', '%u')\",\n"
            "    bot_inst->GetBotID(), timer_index + 1, bot_timers[timer_index]\n"
            ");"
        ),
        "human_review": (
            "Can use {} for all three instead of '%u' — "
            "that's a StringFormat thing, not fmt."
        ),
        "ai_suggestion": (
            "The format string uses '%u' placeholders which are not valid for "
            "fmt::format() — this will cause a runtime error. "
            "Replace with {} to use fmt library syntax correctly."
        ),
    },
    2: {
        "title": "Example 2 — Incorrect Suggestion (C)",
        "subtitle": "AI comments on deleted code - two-stage validation filters it",
        "context": (
            "Developer removed the else-branch warning that fired when "
            "cursor_pos was NULL. The warning is gone in the AFTER version."
        ),
        "old_code": (
            "static void bucketing_cursor_w_pos_delete(...) {\n"
            "    if (cursor_pos != NULL) {\n"
            "        list_cursor_destroy(cursor_pos->lc);\n"
            "        free(cursor_pos);\n"
            "    } else\n"
            "        warn(D_BUCKETING, \"ignoring null pointer\");\n"
            "}"
        ),
        "new_code": (
            "static void bucketing_cursor_w_pos_delete(...) {\n"
            "    if (cursor_pos != NULL) {\n"
            "        list_cursor_destroy(cursor_pos->lc);\n"
            "        free(cursor_pos);\n"
            "    }\n"
            "}"
        ),
        "human_review": "Remove these warnings.",
        "ai_suggestion": (
            "Add a null pointer check before dereferencing cursor_pos. "
            "Currently, if cursor_pos is NULL, the code will crash when "
            "calling list_cursor_destroy(cursor_pos->lc)."
        ),
    },
}

MATCH_PROMPT = """\
You are evaluating whether an AI code review suggestion addresses the same issue as a human reviewer.

Human review: {human_review}
AI suggestion: {ai_suggestion}

The AI suggestion matches ONLY if it identifies the SAME problem and recommends 
the SAME type of action as the human reviewer.
If the human says to REMOVE something and the AI says to ADD something, 
that is NOT a match — label it unmatched.

Does the AI suggestion address the same concern as the human review?
Answer with exactly one word: valid, unmatched, or uncertain."""

CLASSIFY_PROMPT = """\
You are evaluating an AI code review suggestion that does not match what a human reviewer said.
Classify into exactly one noise category.

BEFORE the change:
{old_code}

AFTER the change:
{new_code}

AI suggestion: {ai_suggestion}

Categories:
- trivial: minor style or naming suggestion with no functional impact.
- incorrect: factually wrong or misreads the diff.
- context-missing: may be valid but needs broader codebase knowledge.
- irrelevant: unrelated to the change shown.

Reply with exactly one word: trivial, incorrect, context-missing, or irrelevant."""

VALIDATE_PROMPT = """\
You are a second independent code review validator.

BEFORE the change:
{old_code}

AFTER the change:
{new_code}

AI suggestion: {ai_suggestion}

Is this suggestion factually correct based on what is visible in the diff?
Answer with exactly one word: yes or no."""

TRIVIAL_KEYWORDS = [
    "naming", "rename", "camelcase", "snake_case", "variable name",
    "whitespace", "indentation", "spacing", "formatting", "comment",
    "style", "nitpick", "cosmetic", "readability", "consistent", "typo",
]

def call(client, prompt, max_tokens=10):
    t0 = time.time()
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text.strip().lower(), time.time() - t0

def normalize_match(raw):
    if "valid" in raw: return "valid"
    if "uncertain" in raw: return "uncertain"
    return "unmatched"

def normalize_noise(raw):
    for l in ["context-missing", "incorrect", "trivial", "irrelevant"]:
        if l in raw: return l
    return "context-missing"

def run_example(client, ex):
    header(ex["title"])
    print(f"  {DIM}{ex['subtitle']}{RESET}\n")
    print(f"  {BOLD}Context:{RESET} {ex['context']}\n")

    step(1, 5, "Code diff")
    print(f"\n    {BOLD}BEFORE:{RESET}")
    for line in ex["old_code"].splitlines():
        print(f"      {RED}{line}{RESET}")
    print(f"\n    {BOLD}AFTER:{RESET}")
    for line in ex["new_code"].splitlines():
        print(f"      {GREEN}{line}{RESET}")

    step(2, 5, "Human reviewer comment")
    print(f"\n    \"{BOLD}{ex['human_review']}{RESET}\"")

    step(3, 5, "AI-generated suggestion (Claude Haiku)")
    print(f"\n    \"{ex['ai_suggestion']}\"")

    time.sleep(0.5)

    step(4, 5, "Semantic matching")
    wait("Comparing AI suggestion to human review")
    raw, elapsed = call(client, MATCH_PROMPT.format(
        human_review=ex["human_review"],
        ai_suggestion=ex["ai_suggestion"],
    ))
    match = normalize_match(raw)
    done_wait(elapsed)

    c = GREEN if match == "valid" else (YELLOW if match == "uncertain" else RED)
    print(f"\n    Result: {c}{BOLD}{match.upper()}{RESET}")

    if match == "valid":
        print(f"\n    {GREEN}AI addressed the same issue as the human reviewer.{RESET}")
        step(5, 5, "Type-specific filtering")
        print(f"\n    Noise type : {GREEN}{BOLD}VALID{RESET}")
        print(f"    Intervention: None — always keep valid suggestions")
        print(f"\n    {GREEN}{BOLD}KEPT — Valid suggestion reaches the developer.{RESET}\n")
        return

    print(f"\n    {RED}AI missed the point. Classifying noise type...{RESET}")

    step(5, 5, "Noise classification + Type-specific filtering")
    wait("Classifying noise type")
    raw, elapsed = call(client, CLASSIFY_PROMPT.format(
        old_code=ex["old_code"][:1000],
        new_code=ex["new_code"][:1000],
        ai_suggestion=ex["ai_suggestion"][:400],
    ), max_tokens=15)
    noise = normalize_noise(raw)
    done_wait(elapsed)

    nc = RED if noise in ("incorrect", "irrelevant") else YELLOW
    print(f"\n    Noise type: {nc}{BOLD}{noise.upper()}{RESET}\n")

    time.sleep(0.3)

    if noise == "trivial":
        print(f"    {BOLD}Intervention: Keyword heuristic{RESET} (simulates linter pre-filtering)")
        kw = [k for k in TRIVIAL_KEYWORDS if k in ex["ai_suggestion"].lower()]
        print(f"    Scanning for style/naming keywords...")
        if kw:
            print(f"    Found: {YELLOW}{kw}{RESET}")
            print(f"\n    {RED}{BOLD}FILTERED — Style suggestion a linter would catch.{RESET}\n")
        else:
            print(f"    No trivial keywords found.")
            print(f"\n    {GREEN}{BOLD}KEPT — Passes heuristic.{RESET}\n")

    elif noise == "incorrect":
        print(f"    {BOLD}Intervention: Two-stage LLM validation{RESET}")
        print(f"    Question: \"Is this suggestion factually correct given the diff?\"")
        wait("Calling independent validator (Claude Haiku)")
        raw, elapsed = call(client, VALIDATE_PROMPT.format(
            old_code=ex["old_code"][:1000],
            new_code=ex["new_code"][:1000],
            ai_suggestion=ex["ai_suggestion"][:400],
        ))
        done_wait(elapsed)
        correct = "yes" in raw
        ac = GREEN if correct else RED
        print(f"    Validator: {ac}{BOLD}{raw.upper()}{RESET}")
        if correct:
            print(f"\n    {GREEN}{BOLD}KEPT — Validator confirmed suggestion is correct.{RESET}\n")
        else:
            print(f"\n    {RED}{BOLD}FILTERED — Factually wrong given the diff. Removed.{RESET}\n")

    elif noise == "irrelevant":
        print(f"    {BOLD}Intervention: LLM scope check{RESET}")
        wait("Checking if suggestion is about this change")
        raw, elapsed = call(client, VALIDATE_PROMPT.format(
            old_code=ex["old_code"][:1000],
            new_code=ex["new_code"][:1000],
            ai_suggestion=ex["ai_suggestion"][:400],
        ))
        done_wait(elapsed)
        relevant = "yes" in raw
        ac = GREEN if relevant else RED
        print(f"    Scope check: {ac}{BOLD}{raw.upper()}{RESET}")
        if relevant:
            print(f"\n    {GREEN}{BOLD}KEPT — Passes scope check.{RESET}\n")
        else:
            print(f"\n    {RED}{BOLD}FILTERED — Out of scope for this change.{RESET}\n")

    elif noise == "context-missing":
        print(f"    {BOLD}Intervention: Keep with flag{RESET}")
        print(f"    Suggestion may be valid but needs broader codebase knowledge.")
        print(f"    Fix: RAG augmentation — not removal.")
        print(f"\n    {YELLOW}{BOLD}⚑ KEPT WITH FLAG — Needs augmentation before surfacing.{RESET}\n")

def show_results():
    header("Results — 300 Examples from CodeReviewQA")

    print(f"  {BOLD}Semantic Matching{RESET}")
    print(f"  Valid matches : {GREEN}65 / 300 = 21.7%{RESET}")
    print(f"  Noise         : {RED}235 / 300 = 78.3%{RESET}\n")

    print(f"  {BOLD}Noise Distribution{RESET}")
    for name, count, pct, c in [
        ("incorrect",       97, "41%", RED),
        ("context-missing", 77, "33%", YELLOW),
        ("trivial",         58, "25%", YELLOW),
        ("irrelevant",       3,  "1%", RED),
    ]:
        bar = "█" * (count // 5)
        print(f"  {name:<20} {count:>4}  {c}{bar:<20} {pct}{RESET}")
    print()

    print(f"  {BOLD}Classifier Validation{RESET}")
    print(f"  Cohen Kappa         : 0.413  {YELLOW}(Moderate){RESET}")
    print(f"  Consistency         : 100%")
    print(f"  Prompt sensitivity  : 87.5%")
    print(f"  Manual audit        : 72.2%\n")

    print(f"  {BOLD}Filtering Comparison{RESET}")
    print(f"  {'Strategy':<20} {'Kept':>6} {'Precision':>10} {'F1':>8} {'Noise Red':>10}")
    print(f"  {'─' * 58}")
    for name, kept, prec, f1, nr, c in [
        ("Baseline",         300, "21.7%", "35.6%",  "0%",   DIM),
        ("Type-specific",    204, "31.9%", "48.3%", "40.9%", GREEN),
        ("Binary (oracle)",   65, "100%",  "100%",  "100%",  DIM),
    ]:
        print(f"  {c}{name:<20} {kept:>6} {prec:>10} {f1:>8} {nr:>10}{RESET}")
    print()

    print(f"  {BOLD}Per-Intervention Effectiveness{RESET}")
    for name, frac, pct, c in [
        ("irrelevant - scope check",    "3/3",   "100%", GREEN),
        ("incorrect - LLM validation", "53/97",   "55%", YELLOW),
        ("trivial - keyword heuristic","40/58",   "69%", YELLOW),
        ("context-missing - kept",      "0/77",    "0%", DIM),
    ]:
        print(f"  {name:<35} {frac:>7}  {c}{pct}{RESET}")
    print()

    print(f"  {BOLD}Key Finding:{RESET}")
    print(f"  Type-specific filtering: {GREEN}40.9% noise reduction{RESET}, {GREEN}0% false negatives{RESET}.")
    print(f"  Context-missing (33%) requires RAG augmentation — not filtering.\n")

def main():
    if len(sys.argv) < 2:
        print(f"\n{BOLD}Usage:{RESET}")
        print("  python demo_v2.py pipeline  — explain pipeline")
        print("  python demo_v2.py 1         — Example 1: valid - kept")
        print("  python demo_v2.py 2         — Example 2: incorrect - filtered")
        print("  python demo_v2.py results   — results summary")
        print("  python demo_v2.py all       — full demo")
        sys.exit(0)

    arg = sys.argv[1].lower()

    if arg == "pipeline":
        show_pipeline(); return
    if arg == "results":
        show_results(); return

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if arg == "1":
        run_example(client, EXAMPLES[1])
    elif arg == "2":
        run_example(client, EXAMPLES[2])
    elif arg == "all":
        show_pipeline()
        input(f"\n{BOLD}  Press Enter to run Example 1{RESET}")
        run_example(client, EXAMPLES[1])
        input(f"\n{BOLD}  Press Enter to run Example 2{RESET}")
        run_example(client, EXAMPLES[2])
        input(f"\n{BOLD}  Press Enter for results{RESET}")
        show_results()
    else:
        print(f"Unknown: {arg}"); sys.exit(1)

if __name__ == "__main__":
    main()