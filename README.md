# Noise Categorization and Type-Specific Filtering for AI Code Review

**Alisha Vinod - av3311**

A five-stage pipeline for classifying and filtering noise in AI-generated code review suggestions, evaluated on 300 examples from the CodeReviewQA benchmark dataset.

---

## What This Project Does

AI code review tools generate suggestions that developers reject at high rates. This project builds a pipeline that:

1. Generates AI review suggestions for real code diffs
2. Compares them to human reviewer comments (semantic matching)
3. Classifies mismatched suggestions into four noise categories: **incorrect**, **trivial**, **context-missing**, **irrelevant**
4. Validates the classifier through Cohen's Kappa, consistency, prompt sensitivity, and manual audit
5. Applies a targeted filtering intervention per noise category

**Key finding:** Type-specific filtering removes 40.4% of classifier-labeled noise while preserving all valid suggestions. 

---

## Repository Structure

```
.
├── scripts/                        # Pipeline scripts (run in order)
│   ├── load_dataset.py             # Samples 300 examples from CodeReviewQA
│   ├── generate_ai_reviews.py      # Stage 1: generates AI suggestions
│   ├── semantic_matching.py        # Stage 2: matches AI to human reviews
│   ├── noise_classification.py     # Stage 3: classifies noise into 4 categories
│   ├── validate_labels.py          # Stage 4a: LLM-as-judge validation (Kappa)
│   ├── compute_kappa.py            # Stage 4b: computes Cohen's Kappa
│   ├── consistency_check.py        # Stage 4c: label stability test
│   ├── prompt_sensitivity.py       # Stage 4d: prompt variant test
│   ├── generate_audit_excel.py     # Generates manual audit spreadsheet
│   ├── filtering.py                # Stage 5: type-specific filtering
│   ├── ablation.py                 # Ablation study (each intervention in isolation)
│   ├── demo.py                     # Live demo script
│   └── config.py                   # Shared configuration
│
├── data/
│   ├── raw/
│   │   ├── codereviewqa_sample_300.json   # 300-example dataset sample
│   │   └── codereviewqa_sample_50.json    # 50-example dev sample
│   ├── generated/                         # Pipeline outputs (all stages)
│   │   ├── ai_suggestions.jsonl           # Stage 1 output
│   │   ├── semantic_matches.jsonl         # Stage 2 output
│   │   ├── noise_labels.jsonl             # Stage 3 output
│   │   ├── validation_results.jsonl       # Stage 4a output
│   │   ├── validation_summary.json        # Kappa summary
│   │   ├── consistency_results.json       # Stage 4c output
│   │   ├── prompt_sensitivity_results.json # Stage 4d output
│   │   ├── filtering_results.json         # Stage 5 output
│   │   ├── filtering_kept_typespecific.jsonl # Kept suggestions
│   │   └── ablation_results.json          # Ablation study output
│   └── audit/
│       └── manual_audit.xlsx              # Manual audit spreadsheet
│
├── milestones/
│   ├── 6156_project_proposal.pdf
│   └── 6156_project_progress_report.pdf
|   └── Final_report_6156.pdf
│
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+, Anthropic API key

```bash
# Clone the repo
git clone https://github.com/alishavinod/ai_code_review
cd ai_code_review

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

---

## How to Reproduce

Run scripts in order from the project root:

```bash
# Stage 1: Generate AI review suggestions
python scripts/generate_ai_reviews.py

# Stage 2: Semantic matching against human reviews
python scripts/semantic_matching.py

# Stage 3: Noise classification
python scripts/noise_classification.py

# Stage 4: Validation
python scripts/validate_labels.py        # LLM-as-judge
python scripts/compute_kappa.py          # Cohen's Kappa 
python scripts/consistency_check.py      # Label stability
python scripts/prompt_sensitivity.py     # Prompt variants

# Stage 5: Filtering
python scripts/filtering.py              # Type-specific filtering

# Ablation study
python scripts/ablation.py              # Individual filtering analysis
```

Each script reads from `data/generated/` and writes its output back to `data/generated/`. Checkpointing is implemented at every write — if a script is interrupted, re-running it will resume from the last checkpoint.

**Estimated API cost:** Full pipeline run ~$0.50 using Claude Haiku. Validation steps use Claude Sonnet and cost ~$0.20 additional.

---

## Demo

The demo script runs two examples through the full pipeline live:

```bash
python scripts/demo.py pipeline   # explain pipeline stages
python scripts/demo.py 1          # Example 1: valid suggestion (kept)
python scripts/demo.py 2          # Example 2: context-missing unmatched suggestion (filtered)
python scripts/demo.py results    # show results summary
python scripts/demo.py all        # full demo
```

---

## Results Summary

| Metric | Value |
|---|---|
| Valid match rate | 21.7% (65/300) |
| Dominant noise category | Incorrect (41.3%) |
| Cohen's Kappa | 0.413 (Moderate) |
| Classifier consistency | 100% |
| Type-specific noise reduction | 40.4% |
| False negative rate | 0% |

**Noise distribution:**

| Category | Count | % |
|---|---|---|
| Incorrect | 97 | 41.3% |
| Context-missing | 77 | 32.8% |
| Trivial | 58 | 24.7% |
| Irrelevant | 3 | 1.3% |

---

## AI Tools Used

This project uses AI tools throughout the pipeline and development 
process, as documented below:

### Pipeline (core system)
- **Claude Haiku** (`claude-haiku-4-5-20251001`) - used for all 
  generation, semantic matching, noise classification, two-stage 
  LLM validation, and relevance verification API calls. Default 
  configuration, temperature not modified.
- **Claude Sonnet** (`claude-sonnet-4-5`) - used as independent 
  validator for Cohen's Kappa inter-annotator agreement. Default 
  configuration.

### Development assistance
- **Claude (claude.ai)** - used for research assistance, code generation for some of the scripts, code review, and debugging during development. 
  All AI-generated code and content was reviewed and verified before use.

Other than this, the below activities were performed by me:
- Dataset sampling and selection
- Manual audit labels (18 examples verified by hand)
- Research question design and experimental methodology
- Interpretation of results and findings

## Dataset

**CodeReviewQA** - publicly available at:
https://huggingface.co/datasets/Tomo-Melb/CodeReviewQA

300 examples sampled spanning C, C++, and C# code changes from real open source pull requests.

---

## Milestones

All milestone documents are in `milestones/`:
- Project proposal: `milestones/6156_project_proposal.pdf`
- Progress report: `milestones/6156_project_progress_report.pdf`
- Final report: `milestones/Final_report_6156.pdf`