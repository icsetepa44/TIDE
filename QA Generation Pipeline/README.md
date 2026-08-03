# Evolving-Concept Q&A Generation Pipeline

Turn a corpus of **Bangladeshi customs/tax legal documents** into a grounded,
multi-type **temporal question-answer dataset** — one that asks *how each policy
concept changed over time* and cites every answer back to the exact source clause.

The pipeline runs end-to-end: **entity–concept extraction → sub-entities →
entity clustering → evolving-concept threads → Q&A generation** (9 question types).

> **Core idea:** embeddings only *propose* candidate matches; an LLM (Gemini) is the
> *judge* of entity equivalence and of how a concept evolves. Every generated Q&A pair
> is grounded in specific dated clauses.

---

## What's in the box

| | Numbers (this run) |
|---|---|
| Source documents / clauses | 644 docs · 4,940 clauses |
| Unique entities (incl. sub-entities) | 14,278 |
| Entity clusters (≥2 merged variants) | 13,451 (492) |
| Evolving concept threads | 288 |
| **Generated Q&A pairs** | **~3,200 across 9 types** |

**The 9 Q&A types:** `scenario_short_answer`, `temporal_scenario_short_answer`,
`temporal_mcq`, `event_sorting`, `open_ended`, `perturbation_detection`,
`context_misalignment`, `relative_time_qa`, `multi_entity_concept_qa`.

---

## Repository layout

```
qa-generation-pipeline/
├── README.md                 ← you are here
├── docs/PIPELINE.md          ← deep-dive: every stage, prompt, and design choice
├── requirements.txt
├── LICENSE
├── .gitignore
│
│  # ── pipeline code (flat: scripts import each other as modules) ──
├── break_subentities.py      shared: ApiKeyRotator, KeyPool, watchdog, logging
├── cluster_entities.py       shared: KeyPool, judge rotator; 5-stage clustering
├── split_by_cluster.py
├── concept_cluster.py
├── export_threads.py
├── generate_qa.py            shared: robust QA parser
├── expand_scenarios.py       + _repair_loop.sh
├── add_temporal_qa.py        + _temporal_loop.sh
├── map_qa_clauses.py
├── add_relative_time_qa.py
├── add_multi_entity_qa.py
├── split_qa_by_type.py
├── regenerate_qa.py          v2 rebuild (paid key, grounded)
├── regen_event_sorting.py    date-free event ordering
│
│  # ── data (committed) ──
├── merged_ec_output_canon.json               input (entity–concept extraction)
├── merged_ec_output_canon_subentities.json   after break_subentities
├── merged_ec_output_canon_clustered.json     after cluster_entities
├── entities_index.json · entity_blocks.json · entity_clusters.json
├── concept_results.json · concept_clusters_index.json
├── concept_threads/          evolving threads with full clause objects
├── concept_qa/               per-thread Q&A (the live dataset)
├── concept_qa_multi/         cross-concept Q&A
├── concept_qa_by_type/       Q&A pooled into one file per type ← consumption-ready
└── clusters_by_id_sample/    a few example per-cluster files (full dir is git-ignored)
```

> **Flat by design.** The scripts import each other (`from break_subentities import …`,
> `from cluster_entities import …`) and read their data files as siblings, so all code
> and data live at the repository root. Do not move scripts into subfolders.

---

## Setup

```bash
# 1. Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. API keys (git-ignored — never committed)
cp all_api.txt.example    all_api.txt      # paste your FREE Gemini keys, one per line
cp gemini_api.txt.example gemini_api.txt   # paste your single PAID Gemini key
```

- **Free keys** (`all_api.txt`) drive the high-volume `generateContent` work.
- **Paid key** (`gemini_api.txt`) is used only for **embeddings** and the **v2 regeneration** (quota the free keys can't cover). Paid calls are billed on generated tokens, so `503` retries are free.

---

## Run the pipeline

Every stage is **resume-aware** (checkpoints / done-lists / per-file presence), so a
killed run continues where it stopped. Most scripts accept `--limit N` / `--only <id>`
for smoke tests, and `--force` to redo a stage.

```bash
# ── CLUSTERING ────────────────────────────────────────────────────────────
python break_subentities.py                 # + entity_sub_entities
python cluster_entities.py                   # 5 sub-stages → …_clustered.json
python split_by_cluster.py                   # → clusters_by_id/  (regenerates the full dir)
python concept_cluster.py                    # → concept_results.json
python export_threads.py                     # → concept_threads/

# ── Q&A GENERATION (v1 build) ─────────────────────────────────────────────
python generate_qa.py                        # 6 base types → concept_qa/
./_repair_loop.sh                            # expand scenarios into stories
./_temporal_loop.sh                          # add temporal-scenario type
python map_qa_clauses.py                     # attach clause_text_ids
python add_relative_time_qa.py && python map_qa_clauses.py
python add_multi_entity_qa.py                # → concept_qa_multi/
python split_qa_by_type.py                   # → concept_qa_by_type/

# ── Q&A REGENERATION (v2, higher quality, PAID key) ───────────────────────
python regenerate_qa.py --model gemini-3.5-flash   # rebuild all 9 types, grounded
python regen_event_sorting.py                       # make event_sorting date-free
python split_qa_by_type.py                          # refresh the pooled view
```

---

## Data notes (what's committed vs. regenerable)

This repo ships the **input, intermediates, and generated Q&A** so you can inspect the
dataset without re-running anything. Three things are **git-ignored** to keep the repo
lean (all are reproducible or secret):

| Item | Why excluded | How to get it |
|---|---|---|
| `all_api.txt`, `gemini_api.txt` | **Secrets** | Copy the `.example` files and paste your keys |
| `clusters_by_id/` (389 MB, ~13k files) | Large & fully regenerable | `python split_by_cluster.py` (a sample is in `clusters_by_id_sample/`) |
| `entity_embeddings.npy` (42 MB) | Large binary | `python cluster_entities.py --stage embed` (needs the paid key), or track with [Git LFS](https://git-lfs.com) |
| `*.log`, `*_done.json`, `concept_qa_*_v1/`, `concept_qa_backup*/` | Run logs / checkpoints / one-time backups | Recreated automatically on the next run |

If you want to version the large binaries too, initialise **Git LFS** and remove the
matching lines from `.gitignore`:

```bash
git lfs install
git lfs track "entity_embeddings.npy"
```

---

## Initialise the git repo

```bash
cd qa-generation-pipeline
git init
git add .
git status          # confirm all_api.txt / gemini_api.txt are NOT listed
git commit -m "Evolving-concept Q&A generation pipeline"
```

> **Before pushing**, double-check `git status` / `git ls-files` does not include
> `all_api.txt` or `gemini_api.txt`. The `.gitignore` already excludes them.

---

## License

Released under the [MIT License](LICENSE). Update the copyright holder in `LICENSE`
before publishing.
