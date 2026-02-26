# Factorized resampling experiments

This folder contains **offline** experiment drivers that compute graph scores for
each paper in the collection, using a *factorized* resampling plan:

1. **Systematic paper collection** (Excel workbook)
2. **DAG extraction & validation** — resample **K** independent DAGs per paper
3. **Node scoring** — for each DAG, resample **M** node-scoring passes
4. **Graph scoring** — compute a score for each of the **K×M** trials

The intended primary entrypoints:

- `experiments/download_pdfs_from_collection.py` — mass download PDFs into a local folder structure.
- `experiments/factorized_collection_cli.py` — run the factorized scoring experiment over the collection.

## Minimal runtime file set

If your goal is *only* to run batch scoring experiments (no web UI), the minimal
runtime-relevant code paths are:

### Required

- `graph_app/kg_realtime_scoring.py` — DAG validation + KGScorer (graph score)
- `backend/prompts.py` — LLM prompts + JSON parsing/repair helpers
- `backend/llm_client.py` — offline LLM adapter (no UI protocol output)
- `backend/paper_io.py` — PDF text extraction helpers
- `backend/factorized_experiment.py` — factorized resampling driver
- `experiments/paper_collection.py` — spreadsheet reader
- `experiments/plotting.py` — KDE plotting helper
- `experiments/factorized_collection_cli.py` — main CLI
- `experiments/download_pdfs_from_collection.py` — optional downloader

### Not required for offline batch scoring

- `frontend/` (React app)
- `backend/server.py` (Flask server)
- `backend/main.py` (interactive demo w/ streaming)
- `experiments/batch_cli.py` (legacy; kept for compatibility)

## Typical workflow

0) Configure an LLM provider key (required to generate new cached runs):

```bash
# choose one
export OPENAI_API_KEY=...
# or
export BROWSER_USE_API_KEY=...
```

1) Download PDFs:

```bash
python -m experiments.download_pdfs_from_collection \
  --collection-xlsx "Paper collection.xlsx" \
  --pdf-root "data/pdfs" \
  --sheets all
```

2) Run factorized scoring:

```bash
python -m experiments.factorized_collection_cli \
  --collection-xlsx "Paper collection.xlsx" \
  --pdf-root "data/pdfs" \
  --out-root "runs/factorized_001" \
  --k-dags 5 \
  --m-node 3 \
  --max-nodes 12
```

Small smoke run (recommended first to build cache quickly):

```bash
python -m experiments.factorized_collection_cli \
  --collection-xlsx "Paper collection.xlsx" \
  --pdf-root "data/pdfs" \
  --out-root "runs/factorized_smoke" \
  --k-dags 2 \
  --m-node 2 \
  --max-nodes 10
```

Outputs:

- `runs/factorized_001/papers_summary.csv` — one row per paper (status + global score stats)
- Per-paper artifact folders with:
  - `extracted_text.txt`
  - `dag/dag_kXXX.json` + `dag/dag_kXXX_validation.json`
  - `node_scores/dag_kXXX/node_scores_mYYY.json`
  - `graph_scores.csv`
  - `summary.json`
  - `record.json`

## Offline ablation studies from cached runs

The ablation scripts run fully offline from cached factorized artifacts (no new
LLM calls):

```bash
python -m experiments.ablation_studies_cli \
  --runs-root runs/factorized_001 \
  --out-root runs/ablation_studies \
  --studyIDs 1,2,3,4,5,6 \
  --reuse-cache
```

Inspect available studies/variants:

```bash
python -m experiments.ablation_studies_cli --list-studies
```

## Hyperparameter tuning study (cache-first)

This repo uses a cache-first ablation study to tune scorer hyperparameters
without making new LLM calls. We run studies over cached factorized outputs and
compare variants using `study_variant_summary.csv` for each study.

Tuning command used:

```bash
python -m experiments.ablation_studies_cli \
  --runs-root runs/factorized_collection \
  --out-root runs/ablation_studies \
  --studyIDs 3,4,5 \
  --reuse-cache \
  --verbose
```

Study coverage:

- Study 3 (`edge_feature_ablation`): edge feature weights and priors
- Study 4 (`propagation_ablation`): trust propagation settings (`agg`, `alpha`, `eta`)
- Study 5 (`graph_component_ablation`): graph-head component weights

Selected default settings (kept in code after tuning):

- `EdgeCombineWeights` in `graph_app/kg_realtime_scoring.py`:
  `role_prior=0.30`, `parent_quality=0.20`, `child_quality=0.20`,
  `alignment=0.10`, `synergy=0.20`
- `PropagationPenalty`:
  `enabled=True`, `agg="min"`, `alpha=1.0`, `eta=2**(-1/8)`,
  `softmin_beta=6.0`, `dampmin_lambda=0.35`
- `GraphScoreWeights`:
  `bridge_coverage=0.25`, `best_path=0.25`, `redundancy=0.15`,
  `fragility=-0.15`, `coherence=0.10`, `coverage=0.10`

Outputs to inspect:

- `runs/ablation_studies/study_03_edge_feature_ablation/study_variant_summary.csv`
- `runs/ablation_studies/study_04_propagation_ablation/study_variant_summary.csv`
- `runs/ablation_studies/study_05_graph_component_ablation/study_variant_summary.csv`
