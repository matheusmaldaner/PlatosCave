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

## Exhaustive numeric grid search from cache

For true combinatorial searches over numeric scorer settings, use the dedicated
grid-search CLI. This stays fully offline and reuses cached `dag/*.json` and
`node_scores/*.json` artifacts.

Supported override key patterns:

- `metric_w.<metric>`: global node metric weights
- `node_q.<Role>.<metric>`: role-specific node quality weights
- `edge_w.<feature>`: edge combine weights
- `graph_w.<component>`: graph score weights
- `penalty.<field>`: propagation / trust-gating parameters

Example search-space JSON:

```json
{
  "name": "edge_weight_grid",
  "normalize_metric_weights": false,
  "params": {
    "edge_w.role_prior": [0.1, 0.2, 0.3],
    "edge_w.parent_quality": [0.1, 0.2, 0.3],
    "edge_w.child_quality": [0.1, 0.2, 0.3],
    "edge_w.alignment": [0.0, 0.1],
    "edge_w.synergy": [0.1, 0.2, 0.3]
  },
  "constraints": [
    {
      "type": "sum_lte",
      "params": [
        "edge_w.role_prior",
        "edge_w.parent_quality",
        "edge_w.child_quality",
        "edge_w.alignment",
        "edge_w.synergy"
      ],
      "value": 1.0
    }
  ]
}
```

Run it:

```bash
python -m experiments.grid_search_cli \
  --runs-root runs/experiments_debug4 \
  --out-root runs/grid_search \
  --search-space experiments/search_spaces/weight_grid_smoke.json \
  --max-workers 8 \
  --reuse-cache \
  --verbose
```

Key outputs:

- `config_index.csv`: every accepted configuration after constraints
- `per_paper_summary.csv`: one row per `(paper, config)`
- `config_summary.csv`: aggregated ranking metrics per configuration
- `top_configs.json`: best configurations by AUC and Spearman

Parallelism:

- The runner parallelizes over papers using worker processes.
- Each worker loads that paper's cached trials once, then scores all accepted
  configurations locally to reduce JSON reload overhead.

## Optuna search from cache

For dense search over numeric settings without enumerating every combination,
use the Optuna tuner. It samples from a defined search space and evaluates each
trial fully offline against cached `dag/*.json` and `node_scores/*.json`.

Install dependency in the experiment environment first:

```bash
.venv-exp/bin/pip install optuna
```

Search-space JSON uses Optuna-style parameter distributions:

```json
{
  "name": "optuna_edge_graph_search",
  "objective_metric": "auc_good_vs_bad",
  "direction": "maximize",
  "sampler": {
    "type": "tpe",
    "seed": 0,
    "n_startup_trials": 20,
    "multivariate": true
  },
  "pruner": {
    "type": "median",
    "n_startup_trials": 10,
    "n_warmup_steps": 5
  },
  "params": {
    "edge_w.role_prior": { "type": "float", "low": 0.0, "high": 0.4, "step": 0.05 },
    "edge_w.parent_quality": { "type": "float", "low": 0.0, "high": 0.4, "step": 0.05 },
    "edge_w.child_quality": { "type": "float", "low": 0.0, "high": 0.4, "step": 0.05 },
    "edge_w.alignment": { "type": "float", "low": 0.0, "high": 0.3, "step": 0.05 },
    "edge_w.synergy": { "type": "float", "low": 0.0, "high": 0.4, "step": 0.05 },
    "graph_w.best_path": { "type": "float", "low": 0.0, "high": 0.5, "step": 0.05 },
    "graph_w.bridge_coverage": { "type": "float", "low": 0.0, "high": 0.5, "step": 0.05 },
    "penalty.alpha": { "type": "float", "low": 0.25, "high": 2.0, "step": 0.25 },
    "penalty.eta": { "type": "float", "low": 0.5, "high": 1.0, "step": 0.05 }
  },
  "constraints": [
    {
      "type": "sum_lte",
      "params": [
        "edge_w.role_prior",
        "edge_w.parent_quality",
        "edge_w.child_quality",
        "edge_w.alignment",
        "edge_w.synergy"
      ],
      "value": 1.0
    }
  ]
}
```

Run it:

```bash
.venv-exp/bin/python -m experiments.optuna_search_cli \
  --runs-root runs/experiments_debug4 \
  --out-root runs/optuna_search \
  --search-space experiments/search_spaces/optuna_debug4_dense_v1.json \
  --n-trials 50 \
  --n-jobs 4 \
  --verbose
```

Key outputs:

- `optuna_study.sqlite3`: persistent Optuna study storage
- `trials.csv`: one row per trial with params and aggregate metrics
- `best_trial.json`: best completed trial
- `top_trials.json`: top completed trials
- `top_trial_per_paper/*.csv`: per-paper summaries for top saved trials
