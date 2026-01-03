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

Outputs:

- `runs/factorized_001/summary.csv` — one row per paper (mean/std/min/max)
- `runs/factorized_001/summary_by_dag.csv` — per-DAG statistics (aggregated across node resamples)
- `runs/factorized_001/good_vs_bad_eval.json` — if the spreadsheet has `Rating` values like `Good`/`Bad`
- Per-paper artifact folders with:
  - `extracted_text.txt`
  - `dag_kXXX.json` + `dag_kXXX_validation.json`
  - `node_scores_kXXX_mYYY.json`
  - `graph_scores.csv`
  - `graph_score_kde.png`
