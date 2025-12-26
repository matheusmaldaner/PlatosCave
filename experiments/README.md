# Experiments

Batch testing for Plato's Cave paper analysis.

## Quick Start

```bash
# From project root

# Single paper, 10 runs
python batch_cli.py --pdf paper.pdf --runs 10

# Quick test (3 runs, verbose)
python batch_cli.py --pdf paper.pdf --runs 3 --verbose

# With logging
python batch_cli.py --pdf paper.pdf --runs 10 \
    --verbose --log-file experiments/run.log

# Multiple papers
python batch_cli.py --papers papers.txt --runs 1
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | - | Path to PDF file |
| `--url` | - | URL to analyze |
| `--papers` | - | File with list of papers |
| `--runs` | 10 | Runs per paper |
| `--output` | `./batch_results` | Output directory |
| `--delay` | 5.0 | Seconds between runs |
| `--verbose` | off | Show progress |
| `--log-file` | - | Save output to file |
| `--agent-aggressiveness` | 5 | 1-10 |
| `--evidence-threshold` | 0.8 | 0.0-1.0 |

## Output

Results are saved to `--output` directory:
- `summary.json` - Full results with statistics
- `summary.csv` - Spreadsheet-friendly format
- `run_N/` - Artifacts from each run
