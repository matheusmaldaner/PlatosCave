#!/usr/bin/env python3
"""
Batch CLI for Plato's Cave paper analysis.

This tool runs multiple analyses on papers to test scoring consistency
and measure determinism across runs.

Usage (from project root):
    # Single paper, multiple runs
    python experiments/batch_cli.py --pdf backend/papers/paper.pdf --runs 10

    # URL mode
    python experiments/batch_cli.py --url "https://arxiv.org/..." --runs 5

    # Multiple papers from file
    python experiments/batch_cli.py --papers backend/papers.txt --runs 10

    # With custom settings and logging
    python experiments/batch_cli.py --pdf paper.pdf --runs 10 \\
        --log-file experiment.log --verbose
"""

import argparse
import asyncio
import csv
import io
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, TextIO
import hashlib
import requests


class TeeWriter:
    """Write to multiple streams simultaneously (for logging to file + console)."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(log_file: Optional[str]) -> Optional[TextIO]:
    """
    Set up logging to capture all stderr output to a file.

    Returns the log file handle (caller should close it when done).
    """
    if not log_file:
        return None

    # Create log file
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Open log file
    log_handle = open(log_path, 'w', encoding='utf-8')

    # Write header
    log_handle.write(f"Plato's Cave Batch Analysis Log\n")
    log_handle.write(f"Started: {datetime.now().isoformat()}\n")
    log_handle.write(f"{'='*60}\n\n")
    log_handle.flush()

    # Redirect stderr to both console and file
    original_stderr = sys.stderr
    sys.stderr = TeeWriter(original_stderr, log_handle)

    print(f"Logging to: {log_path}", file=sys.stderr)

    return log_handle

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env from backend directory (if present)
from dotenv import load_dotenv
load_dotenv(project_root / 'backend' / '.env')

# NOTE: The legacy analyze_paper / AnalysisResult API is no longer present.
# Batch experiments now use the factorized offline pipeline.
from backend.factorized_experiment import run_factorized_resampling_for_pdf
from backend.llm_client import LLMConfig


@dataclass
class RunResult:
    """Result of a single analysis run."""
    run_number: int
    success: bool
    integrity_score: float
    graph_details: Optional[Dict[str, float]] = None
    artifacts_dir: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperSummary:
    """Summary of all runs for a single paper."""
    paper_id: str
    paper_source: str  # PDF path or URL
    num_runs: int
    successful_runs: int
    scores: List[float] = field(default_factory=list)

    # Statistics
    mean_score: float = 0.0
    std_dev: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    range_score: float = 0.0

    # Per-run details
    runs: List[RunResult] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['runs'] = [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.runs]
        return result


def calculate_statistics(scores: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of scores."""
    if not scores:
        return {
            "mean": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0
        }

    return {
        "mean": statistics.mean(scores),
        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "range": max(scores) - min(scores)
    }


def generate_paper_id(source: str) -> str:
    """Generate a short, filesystem-safe ID for a paper."""
    # Extract filename without extension for PDFs
    if source.endswith('.pdf'):
        name = Path(source).stem
        # Clean up the name
        name = name.replace(' ', '_')[:50]
        return name

    # For URLs, use hash
    url_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    return f"url_{url_hash}"


async def run_single_analysis(
    paper_source: str,
    run_number: int,
    output_dir: Path,
    is_url: bool,
    settings: Dict[str, Any],
    verbose: bool = False
) -> RunResult:
    """Run a single analysis and return structured result."""
    run_start = time.time()
    timestamp = datetime.now().isoformat()

    if verbose:
        print(f"  [Run {run_number}] Starting analysis...", file=sys.stderr)

    try:
        # Create run-specific output directory
        run_output_dir = output_dir / f"run_{run_number}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Progress callback for verbose mode
        def progress_callback(stage: str, message: str):
            if verbose:
                print(f"    [{stage}] {message}", file=sys.stderr)

        # Resolve input PDF (preferred workflow is local PDFs; URL mode supports direct PDF URLs)
        pdf_path = paper_source
        if is_url:
            if not (paper_source.lower().startswith('http://') or paper_source.lower().startswith('https://')):
                raise ValueError(f"Not a valid URL: {paper_source}")
            if not paper_source.lower().endswith('.pdf'):
                raise ValueError(
                    "URL mode in batch_cli currently supports direct PDF URLs only. "
                    "Use experiments/download_pdfs_from_collection.py to mass-download PDFs first."
                )
            pdf_path = str(run_output_dir / "input.pdf")
            r = requests.get(paper_source, timeout=60)
            r.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(r.content)

        # Run the (offline) factorized pipeline with a single (K=1, M=1) trial.
        llm_cfg = LLMConfig(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0")),
            max_tokens=None,
        )

        summary = await run_factorized_resampling_for_pdf(
            pdf_path=str(pdf_path),
            out_dir=str(run_output_dir),
            llm_cfg=llm_cfg,
            k_dags=1,
            m_node_resamples=1,
            max_nodes=int(settings.get('max_nodes', 12)),
            exa_k=int(settings.get('exa_k', 6)),
            concurrency=int(settings.get('concurrency', 3)),
            reuse_cached=False,
        )

        # One trial => mean is the run score.
        score = float(summary.get("global", {}).get("mean", 0.0))
        n_success = int(summary.get("global", {}).get("n_success", 0))

        duration = time.time() - run_start

        if verbose:
            status = "SUCCESS" if n_success > 0 else "FAILED"
            print(f"  [Run {run_number}] {status} - Score: {score:.3f} ({duration:.1f}s)", file=sys.stderr)

        return RunResult(
            run_number=run_number,
            success=n_success > 0,
            integrity_score=score,
            graph_details=None,
            artifacts_dir=str(run_output_dir),
            duration_seconds=duration,
            error=None if n_success > 0 else "No successful score produced (see run directory artifacts)",
            timestamp=timestamp
        )

    except Exception as e:
        duration = time.time() - run_start
        error_msg = str(e)

        if verbose:
            print(f"  [Run {run_number}] ERROR: {error_msg} ({duration:.1f}s)", file=sys.stderr)

        return RunResult(
            run_number=run_number,
            success=False,
            integrity_score=0.0,
            graph_details=None,
            artifacts_dir=None,
            duration_seconds=duration,
            error=error_msg,
            timestamp=timestamp
        )


async def run_batch_for_paper(
    paper_source: str,
    paper_id: str,
    num_runs: int,
    output_dir: Path,
    is_url: bool,
    settings: Dict[str, Any],
    delay: float = 5.0,
    verbose: bool = False
) -> PaperSummary:
    """Run multiple analyses for a single paper."""
    paper_output_dir = output_dir / paper_id
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing: {paper_source}", file=sys.stderr)
    print(f"Output directory: {paper_output_dir}", file=sys.stderr)
    print(f"Running {num_runs} analysis runs...", file=sys.stderr)

    runs: List[RunResult] = []
    scores: List[float] = []

    for run_num in range(1, num_runs + 1):
        # Run analysis
        run_result = await run_single_analysis(
            paper_source=paper_source,
            run_number=run_num,
            output_dir=paper_output_dir,
            is_url=is_url,
            settings=settings,
            verbose=verbose
        )
        runs.append(run_result)

        # Collect successful scores
        if run_result.success:
            scores.append(run_result.integrity_score)

        # Save incremental results after each run
        save_incremental_results(paper_output_dir, runs, paper_source, settings)

        # Delay between runs (except after last run)
        if run_num < num_runs and delay > 0:
            if verbose:
                print(f"  Waiting {delay}s before next run...", file=sys.stderr)
            await asyncio.sleep(delay)

    # Calculate statistics
    stats = calculate_statistics(scores)

    # Build summary
    summary = PaperSummary(
        paper_id=paper_id,
        paper_source=paper_source,
        num_runs=num_runs,
        successful_runs=len(scores),
        scores=scores,
        mean_score=stats['mean'],
        std_dev=stats['std_dev'],
        min_score=stats['min'],
        max_score=stats['max'],
        range_score=stats['range'],
        runs=runs,
        timestamp=datetime.now().isoformat(),
        settings=settings
    )

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Summary for: {paper_id}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Successful runs: {summary.successful_runs}/{summary.num_runs}", file=sys.stderr)
    if scores:
        print(f"  Mean score:      {summary.mean_score:.4f}", file=sys.stderr)
        print(f"  Std deviation:   {summary.std_dev:.4f}", file=sys.stderr)
        print(f"  Min score:       {summary.min_score:.4f}", file=sys.stderr)
        print(f"  Max score:       {summary.max_score:.4f}", file=sys.stderr)
        print(f"  Range:           {summary.range_score:.4f}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    return summary


def save_incremental_results(output_dir: Path, runs: List[RunResult], source: str, settings: Dict):
    """Save incremental results after each run (for crash recovery)."""
    incremental_file = output_dir / '_incremental_results.json'
    data = {
        'paper_source': source,
        'settings': settings,
        'runs_completed': len(runs),
        'runs': [r.to_dict() for r in runs],
        'last_updated': datetime.now().isoformat()
    }
    with open(incremental_file, 'w') as f:
        json.dump(data, f, indent=2)


def save_summary_json(summary: PaperSummary, output_path: Path):
    """Save summary as JSON."""
    with open(output_path, 'w') as f:
        json.dump(summary.to_dict(), f, indent=2)
    print(f"Saved JSON summary: {output_path}", file=sys.stderr)


def save_summary_csv(summary: PaperSummary, output_path: Path):
    """Save summary as CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'run_number', 'success', 'integrity_score', 'duration_seconds',
            'error', 'timestamp', 'artifacts_dir'
        ])

        # Per-run data
        for run in summary.runs:
            writer.writerow([
                run.run_number,
                run.success,
                run.integrity_score,
                run.duration_seconds,
                run.error or '',
                run.timestamp,
                run.artifacts_dir or ''
            ])

        # Blank row
        writer.writerow([])

        # Summary statistics
        writer.writerow(['Statistics', 'Value'])
        writer.writerow(['paper_id', summary.paper_id])
        writer.writerow(['paper_source', summary.paper_source])
        writer.writerow(['num_runs', summary.num_runs])
        writer.writerow(['successful_runs', summary.successful_runs])
        writer.writerow(['mean_score', summary.mean_score])
        writer.writerow(['std_dev', summary.std_dev])
        writer.writerow(['min_score', summary.min_score])
        writer.writerow(['max_score', summary.max_score])
        writer.writerow(['range_score', summary.range_score])

    print(f"Saved CSV summary: {output_path}", file=sys.stderr)


def load_papers_from_file(papers_file: str) -> List[str]:
    """Load list of papers from a file (one per line)."""
    papers = []
    with open(papers_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                papers.append(line)
    return papers


async def run_batch(args):
    """Main batch processing function."""
    # Determine input sources
    papers_to_analyze = []

    if args.pdf:
        papers_to_analyze.append(('pdf', args.pdf))
    elif args.url:
        papers_to_analyze.append(('url', args.url))
    elif args.papers:
        # Load from file
        paper_list = load_papers_from_file(args.papers)
        for paper in paper_list:
            # Detect if URL or PDF
            if paper.startswith('http://') or paper.startswith('https://'):
                papers_to_analyze.append(('url', paper))
            else:
                papers_to_analyze.append(('pdf', paper))

    if not papers_to_analyze:
        print("Error: No papers to analyze", file=sys.stderr)
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build settings dict
    settings = {
        # Legacy knobs kept for backwards compatibility with logs/metadata
        'agent_aggressiveness': args.agent_aggressiveness,
        'evidence_threshold': args.evidence_threshold,
        # Offline pipeline knobs (used)
        'max_nodes': int(os.environ.get('MAX_NODES', '12')),
        'exa_k': int(os.environ.get('EXA_K', '6')),
        'retrieval_mode': str(os.environ.get('RETRIEVAL_MODE', 'llm')),
        'concurrency': int(os.environ.get('SCORING_CONCURRENCY', '3')),
        # Batch bookkeeping
        'runs_per_paper': args.runs,
        'delay_between_runs': args.delay,
    }

    print(f"\nPlato's Cave Batch Analysis", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Papers to analyze: {len(papers_to_analyze)}", file=sys.stderr)
    print(f"Runs per paper:    {args.runs}", file=sys.stderr)
    print(f"Output directory:  {output_dir}", file=sys.stderr)
    print(f"Settings:          aggressiveness={args.agent_aggressiveness}, threshold={args.evidence_threshold}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Process each paper
    all_summaries: List[PaperSummary] = []

    for paper_type, paper_source in papers_to_analyze:
        is_url = paper_type == 'url'
        paper_id = generate_paper_id(paper_source)

        summary = await run_batch_for_paper(
            paper_source=paper_source,
            paper_id=paper_id,
            num_runs=args.runs,
            output_dir=output_dir,
            is_url=is_url,
            settings=settings,
            delay=args.delay,
            verbose=args.verbose
        )

        all_summaries.append(summary)

        # Save individual paper summary
        paper_output_dir = output_dir / paper_id
        save_summary_json(summary, paper_output_dir / 'summary.json')
        save_summary_csv(summary, paper_output_dir / 'summary.csv')

    # Save combined summary if multiple papers
    if len(all_summaries) > 1:
        combined_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_papers': len(all_summaries),
            'settings': settings,
            'papers': [s.to_dict() for s in all_summaries]
        }
        with open(output_dir / 'combined_summary.json', 'w') as f:
            json.dump(combined_summary, f, indent=2)
        print(f"\nSaved combined summary: {output_dir / 'combined_summary.json'}", file=sys.stderr)

    print(f"\nBatch analysis complete!", file=sys.stderr)
    print(f"Results saved to: {output_dir}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Batch testing CLI for Plato's Cave paper analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from project root):
  # Single PDF, 10 runs
  python experiments/batch_cli.py --pdf backend/papers/paper.pdf --runs 10

  # URL analysis, 5 runs with logging
  python experiments/batch_cli.py --url "https://arxiv.org/..." --runs 5 --log-file run.log

  # Multiple papers from file
  python experiments/batch_cli.py --papers backend/papers.txt --runs 10

  # With custom settings, verbose output, and logging
  python experiments/batch_cli.py --pdf paper.pdf --runs 3 --verbose \\
      --log-file experiment.log --agent-aggressiveness 7
        """
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdf", type=str,
                             help="Path to PDF file to analyze")
    input_group.add_argument("--url", type=str,
                             help="URL to analyze")
    input_group.add_argument("--papers", type=str,
                             help="Path to file with list of papers (one per line)")

    # Batch settings
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs per paper (default: 10)")
    parser.add_argument("--output", type=str, default="./batch_results",
                        help="Output directory (default: ./batch_results)")

    # Analysis settings
    parser.add_argument("--agent-aggressiveness", type=int, default=5,
                        help="Agent aggressiveness 1-10 (default: 5)")
    parser.add_argument("--evidence-threshold", type=float, default=0.8,
                        help="Evidence threshold 0.0-1.0 (default: 0.8)")

    # Execution settings
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Delay between runs in seconds (default: 5.0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress to stderr")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Save all output to a log file (e.g., --log-file run.log)")

    args = parser.parse_args()

    # Setup logging if requested
    log_handle = setup_logging(args.log_file)

    try:
        # Run the batch processing
        asyncio.run(run_batch(args))
    finally:
        # Close log file if opened
        if log_handle:
            print(f"\nLog saved to: {args.log_file}", file=sys.stderr)
            log_handle.close()


if __name__ == "__main__":
    main()
