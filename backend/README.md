# Plato’s Cave — Backend

This repository contains the backend implementation for **Plato’s Cave**, a system for scientific paper verification using **trust-propagating knowledge graphs** and **multi-agent external verification**.

The backend is responsible for:
- Extracting structured claims from scientific papers
- Constructing validated directed acyclic graphs (DAGs)
- Verifying claims via external sources
- Computing node-, edge-, and graph-level quality scores
- Producing interpretable metrics for downstream analysis and evaluation

---

## System Overview

The backend implements the full Plato’s Cave pipeline:

### 1. Paper Ingestion
- Accepts either a direct paper URL or a natural-language query
- Locates the most relevant paper (arXiv, publisher PDF, or academic site)
- Extracts core textual content (title, abstract, claims, methods, results)

### 2. DAG Extraction
- Uses a large language model to extract a **directed acyclic graph**
- Nodes correspond to semantic roles (e.g., Hypothesis, Evidence, Method, Result)
- Edges encode logical dependency relationships
- Graph is validated for acyclicity and role-transition constraints

### 3. Multi-Agent Verification
- Independent verification agents evaluate individual nodes
- Agents search external sources (papers, datasets, code repositories, replications)
- Each node is assigned a quality score using role-specific metrics

### 4. Trust Propagation and Scoring
- Node quality is propagated through the graph using a **trust-gating mechanism**
- Weak upstream evidence suppresses downstream confidence
- Graph-level quality scores are computed from node and edge scores

### 5. Evaluation and Analysis
- Outputs node-level, edge-level, and paper-level metrics
- Supports ablation studies, correlation analysis, and case studies

---
