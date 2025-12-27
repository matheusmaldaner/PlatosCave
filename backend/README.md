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
- Accepts a direct paper URL, a natural-language query, or a PDF file  
- For URLs or natural-language queries, a web retrieval agent (optionally backed by Exa) locates the most relevant paper source (e.g., arXiv, publisher PDF, or academic site)  
- Extracts core textual content, including **title, abstract, claims, methods, and results**

### 2. DAG Extraction
- Uses a large language model to extract a **directed acyclic graph (DAG)** from the paper text  
- Nodes correspond to semantic roles (e.g., Hypothesis, Evidence, Method, Result)  
- Edges encode logical dependency relationships between claims  
- The extracted graph is validated for acyclicity and role-transition constraints  

### 3. Multi-Agent Verification
- Independent verification agents evaluate individual nodes in the graph  
- Agents search external sources such as related papers, datasets, code repositories, and replication studies  
- Each node is assigned a quality score using role-specific evaluation metrics  

### 4. Trust Propagation and Scoring
- Node quality scores are propagated through the graph using a **trust-gating mechanism**  
- Weak upstream evidence suppresses confidence in downstream claims  
- Graph-level quality scores are computed from aggregated node and edge scores  

### 5. Evaluation and Analysis
- Produces node-level, edge-level, and paper-level metrics  
- Supports ablation studies, correlation analysis, and qualitative case studies  
---
