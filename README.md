<div align="center">
<img src="docs/img/banner.png" alt="Plato's Cave Logo">


_Escape the Cave of Complexity_

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-prototype-yellow)]()
![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-orange)
</div>

---

Plato's Cave is a **learning-focused research tool** that helps you understand complex academic papers by analyzing PDFs and URLs. 

Like emerging from Plato's allegorical cave into enlightenment, this tool illuminates the dense shadows of academic literature, extracting key insights, visualizations, and summaries through progressive AI assistance.

## 🚀 Quick Start

A simple, self-contained web application for analyzing research papers from URLs or PDF files.

## Tech Stack

- **Gatsby** - Static site generator
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

## Getting Started (FrontEnd)

### Installation

```bash
npm install
```

### Development

```bash
gatsby develop
```

This will start the development server at `http://localhost:8000`


## Getting Started (Backend)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install browser-use
uvx playwright install chromium --with-deps
```

Create a .env file
```
touch .env
```

Add your API key to the file (get $10 free [here](https://cloud.browser-use.com/dashboard/api))
```
BROWSER_USE_API_KEY="bu_YOURKEY"
```

```bash
# install VNC server
sudo apt-get install x11vnc

# start VNC server
x11vnc -display :0 -forever -shared

sudo apt update
sudo apt install chromium-browser
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

cd backend
python server.py
```

### Agent Logic
Make an acyliclical graph of the paper in the following structure:
{
  text:
  next_node:
}

Prompt One:

You are verifying ONE statement from a scholarly work in an unknown field.
Judge only using internal logic, standard disciplinary conventions, and what a well-read scholar would know WITHOUT the web.
Do not fabricate specifics not present in the statement or obvious from general academic practice.

STATEMENT:
<<<{text}>>>

TASKS (do all):
1) Classify the statement into ONE category:
   {definition | theoretical-claim | methodological-claim | empirical-result | data-statement | model/equation | causal-claim | statistical-inference | citation/attribution | quotation | normative-claim | limitation/scope | conclusion/implication | forecast/prediction}
2) INTERNAL correctness (0–1): Does it make sense on its own terms (logic, math if any, terminology, no category errors)?
3) CONTEXTUAL plausibility (0–1): Is it broadly plausible for a typical paper in its field (without checking sources)?
4) SUPPORT present in-text (select all): {figure/table ref | equation ref | cited-source ref | definition given | none}
5) Red flags (0–3 short items): e.g., undefined term, missing unit/timeframe, category error, overclaim, non sequitur, math mismatch.
6) Rationale (1–3 sentences) citing specific phrases or structural issues in the statement.

Return STRICT JSON ONLY:
{
  "classification": "",
  "internal_score": 0.0,
  "contextual_plausibility": 0.0,
  "support_markers": [],
  "red_flags": [],
  "rationale": ""
}



Potential Agents For Improved Fact Checking: 
* equation-verifier agent	symbolic math, gradient comparison, consistency checks.
* replication agent	re-runs experiments and compares metrics.
* citation agent	fetches and summarizes cited works, flags mis-citations.
* explanation-tester agent	runs stability and faithfulness experiments.
* consistency-auditor agent	scans text for overclaims or unsupported assertions.
