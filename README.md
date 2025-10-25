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
npm run dev
```

This will start the development server at `http://localhost:8000`

### Build

```bash
npm run build
```

### Serve Production Build

```bash
npm run serve
```

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
