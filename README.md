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

## Getting Started

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

## Features

- ✅ Clean ChatGPT-like landing page
- ✅ URL input support
- ✅ PDF file upload (drag-and-drop + file picker)
- ✅ PDF-only validation
- ✅ Responsive design
- ✅ Console logging for debugging

## Project Structure

```
research_summarizer/
├── src/
│   ├── components/
│   │   └── FileUploader.tsx    # Main input component
│   ├── pages/
│   │   └── index.tsx            # Landing page
│   ├── styles/
│   │   └── global.css           # Global styles
│   └── images/
│       └── icon.svg             # App icon
├── gatsby-config.ts             # Gatsby configuration
├── gatsby-browser.tsx           # Browser APIs
├── tailwind.config.js           # Tailwind configuration
├── tsconfig.json                # TypeScript configuration
└── package.json                 # Dependencies
```

## Next Steps

1. Add backend API integration
2. Implement PDF processing
3. Add URL content fetching
4. Display analysis results
5. Add loading states

## Self-Contained

This project is completely self-contained with its own:
- Dependencies (package.json)
- Configuration files
- Source code
- No external dependencies from parent directories
