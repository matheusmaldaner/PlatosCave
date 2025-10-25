# Research Paper Summarizer

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



# Logos

# Agent Logic
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

Which ones we build out will depend on target paper types we use for the demo
