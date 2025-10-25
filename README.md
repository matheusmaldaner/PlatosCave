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
