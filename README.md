# Plato's Cave

<div align="center">
<img src="docs/img/banner.png" alt="Plato's Cave Logo">

*A Human-Centered Agentic System for Validating Research Papers*

[![1st Place](https://img.shields.io/badge/1st%20Place-UF%20AI%20Days%20GatorHack-yellow)](https://www.hackathonparty.com/hackathons/26/projects/355)
[![License](https://img.shields.io/badge/license-Research%20Use%20Only-red.svg)](LICENSE.md)
[![Commercial License](https://img.shields.io/badge/commercial%20license-contact%20author-blue.svg)](mailto:mkunzlermaldaner@ufl.edu)
![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-orange)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Demo Video](#demo-video)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Frontend Setup](#frontend-setup)
  - [Backend Setup](#backend-setup)
  - [Docker Setup for Web Agent](#docker-setup-for-web-agent)
- [Project Structure](#project-structure)
- [Frontend Components](#frontend-components)
- [API Endpoints](#api-endpoints)
- [Scoring Mathematics](#scoring-mathematics)
- [Configuration](#configuration)
- [License](#license)
- [Citation](#citation)

---

## Overview

Plato's Cave is an AI-powered research paper analysis and integrity verification system. Like emerging from Plato's allegorical cave into enlightenment, this tool illuminates the shadows of academic literature by:

- Decomposing research papers into structured knowledge graphs
- Evaluating citations, methodology, and reproducibility
- Providing real-time integrity scoring with transparent metrics
- Enabling web-based verification of claims through autonomous browsing

<div align="center">

![Green Cave](https://github.com/matheusmaldaner/PlatosCave/blob/main/green_cave.gif)

</div>

---

## Demo Video

We developed Plato's Cave as part of the University of Florida's AI Days Hackathon:

<div align="center">
  <a href="https://www.youtube.com/watch?v=wvmJdUhuj4s" target="_blank">
    <img src="https://img.youtube.com/vi/wvmJdUhuj4s/maxresdefault.jpg"
      alt="Watch the demo video"
      width="600"
      style="border-radius: 8px;"
    />
  </a>
</div>

---

## Features

- **PDF Analysis**: Upload research papers in PDF format for comprehensive analysis
- **URL Analysis**: Analyze papers directly from URLs (ArXiv, journals, etc.)
- **Knowledge Graph Visualization**: Interactive graph showing paper structure and relationships
- **Real-time Progress Tracking**: WebSocket-based updates during analysis
- **Integrity Scoring**: Transparent scoring based on citations, methodology, reproducibility
- **Remote Browser Viewing**: Watch the AI agent browse and verify claims in real-time
- **Configurable Analysis Parameters**: Adjust agent aggressiveness and evidence thresholds

---

## Architecture

The system consists of three main components:

1. **Frontend**: React-based (Gatsby) single-page application with Tailwind CSS styling
2. **Backend**: Python Flask server with Socket.IO for real-time communication
3. **Browser Service**: Docker-containerized Chromium instance for web automation

```
                                    +------------------+
                                    |   Remote Browser |
                                    |   (Docker/CDP)   |
                                    +--------+---------+
                                             |
+------------------+    WebSocket    +-------+--------+
|    Frontend      | <-------------> |    Backend     |
|  (Gatsby/React)  |    REST API     |  (Flask/SocketIO)|
|   Port 8000      | <-------------> |    Port 5000    |
+------------------+                 +----------------+
```

---


## Quick Start

You need to run the frontend, backend, and Docker container in three separate terminal sessions.

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+
- Docker and Docker Compose
- uv package manager (installed automatically via setup)

### Frontend Setup

```bash
cd frontend
npm install
npm run develop
```

The development server will start at `http://localhost:8000`

If you encounter module resolution errors:
```bash
npm install lucide-react react-dropzone
```

### Backend Setup

#### Recommended: Using Task Runner

If you have [Task](https://taskfile.dev/) installed, this is the easiest way to get started:

```bash
# Run from project root - handles uv install, dependencies, and .env setup
task setup

# Start the backend server
task start-server
```

After running `task setup`, edit the `.env` file to add your API key:
```
BROWSER_USE_API_KEY=bu_YOURKEY
```

You can get a free API key at [browser-use.com](https://cloud.browser-use.com/#settings/api-keys/new).

Additional task commands:
```bash
task clean          # Clean up virtual environment
task setup-vnc      # Install VNC server (Linux only)
```

#### Manual Setup (Alternative)

If you don't have Task installed, follow these steps:

```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to backend and sync dependencies
cd backend
uv sync
source .venv/bin/activate

# Install Playwright browser
uvx playwright install chromium --with-deps
```

Create a `.env` file in the backend directory:

```bash
cp ../sample.env .env
```

Edit `.env` and add your API key:
```
BROWSER_USE_API_KEY="bu_YOURKEY"
```

Start the server:
```bash
python server.py
```

The backend will run on `http://localhost:5000`

### Docker Setup (Backend + Browser)

Run the backend and remote browser using Docker Compose:

```bash
# From project root
cp sample.env .env
# Edit .env with your API keys

# Start backend services
docker compose up -d
```

This starts:
- Backend API on `http://localhost:5001`
- Remote browser CDP on `http://localhost:9222`
- Browser VNC viewer on `http://localhost:7900`

Useful commands:
```bash
docker compose logs -f      # View logs
docker compose down         # Stop services
docker compose up --build   # Rebuild after changes
```
---

## Project Structure

```
PlatosCave/
├── frontend/                    # Gatsby/React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── landing/         # Landing page components
│   │   │   │   ├── Navbar.tsx   # Navigation bar
│   │   │   │   ├── Hero.tsx     # Hero section with search
│   │   │   │   ├── SearchBar.tsx# Search/upload input
│   │   │   │   ├── PaperCard.tsx# Sample paper card
│   │   │   │   └── Footer.tsx   # Page footer
│   │   │   ├── BrowserViewer.tsx
│   │   │   ├── FileUploader.tsx
│   │   │   ├── ProgressBar.tsx
│   │   │   ├── SettingsDrawer.tsx
│   │   │   └── XmlGraphViewer.tsx
│   │   ├── pages/
│   │   │   └── index.tsx        # Main page component
│   │   └── styles/
│   │       └── global.css       # Global styles and animations
│   ├── tailwind.config.js       # Tailwind configuration
│   └── package.json
├── backend/
│   ├── server.py                # Flask server with Socket.IO
│   ├── main.py                  # Analysis pipeline
│   ├── prompts.py               # LLM prompt templates
│   ├── verification_pipeline.py # Claim verification logic
│   ├── docker-compose.browser.yaml
│   └── pyproject.toml
├── graph_app/
│   ├── kg_realtime_scoring.py   # Knowledge graph scoring
│   ├── service_adapter.py       # API for graph operations
│   └── README.md                # Scoring mathematics
├── docs/
│   ├── img/                     # Documentation images
│   └── CLOUDFLARE_DEPLOYMENT.md # Deployment guide
├── docker-compose.yml           # Docker services (backend + browser)
├── taskfile.yml                 # Task runner configuration
├── sample.env                   # Environment template
└── README.md
```

---

## Frontend Components

The landing page follows a modern SaaS design pattern with the following components:

### Navbar
- Sticky navigation bar with logo
- Navigation links: How it works, Models, Pricing, About
- "Get Started" call-to-action button

### Hero Section
- Two-column layout (text left, showcase card right)
- Large serif headline with supporting text
- Primary and secondary action buttons
- Trust indicators with institution badges

### SearchBar
- Pill-shaped input with typewriter animation
- PDF upload via drag-and-drop or file picker
- URL/DOI input support
- Play button to start analysis

### PaperCard
- Sample paper showcase with integrity badge
- Verification checklist display
- Floating alert notification
- Action button for full analysis

### Analysis View
- Progress bar with stage indicators
- Interactive knowledge graph visualization
- Settings drawer for parameter adjustment
- Real-time integrity score display

---

## API Endpoints

### REST Endpoints

| Method | Endpoint          | Description                    |
| ------ | ----------------- | ------------------------------ |
| POST   | `/api/upload`     | Upload PDF for analysis        |
| POST   | `/api/analyze-url`| Analyze paper from URL         |
| POST   | `/api/cleanup`    | Reset browser session          |

### WebSocket Events

| Event           | Direction | Description                    |
| --------------- | --------- | ------------------------------ |
| `connect`       | Client    | Client connects to server      |
| `disconnect`    | Client    | Client disconnects             |
| `status_update` | Server    | Progress/data updates          |

### WebSocket Message Types

```typescript
// Progress update
{ type: "UPDATE", stage: string, text: string }

// Knowledge graph data
{ type: "GRAPH_DATA", data: string }

// Analysis complete
{ type: "DONE", score: number }

// Browser address info
{ type: "BROWSER_ADDRESS", novnc_url: string, cdp_url: string }

// Error notification
{ type: "ERROR", message: string }
```

---

## Scoring Mathematics

The integrity scoring system uses a trust-propagating knowledge graph with:

- **Node Quality**: Convex blend of six metrics (credibility, relevance, evidence strength, method rigor, reproducibility, citation support)
- **Edge Confidence**: Role-aware transitions with trust gating from parent nodes
- **Graph Score**: Composite of path reliability, coverage, coherence, redundancy, and fragility

For detailed mathematical formulations, see [graph_app/README.md](./graph_app/README.md).

---

## Configuration

### Environment Variables

| Variable                       | Description                              |
| ------------------------------ | ---------------------------------------- |
| `BROWSER_USE_API_KEY`          | Browser-Use API key for web automation   |
| `REMOTE_BROWSER_CDP_URL`       | Chrome DevTools Protocol URL             |
| `REMOTE_BROWSER_NOVNC_URL`     | VNC viewer URL for browser               |

### Analysis Settings

Settings can be adjusted via the frontend drawer:

- **Agent Aggressiveness**: Controls depth of verification (1-10)
- **Evidence Threshold**: Minimum confidence for accepting evidence (0-1)
- **Metric Weights**: Importance of each scoring dimension

---

## License

Copyright 2025 Matheus Kunzler Maldaner. All Rights Reserved.

This project is licensed under the **Plato's Cave Research and Academic Use License**.

See [LICENSE.md](./LICENSE.md) for complete terms and conditions.

For commercial licensing inquiries, contact: mkunzlermaldaner@ufl.edu

---

## Citation

If you use this software in academic research, please cite:

```bibtex
@software{maldaner2025platoscave,
  author = {Maldaner, Matheus Kunzler and Valle, Raul and Wormald, Stephen and O'Connor, Kristian and Woelke, James},
  title = {Plato's Cave: A Trust-Propagating Knowledge Graph System for Academic Research Verification},
  year = {2025},
  institution = {University of Florida},
  note = {Winner: 1st Place, UF AI Days GatorHack},
  url = {https://github.com/matheusmaldaner/PlatosCave}
}
```
