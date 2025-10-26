<div align="center">
<img src="docs/img/banner.png" alt="Plato's Cave Logo">


_Appeal to Logos_

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-prototype-yellow)]()
![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-orange)

---

 *A learning-focused research tool that helps you illuminate complex academic papers.*

</div>

---

### 🧠 Overview

Plato’s Cave helps you **comprehend dense academic material** by analyzing PDFs and URLs using progressive AI assistance.  
Like emerging from Plato’s allegorical cave into enlightenment, this tool illuminates the **shadows of academic literature**, uncovering key insights, visualizations, and summaries.  

![Green Cave](https://github.com/matheusmaldaner/PlatosCave/blob/main/green_cave.gif)

## 🚀 Quick Start

You have to run the frontend, backend and docker image on **three** separate terminal sessions. Follow the commands below:

### (1) 🪶 Frontend Setup

```bash
cd frontend
npm install
gatsby develop
```

This will start the development server at `http://localhost:8000`

### (2) 🖥️ Backend Setup

On a separate terminal, while the frontend is still running, run the following commands:

```bash
# install uv 
# curl -LsSf https://astral.sh/uv/install.sh | sh
cd backend
uv sync
source .venv/bin/activate
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

python server.py
```

### (3) 🐳 Docker Image for Web Agent

Also an "image" here is not to be interpreted as a picture. You can think of a Docker image as a package that has everything you need to run some software (it has the libraries, code, envs, configs...)

```bash 
# start the docker (open Docker Desktop if on Windows -> Resources -> WSL Integration -> Enable Integration)
docker compose -f docker-compose.browser.yaml up --build remote-browser
# alternatively you can run it detached:
# docker compose -f docker-compose.browser.yaml up -d
```


## 🧩 Tech Stack


| Component         | Technology            |
| ----------------- | --------------------- |
| 🖼️ Frontend      | Gatsby.js             |
| 🧠 Backend        | Python (FastAPI)      |
| 🌐 Web Automation | Playwright + Chromium |
| 🤖 AI Integration | Browser-Use API       |
| 🐳 Environment    | Docker                |



<hr style="border: 0.5px solid #e5e5e5; margin: 20px 0;">

## 🔢 Scoring Mathematics Located [Here](https://github.com/matheusmaldaner/PlatosCave/blob/main/graph_app/README.md)
