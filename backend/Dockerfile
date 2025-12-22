FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install chromium --with-deps

COPY backend/ ./

COPY graph_app/ ./graph_app/

RUN mkdir -p papers

EXPOSE 5001

CMD ["python", "server.py"]
