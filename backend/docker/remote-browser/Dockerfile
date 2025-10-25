# syntax=docker/dockerfile:1
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    CDP_PORT=9222 \
    VNC_PORT=5900 \
    NOVNC_PORT=7900 \
    CHROME_START_URL=about:blank

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    chromium-sandbox \
        xvfb \
        x11vnc \
        fluxbox \
        novnc \
        websockify \
    socat \
        wmctrl \
        curl \
        ca-certificates \
        fonts-liberation \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Stay as root so Chromium sandbox and X11 permissions work out of the box
WORKDIR /root

COPY start-browser.sh /usr/local/bin/start-browser.sh
RUN chmod +x /usr/local/bin/start-browser.sh

EXPOSE ${CDP_PORT} ${VNC_PORT} ${NOVNC_PORT}

ENTRYPOINT ["/usr/local/bin/start-browser.sh"]
