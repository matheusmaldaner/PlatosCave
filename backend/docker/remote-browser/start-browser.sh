#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[remote-browser] [%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

# Configuration
INTERNAL_CDP_PORT="19222"
CHROME_HEALTH_CHECK_INTERVAL=10  # seconds between health checks
CHROME_RESTART_DELAY=3          # seconds to wait before restarting Chrome
MAX_CHROME_RESTARTS=5           # max restarts before giving up
CHROME_RESTART_COUNT=0

# Ensure runtime directories exist (fresh profile each run)
cleanup_chrome_profile() {
  log "Cleaning up Chrome profile and cache..."
  rm -rf "$HOME/chrome-profile" "$HOME/.config/chromium" "$HOME/.cache/chromium"
  mkdir -p "$HOME/chrome-profile"
}

cleanup_chrome_profile
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix
lock_file="/tmp/.X${DISPLAY#:}.lock"
rm -f "$lock_file"

# Start the virtual framebuffer
log "Starting Xvfb on display ${DISPLAY}"
Xvfb ${DISPLAY} -screen 0 1920x1080x24 -ac &
XVFB_PID=$!

# Give X time to come up
sleep 1

# Start a lightweight window manager for better rendering
log "Starting fluxbox"
fluxbox &
FLUXBOX_PID=$!

# Function to launch Chrome with all stability flags
launch_chrome() {
  log "Launching Chromium (internal CDP port ${INTERNAL_CDP_PORT})"
  chromium \
    --remote-debugging-port="${INTERNAL_CDP_PORT}" \
    --remote-allow-origins="*" \
    --user-data-dir="$HOME/chrome-profile" \
    --disable-setuid-sandbox \
    --no-sandbox \
    --disable-dev-shm-usage \
    --disable-gpu \
    --disable-software-rasterizer \
    --no-first-run \
    --no-default-browser-check \
    --disable-session-crashed-bubble \
    --disable-features=InfiniteSessionRestore,TranslateUI \
    --disable-extensions \
    --disable-background-networking \
    --disable-sync \
    --disable-default-apps \
    --disable-hang-monitor \
    --disable-prompt-on-repost \
    --disable-client-side-phishing-detection \
    --disable-component-update \
    --disable-background-timer-throttling \
    --disable-backgrounding-occluded-windows \
    --disable-renderer-backgrounding \
    --disable-ipc-flooding-protection \
    --memory-pressure-off \
    --start-maximized \
    --force-device-scale-factor=1 \
    --allow-insecure-localhost \
    --autoplay-policy=no-user-gesture-required \
    --test-type \
    --noerrdialogs \
    --incognito \
    --class=RemoteBrowser \
    --js-flags="--max-old-space-size=2048" \
    "${CHROME_START_URL}" \
    &
  CHROME_PID=$!
  log "Chrome started with PID ${CHROME_PID}"
}

# Function to check if Chrome CDP is responsive
check_chrome_health() {
  curl -fsS --max-time 5 "http://localhost:${INTERNAL_CDP_PORT}/json/version" > /dev/null 2>&1
}

# Function to restart Chrome
restart_chrome() {
  CHROME_RESTART_COUNT=$((CHROME_RESTART_COUNT + 1))
  
  if [ "$CHROME_RESTART_COUNT" -gt "$MAX_CHROME_RESTARTS" ]; then
    log "ERROR: Chrome has crashed ${MAX_CHROME_RESTARTS} times, giving up. Container will restart."
    exit 1
  fi
  
  log "Restarting Chrome (attempt ${CHROME_RESTART_COUNT}/${MAX_CHROME_RESTARTS})..."
  
  # Kill the old Chrome process if it's still running
  if [ -n "${CHROME_PID:-}" ] && kill -0 "$CHROME_PID" 2>/dev/null; then
    log "Killing stale Chrome process ${CHROME_PID}"
    kill -9 "$CHROME_PID" 2>/dev/null || true
    wait "$CHROME_PID" 2>/dev/null || true
  fi
  
  # Clean up profile to avoid corruption issues
  cleanup_chrome_profile
  
  sleep "$CHROME_RESTART_DELAY"
  launch_chrome
  
  # Wait for Chrome to become responsive
  log "Waiting for Chrome to become responsive..."
  local attempts=0
  while [ $attempts -lt 30 ]; do
    if check_chrome_health; then
      log "Chrome is responsive again!"
      # Reset restart counter on successful recovery
      CHROME_RESTART_COUNT=0
      return 0
    fi
    sleep 1
    attempts=$((attempts + 1))
  done
  
  log "Chrome failed to become responsive after restart"
  return 1
}

# Initial Chrome launch
launch_chrome

# Chrome binds to localhost, socat makes it accessible from outside container on CDP_PORT
log "Starting socat forwarder: external ${CDP_PORT} -> internal ${INTERNAL_CDP_PORT}"
sleep 2
socat TCP-LISTEN:${CDP_PORT},bind=0.0.0.0,fork,reuseaddr TCP:127.0.0.1:${INTERNAL_CDP_PORT} &
SOCAT_PID=$!

# Start VNC server
log "Starting x11vnc on port ${VNC_PORT}"
x11vnc -display ${DISPLAY} -forever -shared -nopw -rfbport ${VNC_PORT} -quiet &
VNC_PID=$!

# Start noVNC (websockify) bridge
log "Starting websockify on port ${NOVNC_PORT}"
websockify --web=/usr/share/novnc ${NOVNC_PORT} localhost:${VNC_PORT} &
WEBSOCKIFY_PID=$!

log "Browser stack is ready"
log " - CDP:    http://localhost:${CDP_PORT}"
log " - noVNC:  http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=1&resize=scale"

# Forward CTRL+C to subprocesses
cleanup() {
  log "Shutting down all processes..."
  kill ${WEBSOCKIFY_PID:-} ${VNC_PID:-} ${SOCAT_PID:-} ${CHROME_PID:-} ${FLUXBOX_PID:-} ${XVFB_PID:-} 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM

# Main monitoring loop - actively watch Chrome and restart if it crashes
log "Starting Chrome health monitoring (interval: ${CHROME_HEALTH_CHECK_INTERVAL}s)..."
while true; do
  sleep "$CHROME_HEALTH_CHECK_INTERVAL"
  
  # Check if Chrome process is still running
  if ! kill -0 "$CHROME_PID" 2>/dev/null; then
    log "WARNING: Chrome process (PID ${CHROME_PID}) has died!"
    restart_chrome || {
      log "Failed to restart Chrome, exiting container for full restart"
      exit 1
    }
    continue
  fi
  
  # Check if Chrome CDP is responsive
  if ! check_chrome_health; then
    log "WARNING: Chrome CDP is not responding!"
    restart_chrome || {
      log "Failed to restart Chrome, exiting container for full restart"
      exit 1
    }
    continue
  fi
  
  # Check if critical support processes are still running
  if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    log "ERROR: Xvfb has died, container restart required"
    exit 1
  fi
  
  if ! kill -0 "$SOCAT_PID" 2>/dev/null; then
    log "ERROR: socat forwarder has died, container restart required"
    exit 1
  fi
done
