set -euo pipefail

log() {
  printf '[remote-browser] %s\n' "$1"
}

# Ensure runtime directories exist (fresh profile each run)
rm -rf "$HOME/chrome-profile" "$HOME/.config/chromium" "$HOME/.cache/chromium"
mkdir -p "$HOME/chrome-profile"
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix
lock_file="/tmp/.X${DISPLAY#:}.lock"
rm -f "$lock_file"

INTERNAL_CDP_PORT="${CDP_PORT}"

# Start the virtual framebuffer
log "Starting Xvfb on display ${DISPLAY}"
Xvfb ${DISPLAY} -screen 0 1920x1080x24 -ac &
XVFB_PID=$!
sleep 1

# Start a lightweight window manager
log "Starting fluxbox"
fluxbox &
FLUXBOX_PID=$!

# Start VNC server
log "Starting x11vnc on port ${VNC_PORT}"
x11vnc -display ${DISPLAY} -forever -shared -nopw -rfbport ${VNC_PORT} -quiet &
VNC_PID=$!

# Start noVNC (websockify) bridge
log "Starting websockify on port ${NOVNC_PORT}"
websockify --web=/usr/share/novnc ${NOVNC_PORT} localhost:${VNC_PORT} &
WEBSOCKIFY_PID=$!

# Function to start Chrome and socat
start_chrome() {
  log "Launching Chromium (internal CDP port ${INTERNAL_CDP_PORT})"
  
  # Clean up any existing socat
  pkill -f "socat.*${INTERNAL_CDP_PORT}" 2>/dev/null || true
  
  # Fresh profile for each Chrome instance
  rm -rf "$HOME/chrome-profile"
  mkdir -p "$HOME/chrome-profile"
  
  chromium \
    --remote-debugging-port="${INTERNAL_CDP_PORT}" \
    --remote-allow-origins="*" \
    --user-data-dir="$HOME/chrome-profile" \
    --disable-setuid-sandbox \
    --no-sandbox \
    --disable-dev-shm-usage \
    --disable-gpu \
    --no-first-run \
    --no-default-browser-check \
    --disable-session-crashed-bubble \
    --disable-features=InfiniteSessionRestore \
    --start-maximized \
    --force-device-scale-factor=1 \
    --allow-insecure-localhost \
    --autoplay-policy=no-user-gesture-required \
    --test-type \
    --noerrdialogs \
    --incognito \
    --disable-background-timer-throttling \
    --disable-backgrounding-occluded-windows \
    --disable-renderer-backgrounding \
    --disable-hang-monitor \
    --class=RemoteBrowser \
    "${CHROME_START_URL}" \
    &
  CHROME_PID=$!
  
  # Wait for Chrome to start listening
  sleep 2
  
  # Start socat forwarder
  log "Starting socat forwarder for CDP port ${INTERNAL_CDP_PORT}"
  socat TCP-LISTEN:${INTERNAL_CDP_PORT},bind=0.0.0.0,fork,reuseaddr TCP:[::1]:${INTERNAL_CDP_PORT} &
  SOCAT_PID=$!
}

# Initial Chrome start
start_chrome

log "Browser stack is ready"
log " - CDP:    http://localhost:${CDP_PORT}"
log " - noVNC:  http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=1&resize=scale"

# Cleanup function
cleanup() {
  log "Shutting down"
  kill ${WEBSOCKIFY_PID} ${VNC_PID} ${SOCAT_PID:-} ${CHROME_PID:-} ${FLUXBOX_PID} ${XVFB_PID} 2>/dev/null || true
  wait 2>/dev/null || true
}

trap cleanup INT TERM

# Monitor Chrome and restart if it crashes
while true; do
  if ! kill -0 ${CHROME_PID} 2>/dev/null; then
    log "⚠️  Chrome crashed! Restarting in 2 seconds..."
    sleep 2
    start_chrome
    log "✅ Chrome restarted"
  fi
  
  # Also check if critical services are down
  if ! kill -0 ${XVFB_PID} 2>/dev/null; then
    log "❌ Xvfb died, exiting"
    cleanup
    exit 1
  fi
  
  sleep 5
done