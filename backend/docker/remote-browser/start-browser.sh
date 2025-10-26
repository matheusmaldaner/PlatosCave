#!/usr/bin/env bash
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

# Use the same port internally and externally (no offset needed)
INTERNAL_CDP_PORT="${CDP_PORT}"

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

# Launch Chromium with remote debugging enabled
log "Launching Chromium (internal CDP port ${INTERNAL_CDP_PORT})"
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
  --class=RemoteBrowser \
  "${CHROME_START_URL}" \
  &
CHROME_PID=$!

# Chrome binds to localhost (may be IPv4 or IPv6), so we need socat to make it accessible from outside container
log "Starting socat forwarder for CDP port ${INTERNAL_CDP_PORT}"
# Use [::1] (IPv6 localhost) as Chrome often binds to IPv6 first
socat TCP-LISTEN:${INTERNAL_CDP_PORT},bind=0.0.0.0,fork,reuseaddr TCP:[::1]:${INTERNAL_CDP_PORT} &
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
trap 'log "Shutting down"; kill ${WEBSOCKIFY_PID} ${VNC_PID} ${SOCAT_PID} ${CHROME_PID} ${FLUXBOX_PID} ${XVFB_PID} 2>/dev/null || true; wait' INT TERM

# Wait for any process to exit, then clean up
wait -n
STATUS=$?
trap - INT TERM
kill ${WEBSOCKIFY_PID} ${VNC_PID} ${SOCAT_PID} ${CHROME_PID} ${FLUXBOX_PID} ${XVFB_PID} 2>/dev/null || true
wait || true
exit ${STATUS}
