# PlatosCave/server.py
import os
import subprocess
import json
import time
from functools import wraps
from pathlib import Path
from typing import Optional
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)

# API Key for authentication (set via environment variable)
API_KEY = os.environ.get('PLATOS_CAVE_API_KEY', '')

# CORS configuration - restrict to allowed origins, support credentials (cookies)
ALLOWED_ORIGINS = [
    "https://platoscave.jackie-courtney.com",
    "https://platoscave.jackieec956.workers.dev",  # Your Cloudflare Pages domain
    "http://localhost:5173",  # Vite dev server
    "http://localhost:8000",  # Local preview
]

CORS(app, 
     resources={r"/*": {"origins": ALLOWED_ORIGINS}},
     supports_credentials=True)

socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    ping_timeout=300,  # 5 minutes before considering connection dead
    ping_interval=25   # Send ping every 25 seconds to keep connection alive
)


def require_api_key(f):
    """Decorator to require API key for protected endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip API key check if no key is configured (development mode)
        if not API_KEY:
            return f(*args, **kwargs)
        
        # Check for API key in header
        provided_key = request.headers.get('X-API-Key', '')
        if provided_key != API_KEY:
            print(f"[SERVER DEBUG] API key rejected: provided='{provided_key[:8]}...' (length {len(provided_key)})", flush=True)
            return jsonify({'error': 'Invalid or missing API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

UPLOAD_FOLDER = 'papers'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BROWSER_COMPOSE_FILE = Path(__file__).parent / 'docker-compose.browser.yaml'

# Separate internal (for health checks) and public (for client access) URLs
# Internal URLs: used by server.py to check Docker container health
BROWSER_CDP_INTERNAL_URL = os.environ.get('REMOTE_BROWSER_CDP_INTERNAL_URL', 'http://localhost:9222')
BROWSER_NOVNC_INTERNAL_URL = os.environ.get('REMOTE_BROWSER_NOVNC_INTERNAL_URL', 'http://localhost:7900')

# Public URLs: sent to frontend and used by main.py to connect
BROWSER_CDP_PUBLIC_URL = os.environ.get('REMOTE_BROWSER_CDP_PUBLIC_URL', 'http://localhost:9222')
BROWSER_NOVNC_PUBLIC_URL = os.environ.get('REMOTE_BROWSER_NOVNC_PUBLIC_URL', 'http://localhost:7900/vnc.html?autoconnect=1&resize=scale')

# Construct health check URLs from internal bases
BROWSER_CDP_HEALTH_URL = f"{BROWSER_CDP_INTERNAL_URL.rstrip('/')}/json/version"
BROWSER_NOVNC_HEALTH_URL = BROWSER_NOVNC_INTERNAL_URL

# Global dictionary to track running processes per session
# session_id -> process object
active_processes = {}


def emit_json_message(payload: dict) -> None:
    """Send a structured payload to the frontend over WebSocket."""
    print(f"[SERVER DEBUG] Emitting message: {payload.get('type', 'UNKNOWN')}", flush=True)
    socketio.emit('status_update', {'data': json.dumps(payload)})
    print(f"[SERVER DEBUG] Message emitted successfully", flush=True)


def wait_for_http_ok(url: str, timeout: float = 30.0, interval: float = 1.5) -> bool:
    """Poll the given URL until it returns HTTP 200 or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib_request.urlopen(url, timeout=interval) as response:
                if 200 <= response.status < 300:
                    return True
        except (URLError, HTTPError, OSError, ConnectionResetError):
            pass
        socketio.sleep(interval)
    return False


def make_cdp_request(url: str, method: str = 'GET', timeout: float = 5.0) -> Optional[str]:
    """Make an HTTP request to CDP endpoint with required Host header."""
    req = urllib_request.Request(url, method=method)
    # Chrome DevTools requires Host header to be localhost or an IP
    req.add_header('Host', 'localhost')
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            if 200 <= response.status < 300:
                return response.read().decode('utf-8')
    except (URLError, HTTPError, ConnectionResetError) as e:
        print(f"[SERVER DEBUG] CDP request error for {url}: {e}", flush=True)
    return None


def fetch_cdp_metadata(url: str, timeout: float = 5.0, retries: int = 3) -> Optional[dict]:
    """
    Retrieve Chrome DevTools metadata JSON from the remote browser.
    Retries on connection errors (browser may be restarting/busy).
    """
    for attempt in range(retries):
        raw = make_cdp_request(url, timeout=timeout)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[SERVER DEBUG] CDP metadata JSON parse error: {e}", flush=True)
                return None
        # Retry on failure
        if attempt < retries - 1:
            wait_time = 0.5 * (2 ** attempt)
            print(f"[SERVER DEBUG] Retrying CDP metadata fetch in {wait_time}s (attempt {attempt + 1}/{retries})", flush=True)
            time.sleep(wait_time)
    return None


def close_all_browser_tabs() -> bool:
    """
    Close all browser tabs/pages via CDP to reset browser state.
    This ensures each new analysis starts with a clean browser session.

    Returns:
        True if successful, False otherwise
    """
    print("[SERVER DEBUG] ========== CLOSING ALL BROWSER TABS ==========", flush=True)
    try:
        # Get list of all pages/tabs (use INTERNAL URL for Docker-to-Docker communication)
        list_url = f"{BROWSER_CDP_INTERNAL_URL.rstrip('/')}/json/list"
        print(f"[SERVER DEBUG] Fetching tab list from: {list_url}", flush=True)

        raw = make_cdp_request(list_url, timeout=5)
        if not raw:
            print("[SERVER DEBUG] Failed to get tab list", flush=True)
            return False
        tabs = json.loads(raw)

        print(f"[SERVER DEBUG] Found {len(tabs)} tabs/targets", flush=True)

        # Close each page (not background_page or other special types)
        closed_count = 0
        for tab in tabs:
            if tab.get('type') == 'page':
                tab_id = tab.get('id')
                tab_url = tab.get('url', 'unknown')
                print(f"[SERVER DEBUG] Closing tab {tab_id}: {tab_url[:80]}", flush=True)

                close_url = f"{BROWSER_CDP_INTERNAL_URL.rstrip('/')}/json/close/{tab_id}"
                result = make_cdp_request(close_url, timeout=5)
                if result is not None:
                    closed_count += 1
                    print(f"[SERVER DEBUG] ✓ Closed tab {tab_id}", flush=True)
                else:
                    print(f"[SERVER DEBUG] Failed to close tab {tab_id}", flush=True)

        print(f"[SERVER DEBUG] Closed {closed_count} tabs successfully", flush=True)
        return True

    except Exception as e:
        print(f"[SERVER DEBUG] Error closing browser tabs: {e}", flush=True)
        return False


def reset_browser_session() -> bool:
    """
    Reset the browser session to a clean state.
    Ensures exactly one about:blank tab exists.

    Returns:
        True if successful, False otherwise
    """
    print("[SERVER DEBUG] ========== RESETTING BROWSER SESSION ==========", flush=True)

    # Retry logic for connection resets (browser may be busy/restarting)
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Get list of all pages/tabs with timeout (use INTERNAL URL for Docker-to-Docker)
            list_url = f"{BROWSER_CDP_INTERNAL_URL.rstrip('/')}/json/list"
            print(f"[SERVER DEBUG] Fetching tab list from: {list_url} (attempt {attempt + 1}/{max_attempts})", flush=True)

            raw = make_cdp_request(list_url, timeout=3)
            if not raw:
                raise ConnectionError("Failed to get tab list")
            tabs = json.loads(raw)

            pages = [tab for tab in tabs if tab.get('type') == 'page']
            print(f"[SERVER DEBUG] Found {len(pages)} pages", flush=True)

            # Check if we already have a blank tab
            blank_tabs = [tab for tab in pages if tab.get('url', '').startswith('about:blank')]
            non_blank_tabs = [tab for tab in pages if not tab.get('url', '').startswith('about:blank')]

            # Close all non-blank tabs (with shorter timeout per tab)
            for tab in non_blank_tabs:
                tab_id = tab.get('id')
                tab_url = tab.get('url', '')
                print(f"[SERVER DEBUG] Closing non-blank tab {tab_id}: {tab_url[:60]}", flush=True)

                close_url = f"{BROWSER_CDP_INTERNAL_URL.rstrip('/')}/json/close/{tab_id}"
                result = make_cdp_request(close_url, timeout=2)
                if result is not None:
                    print(f"[SERVER DEBUG] ✓ Closed non-blank tab {tab_id}", flush=True)
                else:
                    print(f"[SERVER DEBUG] ✗ Failed to close tab {tab_id} (continuing...)", flush=True)

            # If there are multiple blank tabs, keep only one
            if len(blank_tabs) > 1:
                print(f"[SERVER DEBUG] Found {len(blank_tabs)} blank tabs, keeping only one", flush=True)
                tabs_to_close = blank_tabs[1:]  # Keep first, close rest
                for tab in tabs_to_close:
                    tab_id = tab.get('id')
                    close_url = f"{BROWSER_CDP_INTERNAL_URL.rstrip('/')}/json/close/{tab_id}"
                    result = make_cdp_request(close_url, timeout=2)
                    if result is not None:
                        print(f"[SERVER DEBUG] ✓ Closed extra blank tab {tab_id}", flush=True)
                    else:
                        print(f"[SERVER DEBUG] ✗ Failed to close blank tab {tab_id} (continuing...)", flush=True)

            # If no blank tabs exist, create one
            if len(blank_tabs) == 0:
                print(f"[SERVER DEBUG] No blank tabs found, creating one...", flush=True)
                time.sleep(0.2)  # Brief pause

                new_tab_url = f"{BROWSER_CDP_INTERNAL_URL.rstrip('/')}/json/new"
                raw = make_cdp_request(new_tab_url, method='PUT', timeout=3)
                if raw:
                    try:
                        tab_data = json.loads(raw)
                        print(f"[SERVER DEBUG] ✓ Created blank tab with ID: {tab_data.get('id')}", flush=True)
                    except json.JSONDecodeError:
                        print(f"[SERVER DEBUG] Created new tab but couldn't parse response", flush=True)
            else:
                print(f"[SERVER DEBUG] ✓ Kept existing blank tab", flush=True)

            print("[SERVER DEBUG] Browser session reset completed", flush=True)
            return True

        except ConnectionResetError as e:
            if attempt < max_attempts - 1:
                wait_time = 1.0 * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                print(f"[SERVER DEBUG] Connection reset by browser, retrying in {wait_time}s... (attempt {attempt + 1}/{max_attempts})", flush=True)
                time.sleep(wait_time)
                continue
            else:
                print(f"[SERVER DEBUG] Browser connection failed after {max_attempts} attempts - browser may be overwhelmed", flush=True)
                return False
        except (URLError, HTTPError, TimeoutError) as e:
            print(f"[SERVER DEBUG] Browser connection error during reset: {e}", flush=True)
            if attempt < max_attempts - 1:
                print(f"[SERVER DEBUG] Retrying... (attempt {attempt + 1}/{max_attempts})", flush=True)
                time.sleep(1.0)
                continue
            print("[SERVER DEBUG] Browser may be restarting or unresponsive", flush=True)
            return False
        except Exception as e:
            print(f"[SERVER DEBUG] Error resetting browser session: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    return False


def kill_process_safely(process, timeout: float = 5.0) -> bool:
    """
    Safely terminate a process, trying SIGTERM first, then SIGKILL if needed.

    Args:
        process: subprocess.Popen object
        timeout: seconds to wait for graceful termination

    Returns:
        True if process was terminated successfully
    """
    if process is None or process.poll() is not None:
        return True  # Already dead

    print(f"[SERVER DEBUG] Attempting to terminate process PID {process.pid}", flush=True)

    try:
        # Try graceful termination first (SIGTERM)
        process.terminate()
        try:
            process.wait(timeout=timeout)
            print(f"[SERVER DEBUG] ✓ Process {process.pid} terminated gracefully", flush=True)
            return True
        except subprocess.TimeoutExpired:
            # Force kill if still alive (SIGKILL)
            print(f"[SERVER DEBUG] Process {process.pid} didn't terminate, force killing...", flush=True)
            process.kill()
            process.wait(timeout=2)
            print(f"[SERVER DEBUG] ✓ Process {process.pid} force killed", flush=True)
            return True
    except Exception as e:
        print(f"[SERVER DEBUG] Error killing process {process.pid}: {e}", flush=True)
        return False


def ensure_remote_browser_service() -> Optional[dict]:
    """Ensure the remote browser Docker service is running and healthy.
    
    When running inside Docker (via docker-compose), the remote browser container
    is already managed by docker-compose, so we skip trying to start it and just
    verify connectivity.
    """
    print("[SERVER DEBUG] ========== STARTING ensure_remote_browser_service ==========", flush=True)
    emit_json_message({
        'type': 'UPDATE',
        'stage': 'Browser',
        'text': 'Ensuring remote browser service is running...'
    })

    # Check if we're running inside Docker (docker command won't be available)
    # In Docker deployment, the remote-browser container is started by docker-compose
    running_in_docker = not os.path.exists('/var/run/docker.sock')
    
    if not running_in_docker and BROWSER_COMPOSE_FILE.exists():
        # Only try to start browser via docker if we're NOT inside a container
        try:
            subprocess.run(
                ['docker', 'compose', '-f', str(BROWSER_COMPOSE_FILE), 'up', '-d', 'remote-browser'],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=BROWSER_COMPOSE_FILE.parent
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[SERVER DEBUG] Could not start browser via docker: {exc} (will try to connect anyway)", flush=True)
    else:
        print("[SERVER DEBUG] Running in Docker or docker-compose not found - assuming browser container is already running", flush=True)

    # Check noVNC but don't fail if unavailable (it's for viewing only)
    print("[SERVER DEBUG] Checking noVNC endpoint (optional)...", flush=True)
    if wait_for_http_ok(BROWSER_NOVNC_HEALTH_URL, timeout=5.0):
        print("[SERVER DEBUG] noVNC endpoint is accessible", flush=True)
    else:
        print("[SERVER DEBUG] noVNC endpoint not accessible (continuing - not critical)", flush=True)
        # Don't return None - noVNC is optional for viewing, CDP is what matters

    # Wait for CDP endpoint (this is critical)
    print("[SERVER DEBUG] Waiting for CDP endpoint (required)...", flush=True)
    metadata = None
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        metadata = fetch_cdp_metadata(BROWSER_CDP_HEALTH_URL)
        if metadata:
            print("[SERVER DEBUG] CDP endpoint is ready", flush=True)
            break
        socketio.sleep(1.5)

    if not metadata:
        emit_json_message({
            'type': 'ERROR',
            'message': 'Remote browser CDP endpoint did not become ready in time.'
        })
        return None

    # Build both PUBLIC and INTERNAL WebSocket URLs
    # PUBLIC: for frontend (browser outside Docker) - uses localhost
    # INTERNAL: for main.py (runs inside Docker) - uses remote-browser hostname
    ws_url_original = metadata.get('webSocketDebuggerUrl')
    ws_url_public = None
    ws_url_internal = None
    
    if ws_url_original:
        print(f"[SERVER DEBUG] Original WebSocket URL from metadata: {ws_url_original}", flush=True)
        
        parsed_ws = urlparse(ws_url_original)
        
        # Build PUBLIC WebSocket URL (for frontend)
        parsed_cdp_public = urlparse(BROWSER_CDP_PUBLIC_URL)
        public_hostname = parsed_cdp_public.hostname or 'localhost'
        public_port = parsed_cdp_public.port or 9222
        ws_scheme_public = 'wss' if parsed_cdp_public.scheme == 'https' else 'ws'
        ws_netloc_public = f"{public_hostname}:{public_port}"
        ws_url_public = urlunparse((ws_scheme_public, ws_netloc_public, parsed_ws.path, '', '', ''))
        
        # Build INTERNAL WebSocket URL (for main.py inside Docker)
        parsed_cdp_internal = urlparse(BROWSER_CDP_INTERNAL_URL)
        internal_hostname = parsed_cdp_internal.hostname or 'remote-browser'
        internal_port = parsed_cdp_internal.port or 9222
        ws_netloc_internal = f"{internal_hostname}:{internal_port}"
        ws_url_internal = urlunparse(('ws', ws_netloc_internal, parsed_ws.path, '', '', ''))
        
        print(f"[SERVER DEBUG] Public WebSocket URL: {ws_url_public}", flush=True)
        print(f"[SERVER DEBUG] Internal WebSocket URL: {ws_url_internal}", flush=True)

    # Payload for frontend (public URLs)
    browser_payload_public = {
        'type': 'BROWSER_ADDRESS',
        'novnc_url': BROWSER_NOVNC_PUBLIC_URL,
        'cdp_url': BROWSER_CDP_PUBLIC_URL,
        'cdp_websocket': ws_url_public
    }
    
    # Payload for main.py subprocess (internal URLs)
    browser_payload_internal = {
        'type': 'BROWSER_ADDRESS',
        'novnc_url': BROWSER_NOVNC_INTERNAL_URL,
        'cdp_url': BROWSER_CDP_INTERNAL_URL,
        'cdp_websocket': ws_url_internal
    }
    
    print(f"[SERVER DEBUG] About to emit BROWSER_ADDRESS (public): {browser_payload_public}", flush=True)
    emit_json_message(browser_payload_public)
    print(f"[SERVER DEBUG] BROWSER_ADDRESS emitted!", flush=True)

    emit_json_message({
        'type': 'UPDATE',
        'stage': 'Browser',
        'text': 'Remote browser is ready.'
    })

    print("[SERVER DEBUG] ========== ensure_remote_browser_service COMPLETED ==========", flush=True)
    
    # Return both payloads - caller uses internal for subprocess, public for frontend
    return {
        'public': browser_payload_public,
        'internal': browser_payload_internal
    }

def run_script_and_stream_output(filepath, settings):
    """
    Run PDF analysis using main.py with --pdf flag
    Streams real-time updates via WebSocket
    """
    print(f"[SERVER DEBUG] ========== STARTING PDF ANALYSIS ==========", flush=True)
    print(f"[SERVER DEBUG] PDF path: {filepath}", flush=True)
    print(f"[SERVER DEBUG] Settings: {settings}", flush=True)

    # Ensure remote browser is available (needed for claim verification even in PDF mode)
    print("[SERVER DEBUG] Ensuring remote browser service for verification...", flush=True)
    browser_result = ensure_remote_browser_service()
    if browser_result is None:
        print("[SERVER DEBUG] Remote browser failed, falling back to local browser", flush=True)
        browser_info_internal = {}
        browser_info_public = {}
    else:
        browser_info_internal = browser_result.get('internal', {})
        browser_info_public = browser_result.get('public', {})

    command = [
        'python', 'main.py', '--pdf', filepath,
        '--agent-aggressiveness', str(settings.get('agentAggressiveness', 5)),
        '--evidence-threshold', str(settings.get('evidenceThreshold', 0.8))
    ]

    # Set environment to suppress browser-use logs
    env = os.environ.copy()
    env['SUPPRESS_LOGS'] = 'true'

    # Set remote browser environment variables using INTERNAL URLs (for Docker networking)
    if browser_info_internal:
        cdp_url = browser_info_internal.get('cdp_url')
        cdp_ws = browser_info_internal.get('cdp_websocket')
        if cdp_url:
            env['REMOTE_BROWSER_CDP_URL'] = cdp_url
        if cdp_ws:
            env['REMOTE_BROWSER_CDP_WS'] = cdp_ws
        novnc_url = browser_info_internal.get('novnc_url')
        if novnc_url:
            env['REMOTE_BROWSER_NOVNC_URL'] = novnc_url
        print(f"[SERVER DEBUG] Passing internal CDP URL to main.py: {cdp_url}", flush=True)
        print(f"[SERVER DEBUG] Passing internal CDP WS to main.py: {cdp_ws}", flush=True)
    else:
        # Ensure remote browser env vars are not set for local browser fallback
        env.pop('REMOTE_BROWSER_CDP_URL', None)
        env.pop('REMOTE_BROWSER_CDP_WS', None)
        env.pop('REMOTE_BROWSER_NOVNC_URL', None)
        print("[SERVER DEBUG] Using local browser - no remote CDP environment variables set", flush=True)

    print(f"[SERVER DEBUG] Starting subprocess with command: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding='utf-8',
        cwd=os.path.dirname(os.path.abspath(__file__)),  # Run from backend directory
        env=env
    )
    print(f"[SERVER DEBUG] Subprocess started with PID: {process.pid}", flush=True)

    # Re-emit browser info to ensure frontend WebSocket has connected and receives it
    # Use PUBLIC URLs for frontend (so browser opens user's localhost, not Docker internal)
    if browser_info_public and browser_info_public.get('cdp_url'):
        print(f"[SERVER DEBUG] Re-emitting BROWSER_ADDRESS to frontend (public): {browser_info_public}", flush=True)
        socketio.emit('status_update', {'data': json.dumps(browser_info_public)})
        socketio.sleep(0.1)  # Small delay to ensure delivery
        print("[SERVER DEBUG] BROWSER_ADDRESS re-emitted!", flush=True)
    else:
        print("[SERVER DEBUG] Using local browser, no BROWSER_ADDRESS to emit", flush=True)

    for line in process.stdout:
        line = line.strip()
        if line:
            # Only send valid JSON to frontend
            try:
                json.loads(line)  # Validate it's JSON
                socketio.emit('status_update', {'data': line})
                socketio.sleep(0)
            except json.JSONDecodeError:
                # Ignore non-JSON output
                print(f"[SERVER DEBUG] Skipping non-JSON line: {line[:100]}", flush=True)
                pass

    print("[SERVER DEBUG] Subprocess stdout closed, waiting for process to finish...", flush=True)
    process.wait()
    print(f"[SERVER DEBUG] Subprocess finished with return code: {process.returncode}", flush=True)

    # Handle errors
    if process.returncode != 0:
        error_output = process.stderr.read()
        print(f"PDF Analysis Error: {error_output}")
        error_message = json.dumps({"type": "ERROR", "message": error_output})
        socketio.emit('status_update', {'data': error_message})


def run_url_analysis_and_stream_output(url, settings, session_id=None):
    """
    Run URL analysis using browser-use + DAG generation
    Streams real-time updates via WebSocket

    Args:
        url: URL to analyze
        settings: Analysis settings
        session_id: WebSocket session ID for process tracking
    """
    print(f"[SERVER DEBUG] ========== STARTING run_url_analysis_and_stream_output ==========", flush=True)
    print(f"[SERVER DEBUG] URL: {url}", flush=True)
    print(f"[SERVER DEBUG] Settings: {settings}", flush=True)
    print(f"[SERVER DEBUG] Session ID: {session_id}", flush=True)

    # Kill any previous main.py analysis processes
    try:
        print("[SERVER DEBUG] Killing old main.py processes...", flush=True)
        result = subprocess.run(['pkill', '-9', '-f', 'main.py'],
                      capture_output=True,
                      text=True,
                      timeout=3)
        print(f"[SERVER DEBUG] pkill result: return_code={result.returncode}, stdout={result.stdout}, stderr={result.stderr}", flush=True)
        socketio.sleep(1.0)  # Give processes time to fully clean up
        print("[SERVER DEBUG] Old processes killed, waiting completed", flush=True)
    except subprocess.TimeoutExpired:
        print("[SERVER DEBUG] pkill timeout (this is OK)", flush=True)
    except Exception as e:
        print(f"[SERVER DEBUG] pkill error: {e}", flush=True)

    # Reset browser session FIRST, before getting connection info
    # This ensures we start fresh but don't break CDP connections
    print("[SERVER DEBUG] Resetting browser session before getting connection info...", flush=True)
    reset_browser_session()

    # Give browser a moment to stabilize after reset
    socketio.sleep(0.5)

    print("[SERVER DEBUG] Calling ensure_remote_browser_service()...", flush=True)
    browser_result = ensure_remote_browser_service()
    if browser_result is None:
        print("[SERVER DEBUG] Remote browser failed, falling back to local browser", flush=True)
        emit_json_message({
            'type': 'WARNING',
            'message': 'Remote browser unavailable, using local browser fallback'
        })
        browser_info_internal = {}
        browser_info_public = {}
    else:
        browser_info_internal = browser_result.get('internal', {})
        browser_info_public = browser_result.get('public', {})
        print(f"[SERVER DEBUG] Browser info internal: {browser_info_internal}", flush=True)
        print(f"[SERVER DEBUG] Browser info public: {browser_info_public}", flush=True)

    command = [
        'python', 'main.py', '--url', url,
        # Note: main.py doesn't use these settings yet, but we pass them for future use
        '--agent-aggressiveness', str(settings.get('agentAggressiveness', 5)),
        '--evidence-threshold', str(settings.get('evidenceThreshold', 0.8))
    ]

    # Set environment to suppress browser-use logs
    env = os.environ.copy()
    env['SUPPRESS_LOGS'] = 'true'

    # Set remote browser environment variables using INTERNAL URLs (for Docker networking)
    if browser_info_internal:
        cdp_url = browser_info_internal.get('cdp_url')
        cdp_ws = browser_info_internal.get('cdp_websocket')
        if cdp_url:
            env['REMOTE_BROWSER_CDP_URL'] = cdp_url
        if cdp_ws:
            env['REMOTE_BROWSER_CDP_WS'] = cdp_ws
        novnc_url = browser_info_internal.get('novnc_url')
        if novnc_url:
            env['REMOTE_BROWSER_NOVNC_URL'] = novnc_url
        print(f"[SERVER DEBUG] Passing internal CDP URL to main.py: {cdp_url}", flush=True)
        print(f"[SERVER DEBUG] Passing internal CDP WS to main.py: {cdp_ws}", flush=True)
    else:
        # Ensure remote browser env vars are not set for local browser fallback
        env.pop('REMOTE_BROWSER_CDP_URL', None)
        env.pop('REMOTE_BROWSER_CDP_WS', None)
        env.pop('REMOTE_BROWSER_NOVNC_URL', None)
        print("[SERVER DEBUG] Using local browser - no remote CDP environment variables set", flush=True)

    print(f"[SERVER DEBUG] Starting subprocess with command: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding='utf-8',
        cwd=os.path.dirname(os.path.abspath(__file__)),  # Run from backend directory
        env=env  # Pass modified environment
    )
    print(f"[SERVER DEBUG] Subprocess started with PID: {process.pid}", flush=True)

    # Track this process for the session
    if session_id:
        active_processes[session_id] = process
        print(f"[SERVER DEBUG] Tracking process {process.pid} for session {session_id}", flush=True)

    # Re-emit browser info to ensure frontend WebSocket has connected and receives it
    # Use PUBLIC URLs for frontend (so browser opens user's localhost, not Docker internal)
    if browser_info_public and browser_info_public.get('cdp_url'):
        print(f"[SERVER DEBUG] Re-emitting BROWSER_ADDRESS to frontend (public): {browser_info_public}", flush=True)
        socketio.emit('status_update', {'data': json.dumps(browser_info_public)})
        socketio.sleep(0.1)  # Small delay to ensure delivery
        print("[SERVER DEBUG] BROWSER_ADDRESS re-emitted!", flush=True)
    else:
        print("[SERVER DEBUG] Using local browser, no BROWSER_ADDRESS to emit", flush=True)

    print("[SERVER DEBUG] Starting to read subprocess stdout...", flush=True)
    for line in process.stdout:
        line = line.strip()
        if line:
            # Only send valid JSON to frontend (filter out browser-use debug logs)
            try:
                json.loads(line)  # Validate it's JSON
                socketio.emit('status_update', {'data': line})
                socketio.sleep(0)
            except json.JSONDecodeError:
                # Ignore non-JSON output (debug logs from browser-use/LLM)
                print(f"[SERVER DEBUG] Skipping non-JSON line: {line[:100]}", flush=True)
                pass

    print("[SERVER DEBUG] Subprocess stdout closed, waiting for process to finish...", flush=True)
    process.wait()
    print(f"[SERVER DEBUG] Subprocess finished with return code: {process.returncode}", flush=True)

    # Remove from tracking
    if session_id and session_id in active_processes:
        del active_processes[session_id]
        print(f"[SERVER DEBUG] Removed process from tracking for session {session_id}", flush=True)

    # Handle errors
    if process.returncode != 0:
        error_output = process.stderr.read()
        print(f"URL Analysis Error: {error_output}")
        error_message = json.dumps({"type": "ERROR", "message": error_output})
        socketio.emit('status_update', {'data': error_message})


@app.route('/api/upload', methods=['POST'])
@require_api_key
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']

    settings = {
        'agentAggressiveness': request.form.get('agentAggressiveness', 5),
        'evidenceThreshold': request.form.get('evidenceThreshold', 0.8)
    }

    if file and file.filename:
        # Validate file type - only accept PDFs
        if not file.filename.lower().endswith('.pdf'):
            return {'error': 'Only PDF files are supported'}, 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        socketio.start_background_task(run_script_and_stream_output, filepath, settings)
        return {'message': 'Processing started.'}, 202
    return 'No selected file', 400


@app.route('/api/cleanup', methods=['POST'])
@require_api_key
def cleanup():
    """
    Cleanup endpoint: Reset browser state to prepare for new analysis
    Called when user refreshes or clicks logo
    
    Simply resets browser tabs - the next analysis will handle process management
    """
    print("[SERVER DEBUG] ========== /api/cleanup ENDPOINT HIT ==========", flush=True)
    try:
        # Reset browser session (close all tabs, return to blank state)
        print("[SERVER DEBUG] Resetting browser session...", flush=True)
        reset_browser_session()

        print("[SERVER DEBUG] Cleanup completed successfully", flush=True)
        return {'message': 'Cleanup completed'}, 200
    except Exception as e:
        print(f"[SERVER DEBUG] Cleanup error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500


@app.route('/api/analyze-url', methods=['POST'])
@require_api_key
def analyze_url():
    """
    Analyze a research paper from URL using browser-use + DAG generation
    """
    print("[SERVER DEBUG] ========== /api/analyze-url ENDPOINT HIT ==========", flush=True)
    data = request.get_json()
    print(f"[SERVER DEBUG] Received data: {data}", flush=True)

    if 'url' not in data:
        print("[SERVER DEBUG] ERROR: No URL provided in request", flush=True)
        return {'error': 'No URL provided'}, 400

    url = data['url']
    settings = {
        'agentAggressiveness': data.get('agentAggressiveness', 5),
        'evidenceThreshold': data.get('evidenceThreshold', 0.8)
    }

    # Get session ID from request header or generate one
    session_id = data.get('sessionId', request.headers.get('X-Session-ID', None))

    print(f"[SERVER DEBUG] Starting background task for URL: {url}, session: {session_id}", flush=True)
    # Start URL analysis in background and stream updates via WebSocket
    socketio.start_background_task(run_url_analysis_and_stream_output, url, settings, session_id)
    print("[SERVER DEBUG] Background task started, returning 202", flush=True)

    return {'message': 'URL analysis started.', 'url': url}, 202

@socketio.on('connect')
def handle_connect():
    print('[SERVER DEBUG] ========== CLIENT CONNECTED TO WEBSOCKET ==========', flush=True)
    print(f'[SERVER DEBUG] Client ID: {request.sid}', flush=True)

@socketio.on('disconnect')
def handle_disconnect():
    print('[SERVER DEBUG] ========== CLIENT DISCONNECTED FROM WEBSOCKET ==========', flush=True)
    print(f'[SERVER DEBUG] Client ID: {request.sid}', flush=True)

    # Kill any running process associated with this session
    if request.sid in active_processes:
        process = active_processes[request.sid]
        print(f'[SERVER DEBUG] Killing process {process.pid} for disconnected session', flush=True)
        kill_process_safely(process)
        del active_processes[request.sid]

    # Reset browser session when client disconnects
    print('[SERVER DEBUG] Resetting browser session after disconnect...', flush=True)
    reset_browser_session()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)

