# PlatosCave/server.py
import os
import subprocess
import json  # Make sure json is imported
import time
from pathlib import Path
from typing import Optional
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse, urlunparse
from flask import Flask, request
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

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
        except (URLError, HTTPError):
            pass
        socketio.sleep(interval)
    return False


def fetch_cdp_metadata(url: str, timeout: float = 5.0) -> Optional[dict]:
    """Retrieve Chrome DevTools metadata JSON from the remote browser."""
    try:
        with urllib_request.urlopen(url, timeout=timeout) as response:
            if 200 <= response.status < 300:
                raw = response.read().decode('utf-8')
                return json.loads(raw)
    except (URLError, HTTPError, json.JSONDecodeError):
        return None
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
        # Get list of all pages/tabs
        list_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/list"
        print(f"[SERVER DEBUG] Fetching tab list from: {list_url}", flush=True)

        with urllib_request.urlopen(list_url, timeout=5) as response:
            tabs = json.loads(response.read().decode('utf-8'))

        print(f"[SERVER DEBUG] Found {len(tabs)} tabs/targets", flush=True)

        # Close each page (not background_page or other special types)
        closed_count = 0
        for tab in tabs:
            if tab.get('type') == 'page':
                tab_id = tab.get('id')
                tab_url = tab.get('url', 'unknown')
                print(f"[SERVER DEBUG] Closing tab {tab_id}: {tab_url[:80]}", flush=True)

                close_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/close/{tab_id}"
                try:
                    with urllib_request.urlopen(close_url, timeout=5) as close_response:
                        if 200 <= close_response.status < 300:
                            closed_count += 1
                            print(f"[SERVER DEBUG] ✓ Closed tab {tab_id}", flush=True)
                except (URLError, HTTPError) as e:
                    print(f"[SERVER DEBUG] Failed to close tab {tab_id}: {e}", flush=True)

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

    try:
        # Get list of all pages/tabs
        list_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/list"
        print(f"[SERVER DEBUG] Fetching tab list from: {list_url}", flush=True)

        with urllib_request.urlopen(list_url, timeout=5) as response:
            tabs = json.loads(response.read().decode('utf-8'))

        pages = [tab for tab in tabs if tab.get('type') == 'page']
        print(f"[SERVER DEBUG] Found {len(pages)} pages", flush=True)

        # Check if we already have a blank tab
        blank_tabs = [tab for tab in pages if tab.get('url', '').startswith('about:blank')]
        non_blank_tabs = [tab for tab in pages if not tab.get('url', '').startswith('about:blank')]

        # Close all non-blank tabs
        for tab in non_blank_tabs:
            tab_id = tab.get('id')
            tab_url = tab.get('url', '')
            print(f"[SERVER DEBUG] Closing non-blank tab {tab_id}: {tab_url[:60]}", flush=True)

            close_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/close/{tab_id}"
            try:
                req = urllib_request.Request(close_url, method='GET')
                with urllib_request.urlopen(req, timeout=3) as close_response:
                    result = close_response.read().decode('utf-8')
                    print(f"[SERVER DEBUG] ✓ Closed non-blank tab {tab_id}, result: {result[:50]}", flush=True)
            except Exception as e:
                print(f"[SERVER DEBUG] ✗ Failed to close tab {tab_id}: {e}", flush=True)
                import traceback
                traceback.print_exc()

        # If there are multiple blank tabs, keep only one
        if len(blank_tabs) > 1:
            print(f"[SERVER DEBUG] Found {len(blank_tabs)} blank tabs, keeping only one", flush=True)
            tabs_to_close = blank_tabs[1:]  # Keep first, close rest
            print(f"[SERVER DEBUG] Closing {len(tabs_to_close)} extra blank tabs", flush=True)
            for i, tab in enumerate(tabs_to_close, 1):
                tab_id = tab.get('id')
                close_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/close/{tab_id}"
                print(f"[SERVER DEBUG] Closing blank tab {i}/{len(tabs_to_close)}: {tab_id}", flush=True)
                try:
                    req = urllib_request.Request(close_url, method='GET')
                    with urllib_request.urlopen(req, timeout=3) as close_response:
                        result = close_response.read().decode('utf-8')
                        print(f"[SERVER DEBUG] ✓ Closed tab {tab_id}, result: {result[:50]}", flush=True)
                except Exception as e:
                    print(f"[SERVER DEBUG] ✗ Failed to close blank tab {tab_id}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

        # If no blank tabs exist, create one
        if len(blank_tabs) == 0:
            print(f"[SERVER DEBUG] No blank tabs found, creating one...", flush=True)
            time.sleep(0.3)  # Give browser time to process closures

            new_tab_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/new"
            req = urllib_request.Request(new_tab_url, method='PUT')
            with urllib_request.urlopen(req, timeout=5) as response:
                if 200 <= response.status < 300:
                    tab_data = json.loads(response.read().decode('utf-8'))
                    print(f"[SERVER DEBUG] ✓ Created blank tab with ID: {tab_data.get('id')}", flush=True)
        else:
            print(f"[SERVER DEBUG] ✓ Kept existing blank tab", flush=True)

        return True

    except Exception as e:
        print(f"[SERVER DEBUG] Error resetting browser session: {e}", flush=True)
        import traceback
        traceback.print_exc()
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
    """Ensure the remote browser Docker service is running and healthy."""
    print("[SERVER DEBUG] ========== STARTING ensure_remote_browser_service ==========", flush=True)
    emit_json_message({
        'type': 'UPDATE',
        'stage': 'Browser',
        'text': 'Ensuring remote browser service is running...'
    })

    if not BROWSER_COMPOSE_FILE.exists():
        emit_json_message({
            'type': 'ERROR',
            'message': f'Remote browser compose file not found at {BROWSER_COMPOSE_FILE}'
        })
        return None

    try:
        subprocess.run(
            ['docker', 'compose', '-f', str(BROWSER_COMPOSE_FILE), 'up', '-d', 'remote-browser'],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            cwd=BROWSER_COMPOSE_FILE.parent
        )
    except subprocess.CalledProcessError as exc:
        emit_json_message({
            'type': 'ERROR',
            'message': f'Failed to start remote browser service: {exc}'
        })
        return None

    if not wait_for_http_ok(BROWSER_NOVNC_HEALTH_URL):
        emit_json_message({
            'type': 'ERROR',
            'message': 'Remote browser noVNC endpoint did not become ready in time.'
        })
        return None

    metadata = None
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        metadata = fetch_cdp_metadata(BROWSER_CDP_HEALTH_URL)
        if metadata:
            break
        socketio.sleep(1.5)

    if not metadata:
        emit_json_message({
            'type': 'ERROR',
            'message': 'Remote browser CDP endpoint did not become ready in time.'
        })
        return None

    # Rewrite the advertised WebSocket endpoint to use public URL
    ws_url = metadata.get('webSocketDebuggerUrl')
    if ws_url:
        print(f"[SERVER DEBUG] Original WebSocket URL from metadata: {ws_url}", flush=True)
        
        parsed_ws = urlparse(ws_url)
        parsed_cdp_public = urlparse(BROWSER_CDP_PUBLIC_URL)
        
        # Use public hostname and port
        public_hostname = parsed_cdp_public.hostname or 'localhost'
        public_port = parsed_cdp_public.port or 9222
        
        # Determine WebSocket scheme based on public CDP URL (ws for http, wss for https)
        ws_scheme = 'wss' if parsed_cdp_public.scheme == 'https' else 'ws'
        
        # Rebuild WebSocket URL with public hostname/port
        ws_netloc = f"{public_hostname}:{public_port}"
        ws_url = urlunparse((ws_scheme, ws_netloc, parsed_ws.path, '', '', ''))
        
        print(f"[SERVER DEBUG] Rewritten WebSocket URL: {ws_url}", flush=True)

    browser_payload = {
        'type': 'BROWSER_ADDRESS',
        'novnc_url': BROWSER_NOVNC_PUBLIC_URL,
        'cdp_url': BROWSER_CDP_PUBLIC_URL,
        'cdp_websocket': ws_url
    }
    print(f"[SERVER DEBUG] About to emit BROWSER_ADDRESS: {browser_payload}", flush=True)
    emit_json_message(browser_payload)
    print(f"[SERVER DEBUG] BROWSER_ADDRESS emitted!", flush=True)

    emit_json_message({
        'type': 'UPDATE',
        'stage': 'Browser',
        'text': 'Remote browser is ready.'
    })

    print("[SERVER DEBUG] ========== ensure_remote_browser_service COMPLETED ==========", flush=True)
    return browser_payload

def run_script_and_stream_output(filepath, settings):
    command = [
        'python', '../cli.py', filepath,
        '--agent-aggressiveness', str(settings.get('agentAggressiveness', 5)),
        '--evidence-threshold', str(settings.get('evidenceThreshold', 0.8))
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding='utf-8',
        cwd=os.path.dirname(os.path.abspath(__file__))  # Run from backend directory
    )

    for line in process.stdout:
        line = line.strip()
        if line:
            # This part is fine and streams stdout correctly
            socketio.emit('status_update', {'data': line})
            socketio.sleep(0)

    process.wait()
    # CORRECTED: Properly handle and send stderr output as JSON
    if process.returncode != 0:
        error_output = process.stderr.read()
        print(f"CLI Error: {error_output}") # Log error on the server
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
    browser_info = ensure_remote_browser_service()
    if browser_info is None:
        print("[SERVER DEBUG] Remote browser failed, falling back to local browser", flush=True)
        emit_json_message({
            'type': 'WARNING',
            'message': 'Remote browser unavailable, using local browser fallback'
        })
        # Don't set CDP environment variables - let main.py use local browser
        # Continue with process launch without remote browser config
        browser_info = {}  # Empty dict to signal local browser mode
    else:
        print(f"[SERVER DEBUG] Browser info received: {browser_info}", flush=True)

    command = [
        'python', 'main.py', '--url', url,
        # Note: main.py doesn't use these settings yet, but we pass them for future use
        '--agent-aggressiveness', str(settings.get('agentAggressiveness', 5)),
        '--evidence-threshold', str(settings.get('evidenceThreshold', 0.8))
    ]

    # Set environment to suppress browser-use logs
    env = os.environ.copy()
    env['SUPPRESS_LOGS'] = 'true'

    # Only set remote browser environment variables if we successfully got browser info
    if browser_info:
        cdp_url = browser_info.get('cdp_url')
        cdp_ws = browser_info.get('cdp_websocket')
        if cdp_url:
            env['REMOTE_BROWSER_CDP_URL'] = cdp_url
        if cdp_ws:
            env['REMOTE_BROWSER_CDP_WS'] = cdp_ws
        novnc_url = browser_info.get('novnc_url')
        if novnc_url:
            env['REMOTE_BROWSER_NOVNC_URL'] = novnc_url
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
    # Only emit if we have actual remote browser info
    if browser_info and browser_info.get('cdp_url'):
        print(f"[SERVER DEBUG] Re-emitting BROWSER_ADDRESS to frontend: {browser_info}", flush=True)
        socketio.emit('status_update', {'data': json.dumps(browser_info)})
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
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']

    settings = {
        'agentAggressiveness': request.form.get('agentAggressiveness', 5),
        'evidenceThreshold': request.form.get('evidenceThreshold', 0.8)
    }

    if file and file.filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        socketio.start_background_task(run_script_and_stream_output, filepath, settings)
        return {'message': 'Processing started.'}, 202
    return 'No selected file', 400


@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """
    Cleanup endpoint: Kill all running main.py processes and reset browser state
    Called when user refreshes or clicks logo
    """
    print("[SERVER DEBUG] ========== /api/cleanup ENDPOINT HIT ==========", flush=True)
    try:
        # 1. Kill all main.py processes
        result = subprocess.run(['pkill', '-9', '-f', 'main.py'],
                              capture_output=True,
                              text=True,
                              timeout=3)
        print(f"[SERVER DEBUG] Killed main.py processes, return code: {result.returncode}", flush=True)

        # 2. Clear any tracked processes
        num_tracked = len(active_processes)
        active_processes.clear()
        print(f"[SERVER DEBUG] Cleared {num_tracked} tracked processes", flush=True)

        # 3. Reset browser session (close all tabs)
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
    socketio.run(app, port=5000, debug=True)

