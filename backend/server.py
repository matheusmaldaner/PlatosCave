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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

UPLOAD_FOLDER = 'papers'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BROWSER_COMPOSE_FILE = Path(__file__).parent / 'docker-compose.browser.yaml'
BROWSER_NOVNC_URL = os.environ.get('REMOTE_BROWSER_NOVNC_URL', 'http://localhost:7900/vnc.html?autoconnect=1&resize=scale')
BROWSER_CDP_HEALTH_URL = os.environ.get('REMOTE_BROWSER_CDP_HEALTH_URL', 'http://localhost:9222/json/version')
BROWSER_CDP_PUBLIC_URL = os.environ.get('REMOTE_BROWSER_CDP_URL', 'http://localhost:9222')


def emit_json_message(payload: dict) -> None:
    """Send a structured payload to the frontend over WebSocket."""
    socketio.emit('status_update', {'data': json.dumps(payload)})


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


def ensure_remote_browser_service() -> Optional[dict]:
    """Ensure the remote browser Docker service is running and healthy."""
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

    if not wait_for_http_ok(BROWSER_NOVNC_URL):
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

    # Normalise the advertised WebSocket endpoint so the host machine can reach it
    ws_url = metadata.get('webSocketDebuggerUrl')
    if ws_url:
        parsed_ws = urlparse(ws_url)
        parsed_cdp = urlparse(BROWSER_CDP_PUBLIC_URL)

        # Prefer externally configured hostname/port if provided
        hostname = parsed_cdp.hostname or parsed_ws.hostname
        port = parsed_cdp.port or parsed_ws.port

        # Use ws/wss scheme mirroring the public CDP URL (default to ws)
        scheme = 'wss' if parsed_cdp.scheme == 'https' else 'ws'

        netloc = f"{hostname}:{port}" if port else hostname
        ws_url = urlunparse((scheme, netloc, parsed_ws.path, '', '', ''))

    browser_payload = {
        'type': 'BROWSER_ADDRESS',
        'novnc_url': BROWSER_NOVNC_URL,
        'cdp_url': BROWSER_CDP_PUBLIC_URL,
        'cdp_websocket': ws_url
    }
    emit_json_message(browser_payload)

    emit_json_message({
        'type': 'UPDATE',
        'stage': 'Browser',
        'text': 'Remote browser is ready.'
    })

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


def run_url_analysis_and_stream_output(url, settings):
    """
    Run URL analysis using browser-use + DAG generation
    Streams real-time updates via WebSocket
    """
    browser_info = ensure_remote_browser_service()
    if browser_info is None:
        return

    command = [
        'python', 'main.py', '--url', url,
        # Note: main.py doesn't use these settings yet, but we pass them for future use
        '--agent-aggressiveness', str(settings.get('agentAggressiveness', 5)),
        '--evidence-threshold', str(settings.get('evidenceThreshold', 0.8))
    ]

    # Set environment to suppress browser-use logs
    env = os.environ.copy()
    env['SUPPRESS_LOGS'] = 'true'

    cdp_url = browser_info.get('cdp_url')
    cdp_ws = browser_info.get('cdp_websocket')
    if cdp_url:
        env['REMOTE_BROWSER_CDP_URL'] = cdp_url
    if cdp_ws:
        env['REMOTE_BROWSER_CDP_WS'] = cdp_ws
    novnc_url = browser_info.get('novnc_url')
    if novnc_url:
        env['REMOTE_BROWSER_NOVNC_URL'] = novnc_url

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
                pass

    process.wait()

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


@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    """
    Analyze a research paper from URL using browser-use + DAG generation
    """
    data = request.get_json()

    if 'url' not in data:
        return {'error': 'No URL provided'}, 400

    url = data['url']
    settings = {
        'agentAggressiveness': data.get('agentAggressiveness', 5),
        'evidenceThreshold': data.get('evidenceThreshold', 0.8)
    }

    # Start URL analysis in background and stream updates via WebSocket
    socketio.start_background_task(run_url_analysis_and_stream_output, url, settings)

    return {'message': 'URL analysis started.', 'url': url}, 202

@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)

