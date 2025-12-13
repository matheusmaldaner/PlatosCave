import os
import subprocess
import json
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
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=300,
    ping_interval=25
)

UPLOAD_FOLDER = 'papers'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BROWSER_COMPOSE_FILE = Path(__file__).parent / 'docker-compose.browser.yaml'

BROWSER_CDP_INTERNAL_URL = os.environ.get('REMOTE_BROWSER_CDP_INTERNAL_URL', 'http://localhost:9222')
BROWSER_NOVNC_INTERNAL_URL = os.environ.get('REMOTE_BROWSER_NOVNC_INTERNAL_URL', 'http://localhost:7900')

BROWSER_CDP_PUBLIC_URL = os.environ.get('REMOTE_BROWSER_CDP_PUBLIC_URL', 'http://localhost:9222')
BROWSER_NOVNC_PUBLIC_URL = os.environ.get('REMOTE_BROWSER_NOVNC_PUBLIC_URL', 'http://localhost:7900/vnc.html?autoconnect=1&resize=scale')

BROWSER_CDP_HEALTH_URL = f"{BROWSER_CDP_INTERNAL_URL.rstrip('/')}/json/version"
BROWSER_NOVNC_HEALTH_URL = BROWSER_NOVNC_INTERNAL_URL

active_processes = {}


def emit_json_message(payload: dict) -> None:
    print(f"[SERVER] Emitting: {payload.get('type', 'UNKNOWN')}", flush=True)
    socketio.emit('status_update', {'data': json.dumps(payload)})


def wait_for_http_ok(url: str, timeout: float = 30.0, interval: float = 1.5) -> bool:
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


def fetch_cdp_metadata(url: str, timeout: float = 5.0, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            with urllib_request.urlopen(url, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    raw = response.read().decode('utf-8')
                    return json.loads(raw)
        except ConnectionResetError as e:
            if attempt < retries - 1:
                wait_time = 0.5 * (2 ** attempt)
                print(f"[SERVER] Connection reset, retry in {wait_time}s", flush=True)
                time.sleep(wait_time)
            else:
                return None
        except (URLError, HTTPError, json.JSONDecodeError) as e:
            print(f"[SERVER] CDP fetch error: {e}", flush=True)
            return None
    return None


def close_all_browser_tabs() -> bool:
    print("[SERVER] Closing browser tabs", flush=True)
    try:
        list_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/list"
        with urllib_request.urlopen(list_url, timeout=5) as response:
            tabs = json.loads(response.read().decode('utf-8'))

        closed_count = 0
        for tab in tabs:
            if tab.get('type') == 'page':
                tab_id = tab.get('id')
                close_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/close/{tab_id}"
                try:
                    with urllib_request.urlopen(close_url, timeout=5) as close_response:
                        if 200 <= close_response.status < 300:
                            closed_count += 1
                except (URLError, HTTPError) as e:
                    print(f"[SERVER] Failed to close tab: {e}", flush=True)

        print(f"[SERVER] Closed {closed_count} tabs", flush=True)
        return True

    except Exception as e:
        print(f"[SERVER] Error closing tabs: {e}", flush=True)
        return False


def reset_browser_session() -> bool:
    print("[SERVER] Resetting browser", flush=True)

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            list_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/list"
            with urllib_request.urlopen(list_url, timeout=3) as response:
                tabs = json.loads(response.read().decode('utf-8'))

            pages = [tab for tab in tabs if tab.get('type') == 'page']
            blank_tabs = [tab for tab in pages if tab.get('url', '').startswith('about:blank')]
            non_blank_tabs = [tab for tab in pages if not tab.get('url', '').startswith('about:blank')]

            for tab in non_blank_tabs:
                tab_id = tab.get('id')
                close_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/close/{tab_id}"
                try:
                    req = urllib_request.Request(close_url, method='GET')
                    with urllib_request.urlopen(req, timeout=2):
                        pass
                except Exception as e:
                    print(f"[SERVER] Failed to close tab: {e}", flush=True)

            if len(blank_tabs) > 1:
                tabs_to_close = blank_tabs[1:]
                for tab in tabs_to_close:
                    tab_id = tab.get('id')
                    close_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/close/{tab_id}"
                    try:
                        req = urllib_request.Request(close_url, method='GET')
                        with urllib_request.urlopen(req, timeout=2):
                            pass
                    except Exception:
                        pass

            if len(blank_tabs) == 0:
                time.sleep(0.2)
                new_tab_url = f"{BROWSER_CDP_PUBLIC_URL.rstrip('/')}/json/new"
                req = urllib_request.Request(new_tab_url, method='PUT')
                with urllib_request.urlopen(req, timeout=3):
                    pass

            return True

        except ConnectionResetError as e:
            if attempt < max_attempts - 1:
                wait_time = 1.0 * (2 ** attempt)
                print(f"[SERVER] Connection reset, retry in {wait_time}s", flush=True)
                time.sleep(wait_time)
                continue
            else:
                return False
        except Exception as e:
            print(f"[SERVER] Reset error: {e}", flush=True)
            if attempt < max_attempts - 1:
                time.sleep(1.0)
                continue
            return False

    return False


def kill_process_safely(process, timeout: float = 5.0) -> bool:
    if process is None or process.poll() is not None:
        return True

    try:
        process.terminate()
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
            return True
    except Exception as e:
        print(f"[SERVER] Kill error: {e}", flush=True)
        return False


def ensure_remote_browser_service() -> Optional[dict]:
    print("[SERVER] Starting browser service", flush=True)
    emit_json_message({
        'type': 'UPDATE',
        'stage': 'Browser',
        'text': 'Starting browser...'
    })

    if not BROWSER_COMPOSE_FILE.exists():
        emit_json_message({
            'type': 'ERROR',
            'message': f'Browser config not found at {BROWSER_COMPOSE_FILE}'
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
            'message': f'Failed to start browser: {exc}'
        })
        return None

    if wait_for_http_ok(BROWSER_NOVNC_HEALTH_URL, timeout=5.0):
        print("[SERVER] noVNC ready", flush=True)

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
            'message': 'Browser CDP endpoint timeout'
        })
        return None

    ws_url = metadata.get('webSocketDebuggerUrl')
    if ws_url:
        parsed_ws = urlparse(ws_url)
        parsed_cdp_public = urlparse(BROWSER_CDP_PUBLIC_URL)

        public_hostname = parsed_cdp_public.hostname or 'localhost'
        public_port = parsed_cdp_public.port or 9222
        ws_scheme = 'wss' if parsed_cdp_public.scheme == 'https' else 'ws'
        ws_netloc = f"{public_hostname}:{public_port}"
        ws_url = urlunparse((ws_scheme, ws_netloc, parsed_ws.path, '', '', ''))

    browser_payload = {
        'type': 'BROWSER_ADDRESS',
        'novnc_url': BROWSER_NOVNC_PUBLIC_URL,
        'cdp_url': BROWSER_CDP_PUBLIC_URL,
        'cdp_websocket': ws_url
    }
    emit_json_message(browser_payload)

    emit_json_message({
        'type': 'UPDATE',
        'stage': 'Browser',
        'text': 'Browser ready'
    })

    return browser_payload


def run_script_and_stream_output(filepath, settings):
    print(f"[SERVER] Starting PDF analysis: {filepath}", flush=True)

    browser_info = ensure_remote_browser_service()
    if browser_info is None:
        browser_info = {}

    command = [
        'python', 'main.py', '--pdf', filepath,
        '--agent-aggressiveness', str(settings.get('agentAggressiveness', 5)),
        '--evidence-threshold', str(settings.get('evidenceThreshold', 0.8))
    ]

    env = os.environ.copy()
    env['SUPPRESS_LOGS'] = 'true'

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
        env.pop('REMOTE_BROWSER_CDP_URL', None)
        env.pop('REMOTE_BROWSER_CDP_WS', None)
        env.pop('REMOTE_BROWSER_NOVNC_URL', None)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding='utf-8',
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env
    )

    if browser_info and browser_info.get('cdp_url'):
        socketio.emit('status_update', {'data': json.dumps(browser_info)})
        socketio.sleep(0.1)

    for line in process.stdout:
        line = line.strip()
        if line:
            try:
                json.loads(line)
                socketio.emit('status_update', {'data': line})
                socketio.sleep(0)
            except json.JSONDecodeError:
                pass

    process.wait()

    if process.returncode != 0:
        error_output = process.stderr.read()
        print(f"PDF error: {error_output}")
        error_message = json.dumps({"type": "ERROR", "message": error_output})
        socketio.emit('status_update', {'data': error_message})


def run_url_analysis_and_stream_output(url, settings, session_id=None):
    print(f"[SERVER] Starting URL analysis: {url}", flush=True)

    try:
        result = subprocess.run(['pkill', '-9', '-f', 'main.py'],
                      capture_output=True,
                      text=True,
                      timeout=3)
        socketio.sleep(1.0)
    except:
        pass

    reset_browser_session()
    socketio.sleep(0.5)

    browser_info = ensure_remote_browser_service()
    if browser_info is None:
        emit_json_message({
            'type': 'WARNING',
            'message': 'Remote browser unavailable, using local browser'
        })
        browser_info = {}

    command = [
        'python', 'main.py', '--url', url,
        '--agent-aggressiveness', str(settings.get('agentAggressiveness', 5)),
        '--evidence-threshold', str(settings.get('evidenceThreshold', 0.8))
    ]

    env = os.environ.copy()
    env['SUPPRESS_LOGS'] = 'true'

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
        env.pop('REMOTE_BROWSER_CDP_URL', None)
        env.pop('REMOTE_BROWSER_CDP_WS', None)
        env.pop('REMOTE_BROWSER_NOVNC_URL', None)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding='utf-8',
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env
    )

    if session_id:
        active_processes[session_id] = process

    if browser_info and browser_info.get('cdp_url'):
        socketio.emit('status_update', {'data': json.dumps(browser_info)})
        socketio.sleep(0.1)

    for line in process.stdout:
        line = line.strip()
        if line:
            try:
                json.loads(line)
                socketio.emit('status_update', {'data': line})
                socketio.sleep(0)
            except json.JSONDecodeError:
                pass

    process.wait()

    if session_id and session_id in active_processes:
        del active_processes[session_id]

    if process.returncode != 0:
        error_output = process.stderr.read()
        print(f"URL error: {error_output}")
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
        if not file.filename.lower().endswith('.pdf'):
            return {'error': 'Only PDF files are supported'}, 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        socketio.start_background_task(run_script_and_stream_output, filepath, settings)
        return {'message': 'Processing started.'}, 202
    return 'No selected file', 400


@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    print("[SERVER] Cleanup endpoint hit", flush=True)
    try:
        reset_browser_session()
        return {'message': 'Cleanup completed'}, 200
    except Exception as e:
        print(f"[SERVER] Cleanup error: {e}", flush=True)
        return {'error': str(e)}, 500


@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    data = request.get_json()

    if 'url' not in data:
        return {'error': 'No URL provided'}, 400

    url = data['url']
    settings = {
        'agentAggressiveness': data.get('agentAggressiveness', 5),
        'evidenceThreshold': data.get('evidenceThreshold', 0.8)
    }

    session_id = data.get('sessionId', request.headers.get('X-Session-ID', None))

    socketio.start_background_task(run_url_analysis_and_stream_output, url, settings, session_id)
    return {'message': 'URL analysis started.', 'url': url}, 202


@socketio.on('connect')
def handle_connect():
    print(f'[SERVER] Client connected: {request.sid}', flush=True)


@socketio.on('disconnect')
def handle_disconnect():
    print(f'[SERVER] Client disconnected: {request.sid}', flush=True)

    if request.sid in active_processes:
        process = active_processes[request.sid]
        kill_process_safely(process)
        del active_processes[request.sid]

    reset_browser_session()


if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)