from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO
import eventlet
eventlet.monkey_patch()


def create_app() -> Flask:
    app = Flask(__name__)
    # Allow all origins for dev (including Engine.IO polling preflights)
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
    return app


app = create_app()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet", logger=True, engineio_logger=True)


@app.before_request
def _log_request():
    print(f"[HTTP] {request.method} {request.path} origin={request.headers.get('Origin')} acrm={request.headers.get('Access-Control-Request-Method')} acrh={request.headers.get('Access-Control-Request-Headers')}")


@app.after_request
def _add_cors_and_log(resp):
    # Ensure CORS headers are present even if a view forgot to add them
    resp.headers.setdefault('Access-Control-Allow-Origin', '*')
    resp.headers.setdefault('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    if not resp.headers.get('Access-Control-Allow-Headers'):
        resp.headers['Access-Control-Allow-Headers'] = request.headers.get('Access-Control-Request-Headers', '*')
    print(f"[HTTP] resp {resp.status_code} {request.path} ACAO={resp.headers.get('Access-Control-Allow-Origin')} ACAM={resp.headers.get('Access-Control-Allow-Methods')} ACAH={resp.headers.get('Access-Control-Allow-Headers')}")
    return resp


@app.route("/api/upload", methods=["OPTIONS"])
@cross_origin(origins="*")
def upload_options():
    print("[HTTP] OPTIONS /api/upload (preflight)")
    return ('', 204)


@app.route("/api/upload", methods=["POST"])
@cross_origin(origins="*")
def upload():
    # Placeholder: accept file via agent-side service in your stack
    # Emit an initial status update so frontend can verify websockets
    print("[HTTP] POST /api/upload ok; emitting status_update")
    socketio.emit("status_update", {"data": '{"type":"UPDATE","stage":"Validate","text":"Received file"}'})
    return jsonify({"ok": True})


@app.route("/api/analyze-url", methods=["OPTIONS"])
@cross_origin(origins="*")
def analyze_url_options():
    print("[HTTP] OPTIONS /api/analyze-url (preflight)")
    return ('', 204)


@app.route("/api/analyze-url", methods=["POST"])
@cross_origin(origins="*")
def analyze_url():
    payload = request.get_json(silent=True) or {}
    url = payload.get("url")
    print("[HTTP] POST /api/analyze-url ok; emitting status_update")
    socketio.emit("status_update", {"data": '{"type":"UPDATE","stage":"Validate","text":"Received URL"}'})
    return jsonify({"ok": True, "url": url})


@socketio.on('connect')
def on_connect():
    print(f"[SOCKET] connect sid={request.sid} origin={request.headers.get('Origin')}")


@socketio.on('disconnect')
def on_disconnect():
    print(f"[SOCKET] disconnect sid={request.sid}")


if __name__ == "__main__":
    # Run dev server with eventlet
    socketio.run(app, host="0.0.0.0", port=5000)


