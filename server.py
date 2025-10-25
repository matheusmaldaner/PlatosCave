# PlatosCave/server.py
import os
import subprocess
import json # Make sure json is imported
from flask import Flask, request
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'papers'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def run_script_and_stream_output(filepath, settings):
    command = [
        'python', 'cli.py', filepath,
        '--agent-aggressiveness', str(settings.get('agentAggressiveness', 5)),
        '--evidence-threshold', str(settings.get('evidenceThreshold', 0.8))
    ]
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding='utf-8'
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

@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)

