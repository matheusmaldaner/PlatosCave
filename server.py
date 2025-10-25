# PlatosCave/server.py
import os
import subprocess
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# 1. Setup
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allow requests from our frontend
socketio = SocketIO(app, cors_allowed_origins="*")

# Ensure a directory for uploads exists
UPLOAD_FOLDER = 'papers'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def run_script_and_stream_output(filepath):
    """
    Runs cli.py as a separate process and streams its output over a WebSocket.
    """
    command = ['python', 'cli.py', filepath]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Read the output line by line and send it to the frontend
    for line in process.stdout:
        line = line.strip()
        socketio.emit('status_update', {'data': line})
        socketio.sleep(0) # Yield to allow the message to send

    process.wait()
    if process.returncode != 0:
        error_output = process.stderr.read()
        socketio.emit('status_update', {'data': f'ERROR: {error_output}'})

# 2. API Endpoint for File Uploads
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Start the script in a background thread so the UI doesn't freeze
        socketio.start_background_task(run_script_and_stream_output, filepath)
        return {'message': 'File uploaded and processing started.'}, 202

# 3. WebSocket Connection Handler
@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')

# 4. Run the Server
if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)