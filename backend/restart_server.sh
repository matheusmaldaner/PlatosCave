#!/bin/bash
cd /home/matheus/projects/PlatosCave/backend
pkill -9 -f "server.py"
sleep 2
python server.py > /tmp/server.log 2>&1 &
echo "Server restarted with PID: $!"
sleep 3
echo "Server log:"
tail -15 /tmp/server.log
