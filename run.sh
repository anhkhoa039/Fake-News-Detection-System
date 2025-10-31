#!/usr/bin/env bash
set -e

# Simple runner: starts the app, then exposes it via ngrok on port 5001

cd "$(dirname "$0")"

python src/app.py &
APP_PID=$!
echo "[INFO] App started (PID $APP_PID) at http://127.0.0.1:5001"

if command -v ngrok >/dev/null 2>&1; then
  echo "[INFO] Starting ngrok tunnel on port 5001"
  ngrok http 5001
else
  echo "[WARN] ngrok not found. Install from https://ngrok.com/download"
  wait $APP_PID
fi


