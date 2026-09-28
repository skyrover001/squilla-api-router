#!/usr/bin/env bash
# Squilla API Router - macOS background starter.
#
# Usage:
#   ./start_router_mac.sh                  # default port 8021
#   ./start_router_mac.sh 8022             # custom port
#   PORT=8023 ./start_router_mac.sh        # custom via env
#
# The service runs detached with nohup, so closing the terminal does NOT
# stop it.  Logs:
#   stdout -> uvicorn_<port>.log, stderr -> uvicorn_<port>.err.log
# To stop: kill $(cat squilla_router_<port>.pid)
#
# NOTE: macOS has no `seq`; a while loop is used below.

set -euo pipefail

PORT="${1:-${PORT:-8021}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"   # directory that CONTAINS squilla_api_router/
PYTHON="${PYTHON:-python3}"

# Run from the repo itself; macOS handles non-ASCII paths fine.
# If your filesystem does not, point RUN_DIR at an ASCII copy of the repo.
RUN_DIR="${SQUILLA_RUN_DIR:-$SCRIPT_DIR}"

cd "$RUN_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export SQUILLA_REPO="$ROOT_DIR"

# 1) Stop any previous instance on this port (macOS: lsof).
if [ -f "$RUN_DIR/squilla_router_$PORT.pid" ]; then
    OLD_PID="$(cat "$RUN_DIR/squilla_router_$PORT.pid" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping previous instance (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$RUN_DIR/squilla_router_$PORT.pid"
fi
_PORT_PID="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$_PORT_PID" ]; then
    echo "Stopping port-$PORT process (PID $_PORT_PID)..."
    kill "$_PORT_PID" 2>/dev/null || true
    sleep 2
fi

# 2) Start detached.
nohup "$PYTHON" "$SCRIPT_DIR/_run_server.py" --host 127.0.0.1 --port "$PORT" \
    > "$RUN_DIR/uvicorn_$PORT.log" \
    2> "$RUN_DIR/uvicorn_$PORT.err.log" &
NEW_PID=$!
echo "$NEW_PID" > "$RUN_DIR/squilla_router_$PORT.pid"
disown || true

# 3) Wait for health (while loop, macOS-compatible).
ok=false
i=0
while [ "$i" -lt 30 ]; do
    sleep 2
    if curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" > /tmp/squilla_health.json 2>/dev/null; then
        ok=true
        break
    fi
    i=$((i + 1))
done

if [ "$ok" = true ]; then
    echo "Squilla Router is running:"
    echo "  PID:   $NEW_PID"
    echo "  URL:   http://127.0.0.1:$PORT/v1"
    echo "  ML:    $(cat /tmp/squilla_health.json)"
    echo "  Logs:  $RUN_DIR/uvicorn_$PORT.log / .err.log"
    echo "  Stop:  kill \$(cat $RUN_DIR/squilla_router_$PORT.pid)"
else
    echo "FAILED to start. Check $RUN_DIR/uvicorn_$PORT.err.log"
    exit 1
fi
