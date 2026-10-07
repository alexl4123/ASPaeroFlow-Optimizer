#!/usr/bin/env bash
# Start the XAI interface on this machine without docker: the session service (port 8090), the
# clinguin backend (8000) and the Angular dev server (4200, which passes /api to 8000).
# Ctrl-C stops all three. Logs go to $XAI_LOGS (default /tmp/xai_logs).
#
#   XAI_INSTANCES=<folder of instance folders> XAI_INSTANCE=<one of them> \
#   OPT_PY=<python of the optimizer env> XAI_PY=<python of the ASPaeroFlow-XAI env> \
#       07_heuristic_controller/run_xai_local.sh
#
# Optional: XAI_REPLAY=<trace folder> (replay instead of a live run), NODE=<node binary, >= 20.19>,
# XAI_REPO=<path of ASPaeroFlow-XAI> (default: next to this repository), XAI_SESSIONS (default /tmp/xai_sessions),
# XAI_OPTIONS=<JSON> (optimizer options of live sessions, passed to the clinguin backend as ASPAEROFLOW_OPTIONS;
# default the June 2026 study's options), e.g.
#   XAI_OPTIONS='{"max_number_sectors": 100000, "timestep_granularity": 4, "max_delay_per_iteration": 9, "seed": 11904657}'
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OPT=$(dirname "$HERE")
XAI=${XAI_REPO:-$(dirname "$OPT")/ASPaeroFlow-XAI}
OPT_PY=${OPT_PY:-python}
XAI_PY=${XAI_PY:-python}
NODE=${NODE:-node}
INSTANCES=${XAI_INSTANCES:?set XAI_INSTANCES to a folder that contains instance folders}
INSTANCE=${XAI_INSTANCE:-}
REPLAY=${XAI_REPLAY:-}
SESSIONS=${XAI_SESSIONS:-/tmp/xai_sessions}
LOGS=${XAI_LOGS:-/tmp/xai_logs}
OPTIONS=${XAI_OPTIONS:-}
if [ -n "$OPTIONS" ] && ! "$OPT_PY" -c 'import json, sys; assert isinstance(json.loads(sys.argv[1]), dict)' "$OPTIONS" 2>/dev/null; then
  echo "XAI_OPTIONS is not a JSON object: $OPTIONS" >&2
  exit 1
fi
mkdir -p "$LOGS" "$SESSIONS"

node_version=$("$NODE" -e 'const [a,b]=process.versions.node.split(".").map(Number); console.log(a*1000+b)')
if [ "$node_version" -lt 20019 ]; then
  echo "Angular CLI 20 needs Node >= 20.19; $NODE is $("$NODE" --version). Try: nvm install 22 && nvm use 22" >&2
  exit 1
fi
[ -d "$XAI/angular_frontend/node_modules" ] || { echo "run 'npm ci' in $XAI/angular_frontend first" >&2; exit 1; }

pids=()
stop() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap stop EXIT INT TERM

wait_for() {  # url, name, log
  for _ in $(seq 1 120); do
    if curl -s -o /dev/null "$1"; then return 0; fi
    sleep 1
  done
  echo "$2 did not start; see $3" >&2
  exit 1
}

(cd "$OPT" && exec "$OPT_PY" 07_heuristic_controller/session_service.py --host 127.0.0.1 --port 8090 \
    --instances-root "$INSTANCES" --sessions-root "$SESSIONS") > "$LOGS/service.log" 2>&1 &
pids+=($!)
wait_for http://127.0.0.1:8090/health "session service" "$LOGS/service.log"
echo "session service up (instances: $(curl -s http://127.0.0.1:8090/instances))"

(cd "$XAI" && CLINGUIN_HOST=127.0.0.1 ASPAEROFLOW_SERVICE_URL=http://127.0.0.1:8090 \
    ASPAEROFLOW_INSTANCE="$INSTANCE" ASPAEROFLOW_REPLAY="$REPLAY" ASPAEROFLOW_OPTIONS="$OPTIONS" TELEMETRY_DIR="$LOGS/telemetry" \
    exec "$XAI_PY" start.py server --backend=ATFCMSessionBackend --server-port 8000 \
    --domain-files atfcm_frontend/encoding.lp atfcm_frontend/instance.lp --ui-files atfcm_frontend/ui.lp) \
    > "$LOGS/clinguin.log" 2>&1 &
pids+=($!)
wait_for http://127.0.0.1:8000/health "clinguin backend" "$LOGS/clinguin.log"
echo "clinguin backend up"

(cd "$XAI/angular_frontend" && exec "$NODE" node_modules/@angular/cli/bin/ng.js serve --host 127.0.0.1 --port 4200) \
    > "$LOGS/angular.log" 2>&1 &
pids+=($!)
wait_for http://127.0.0.1:4200/ "Angular dev server" "$LOGS/angular.log"
echo
echo "Open http://localhost:4200   (logs: $LOGS; Ctrl-C stops everything)"
wait
