# Controlling ASPaeroFlow from the XAI interface

Two ways to connect the optimizer (`01_ASPaeroFlow`) to the clinguin interface (`../ASPaeroFlow-XAI`):

| | `controller_mockup_2.py` (June 2026 study) | `session_service.py` (prototype, branch `xai`) |
|---|---|---|
| processes | controller + optimizer + one optimizer process per explanation | one service process |
| transport | ZeroMQ, 6 ports, lockstep handshake | HTTP on one port, server-sent events |
| per iteration to the browser | all matrices, base64 (megabytes) | a summary of about 1 KB |
| stepping | free run, pause between iterations | one step at a time, or run with a pace |
| explanations | the iteration re-run with one forced fact, six global metrics | the iteration's ASP sub-problem solved again in-process: answer, cost ladder per priority level, ties, scope; what-if with user locks |
| clinguin backend | `ATFCMBackend` | `ATFCMSessionBackend` |

## Session service

    python 07_heuristic_controller/session_service.py --instances-root DIR --sessions-root DIR [--port 8090]

`--instances-root` is a folder of instance folders (each with `flights.csv`, ...). Every live session
writes an iteration trace into `--sessions-root/<id>/` (see `01_ASPaeroFlow/src/aspaeroflow/xai/trace.py`);
a trace can be replayed later with `POST /sessions {"replay": "<trace folder>"}`, without an optimizer.
Endpoints and event format: docstring of `session_service.py`. Quick check with curl:

    curl -s localhost:8090/instances
    curl -s -X POST localhost:8090/sessions -H 'Content-Type: application/json' -d '{"instance": "CENTRAL-EUROPE-7x7", "options": {"max_number_sectors": 100000}}'
    curl -s -X POST localhost:8090/sessions/<id>/step
    curl -s -X POST localhost:8090/sessions/<id>/iterations/1/explain -H 'Content-Type: application/json' -d '{"question": "sectors"}'
    curl -s -N localhost:8090/sessions/<id>/events

The same explanations without any server, on a trace written by `main.py --xai-trace-dir=T`:

    python 01_ASPaeroFlow/xai_explain.py T --iteration 6 --hotspot --flight 18 --sectors --alternatives
    python 01_ASPaeroFlow/xai_explain.py T --iteration 6 --what-if "keep 18" "avoid 22"

Tests: `python -m unittest discover -s 01_ASPaeroFlow/xai_tests` (trace, explanations, service, exposure).
