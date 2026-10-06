#!/usr/bin/env python3
"""HTTP + server-sent events in front of the XAI sessions of 01_ASPaeroFlow (prototype).

One process, one port. The optimizer runs in this process (one worker thread per session), so the
three-hop ZeroMQ lockstep of controller_mockup_2.py is not needed: every request gets a reply or an
error with a status code, and the event stream is a numbered log a client can resume from.

    python 07_heuristic_controller/session_service.py --instances-root DIR --sessions-root DIR [--port 8090]

Endpoints (JSON):
    GET  /health
    GET  /instances                                   folders under --instances-root
    POST /sessions        {"instance": name} | {"data_dir": path} | {"replay": trace folder [, "instance": name]}, "options": {...}
    GET  /sessions/{id}                               status, last objectives, final summary
    GET  /sessions/{id}/graph                         vertices (coordinates), edges, initial sectors
    POST /sessions/{id}/step                          one iteration, returns its summary
    POST /sessions/{id}/run   {"pace_ms": 0}          iterate in the background until paused or finished
    POST /sessions/{id}/pause
    GET  /sessions/{id}/iterations                    summaries so far
    GET  /sessions/{id}/iterations/{n}                full record (without the instance text)
    POST /sessions/{id}/iterations/{n}/explain        {"question": hotspot|flight|sectors|tie|alternatives, "flight": F}
    POST /sessions/{id}/iterations/{n}/what-if        {"locks": ["keep 18", "avoid 22", "max_delay 19 2", ...]}
    GET  /sessions/{id}/events?after=SEQ              text/event-stream; "id:" = sequence number (Last-Event-ID works)
Events: {"v": 1, "seq": n, "type": "state" | "iteration" | "finished" | "error", "data": {...}}, each a few KB.
"""

import argparse
import asyncio
import json
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

OPTIMIZER_DIR = Path(__file__).resolve().parents[1] / "01_ASPaeroFlow"
sys.path.insert(0, str(OPTIMIZER_DIR))

from src.aspaeroflow.xai.session import OptimizerSession, ReplaySession  # noqa: E402

PROTOCOL_VERSION = 1


class EventLog:
    """Numbered events of one session; readers ask for everything after the last number they saw."""

    def __init__(self):
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, kind: str, data: Any) -> int:
        with self._lock:
            seq = len(self._events) + 1
            self._events.append({"v": PROTOCOL_VERSION, "seq": seq, "type": kind, "data": data})
            return seq

    def after(self, seq: int) -> List[Dict[str, Any]]:
        with self._lock:
            return self._events[seq:]


class Handle:
    """A session with its event log and its background worker."""

    def __init__(self, session, kind: str):
        self.session = session
        self.kind = kind
        self.events = EventLog()
        self.worker: Optional[threading.Thread] = None
        self.stop = threading.Event()
        self.pace = 0.0
        self.last: Optional[Dict[str, Any]] = None

    def state(self) -> Dict[str, Any]:
        running = self.worker is not None and self.worker.is_alive()
        return {"status": "running" if running else self.session.status, "kind": self.kind,
                "iterations": len(self.session.records) if self.kind == "live" else self.session.cursor,
                "last": self.last, "final": self.session.final, "initial": self.session.initial}

    def step(self) -> Optional[Dict[str, Any]]:
        summary = self.session.step()
        if summary is not None:
            self.last = summary
            self.events.append("iteration", summary)
        if self.session.status == "finished":
            self.events.append("finished", self.session.final)
        return summary

    def run(self) -> None:
        try:
            while not self.stop.is_set() and self.session.status != "finished":
                self.step()
                if self.pace:
                    self.stop.wait(self.pace)
        except Exception as exc:  # the worker must report, not die silently
            self.events.append("error", {"message": str(exc), "trace": traceback.format_exc(limit=5)})
        finally:
            self.events.append("state", self.state() | {"status": self.session.status})


class NewSession(BaseModel):
    instance: Optional[str] = None
    data_dir: Optional[str] = None
    replay: Optional[str] = None
    options: Dict[str, Any] = {}


class RunRequest(BaseModel):
    pace_ms: int = 0


class ExplainRequest(BaseModel):
    question: str
    flight: Optional[int] = None


class WhatIfRequest(BaseModel):
    locks: List[str]


def create_app(instances_root: Optional[Path], sessions_root: Path) -> FastAPI:
    app = FastAPI(title="ASPaeroFlow XAI sessions", version=str(PROTOCOL_VERSION))
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    sessions: Dict[str, Handle] = {}
    sessions_root.mkdir(parents=True, exist_ok=True)

    def get(sid: str) -> Handle:
        if sid not in sessions:
            raise HTTPException(404, f"no session {sid}")
        return sessions[sid]

    def busy(h: Handle) -> bool:
        return h.worker is not None and h.worker.is_alive()

    @app.get("/health")
    def health():
        return {"ok": True, "protocol": PROTOCOL_VERSION, "sessions": len(sessions)}

    @app.get("/instances")
    def instances():
        if instances_root is None or not instances_root.exists():
            return []
        return sorted(p.name for p in instances_root.iterdir() if (p / "flights.csv").exists())

    @app.post("/sessions")
    def new_session(req: NewSession):
        sid = uuid.uuid4().hex[:8]
        if req.replay:
            folder = Path(req.replay)
            if not (folder / "trace.jsonl").exists():
                raise HTTPException(400, f"not a trace folder: {folder}")
            data_dir = None
            if req.instance:          # the instance the trace was made from, under --instances-root
                if instances_root is None:
                    raise HTTPException(400, "the service was started without --instances-root")
                data_dir = instances_root / req.instance
            handle = Handle(ReplaySession(folder, data_dir), "replay")
        else:
            if req.instance:
                if instances_root is None:
                    raise HTTPException(400, "the service was started without --instances-root")
                data_dir = instances_root / req.instance
            elif req.data_dir:
                data_dir = Path(req.data_dir)
            else:
                raise HTTPException(400, "give one of instance, data_dir, replay")
            if not (data_dir / "flights.csv").exists():
                raise HTTPException(400, f"not an instance folder: {data_dir}")
            handle = Handle(OptimizerSession(data_dir, sessions_root / sid, req.options), "live")
        begun = handle.session.begin()
        sessions[sid] = handle
        handle.events.append("state", handle.state())
        return {"id": sid} | begun

    @app.get("/sessions/{sid}")
    def session_state(sid: str):
        return get(sid).state()

    @app.delete("/sessions/{sid}")
    def delete_session(sid: str):
        h = get(sid)
        h.stop.set()
        del sessions[sid]
        return {"deleted": sid}

    @app.get("/sessions/{sid}/graph")
    def graph(sid: str):
        return get(sid).session.graph()

    @app.post("/sessions/{sid}/step")
    def step(sid: str):
        h = get(sid)
        if busy(h):
            raise HTTPException(409, "the session is running; pause it first")
        summary = h.step()
        return {"iteration": summary, "status": h.session.status}

    @app.post("/sessions/{sid}/run")
    def run(sid: str, req: RunRequest):
        h = get(sid)
        if busy(h):
            return h.state()
        h.stop.clear()
        h.pace = max(0, req.pace_ms) / 1000.0
        h.worker = threading.Thread(target=h.run, daemon=True)
        h.worker.start()
        h.events.append("state", h.state())
        return h.state()

    @app.post("/sessions/{sid}/pause")
    def pause(sid: str):
        h = get(sid)
        h.stop.set()
        if h.worker is not None:
            h.worker.join(timeout=60)       # returns after the iteration in progress
        return h.state()

    @app.get("/sessions/{sid}/iterations")
    def iterations(sid: str):
        h = get(sid)
        return [e["data"] for e in h.events.after(0) if e["type"] == "iteration"]

    @app.get("/sessions/{sid}/iterations/{n}")
    def iteration(sid: str, n: int):
        try:
            return get(sid).session.iteration(n)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/sessions/{sid}/iterations/{n}/explain")
    def explain(sid: str, n: int, req: ExplainRequest):
        started = time.time()
        try:
            result = get(sid).session.explain(n, req.question, req.flight)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return result | {"seconds": round(time.time() - started, 3)}

    @app.post("/sessions/{sid}/iterations/{n}/what-if")
    def what_if(sid: str, n: int, req: WhatIfRequest):
        started = time.time()
        try:
            result = get(sid).session.what_if(n, req.locks)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except (ValueError, IndexError) as exc:
            raise HTTPException(400, f"bad lock: {exc}")
        return result | {"seconds": round(time.time() - started, 3)}

    @app.get("/sessions/{sid}/events")
    async def events(sid: str, request: Request, after: int = 0):
        h = get(sid)
        last = request.headers.get("last-event-id")
        seq = int(last) if last and last.isdigit() else after

        async def stream():
            nonlocal seq
            while not await request.is_disconnected():
                batch = h.events.after(seq)
                for event in batch:
                    seq = event["seq"]
                    yield f"id: {seq}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"
                if not batch:
                    yield ": keep-alive\n\n"
                    await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main(argv=None) -> None:
    import uvicorn
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instances-root", type=Path, default=None)
    p.add_argument("--sessions-root", type=Path, default=Path("xai_sessions"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args(argv)
    uvicorn.run(create_app(args.instances_root, args.sessions_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
