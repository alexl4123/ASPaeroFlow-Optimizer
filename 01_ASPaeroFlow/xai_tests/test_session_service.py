"""The HTTP session service (07_heuristic_controller/session_service.py) on the 10-flight fixture.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import importlib.util
import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURES = HERE / "fixtures"
INSTANCE = "EAST-ASIA-3x3-V2_0000010_SEED150699"

spec = importlib.util.spec_from_file_location("session_service", REPO / "07_heuristic_controller" / "session_service.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class _Response:
    def __init__(self, status_code, body):
        self.status_code, self.text = status_code, body

    def json(self):
        return json.loads(self.text)


class _Client:
    """Just enough of an HTTP client (standard library only) for these tests."""

    def __init__(self, base):
        self.base = base

    def _call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return _Response(r.status, r.read().decode())
        except urllib.error.HTTPError as e:
            return _Response(e.code, e.read().decode())

    def get(self, path):
        return self._call("GET", path)

    def post(self, path, json=None):
        return self._call("POST", path, json if json is not None else {})


class SessionService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        config = uvicorn.Config(service.create_app(FIXTURES, Path(cls.tmp.name)), host="127.0.0.1",
                                port=port, log_level="warning")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                urllib.request.urlopen(cls.base + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        cls.client = _Client(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.tmp.cleanup()

    def new(self, **body):
        r = self.client.post("/sessions", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["id"]

    def test_instances_and_graph(self):
        self.assertIn(INSTANCE, self.client.get("/instances").json())
        sid = self.new(instance=INSTANCE)
        graph = self.client.get(f"/sessions/{sid}/graph").json()
        self.assertGreater(len(graph["vertices"]), 0)
        self.assertGreater(len(graph["edges"]), 0)

    def test_step_until_finished_then_explain(self):
        sid = self.new(instance=INSTANCE)
        summaries = []
        for _ in range(50):
            r = self.client.post(f"/sessions/{sid}/step").json()
            if r["iteration"] is not None:
                summaries.append(r["iteration"])
            if r["status"] == "finished":
                break
        self.assertEqual(self.client.get(f"/sessions/{sid}").json()["status"], "finished")
        self.assertEqual(summaries[-1]["objectives"]["OVERLOAD"], 0)
        n = summaries[0]["iteration"]
        for question in ("hotspot", "sectors", "tie", "alternatives"):
            r = self.client.post(f"/sessions/{sid}/iterations/{n}/explain", json={"question": question})
            self.assertEqual(r.status_code, 200, r.text)
        f = summaries[0]["decision_flights"][0]
        r = self.client.post(f"/sessions/{sid}/iterations/{n}/explain", json={"question": "flight", "flight": f})
        self.assertIn("answer", r.json())
        r = self.client.post(f"/sessions/{sid}/iterations/{n}/what-if", json={"locks": [f"keep {f}"]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("feasible", r.json())
        self.assertEqual(self.client.post(f"/sessions/{sid}/iterations/999/explain",
                                          json={"question": "hotspot"}).status_code, 404)

    def test_run_in_background_and_replay(self):
        sid = self.new(instance=INSTANCE)
        self.client.post(f"/sessions/{sid}/run", json={"pace_ms": 0})
        for _ in range(600):
            if self.client.get(f"/sessions/{sid}").json()["status"] == "finished":
                break
            time.sleep(0.1)
        live = self.client.get(f"/sessions/{sid}/iterations").json()
        self.assertGreater(len(live), 0)
        replay = self.new(replay=str(Path(self.tmp.name) / sid))
        replayed = [self.client.post(f"/sessions/{replay}/step").json()["iteration"] for _ in live]
        self.assertEqual([r["objectives"] for r in replayed], [r["objectives"] for r in live])

    def test_event_stream_is_numbered_and_resumable(self):
        sid = self.new(instance=INSTANCE)
        self.client.post(f"/sessions/{sid}/step")
        self.client.post(f"/sessions/{sid}/step")

        def read_events(headers, count):
            req = urllib.request.Request(f"{self.base}/sessions/{sid}/events", headers=headers)
            events = []
            with urllib.request.urlopen(req, timeout=30) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
                        if len(events) == count:
                            break
            return events

        first = read_events({}, 2)
        self.assertEqual([e["seq"] for e in first], [1, 2])
        resumed = read_events({"Last-Event-ID": "1"}, 1)
        self.assertEqual(resumed[0]["seq"], 2)
        self.assertEqual(resumed[0], first[1])


if __name__ == "__main__":
    unittest.main()
