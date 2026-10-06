"""The 10-flight ASPaeroFlow runs (plain and with the XAI trace) that several test modules read.

get() runs both on its first call and caches (plain stdout rows, traced stdout rows, trace folder) for the
rest of the interpreter; the temporary folder is removed at interpreter exit (atexit) and by nobody else.
Each test module still works alone: its first get() runs the optimizer.
"""
import atexit
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
REPO = OPT.parent
INSTANCE = HERE / "fixtures" / "EAST-ASIA-3x3-V2_0000010_SEED150699"

_cache = None


def run(extra):
    """main.py on the 10-flight instance; the JSON lines it prints."""
    cmd = [sys.executable, str(OPT / "main.py"), f"--data-dir={INSTANCE}",
           f"--encoding-path={OPT / 'encoding.lp'}", "--save-results=false", *extra]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def get():
    """(plain rows, traced rows, trace folder) of the two runs, computed once."""
    global _cache
    if _cache is None:
        tmp = Path(tempfile.mkdtemp(prefix="aspaeroflow-xai-test-"))
        atexit.register(shutil.rmtree, tmp, True)
        trace_dir = tmp / "trace"
        plain = run([])
        traced = run([f"--xai-trace-dir={trace_dir}"])
        _cache = (plain, traced, trace_dir)
    return _cache
