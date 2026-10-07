Reduced K30 study trace: CENTRAL-EUROPE-7x7 at 15-minute time steps (timestep granularity 4, 100 flights), 10 kept
steps. `trace.jsonl` and `encoding.lp` are copied unchanged, `run.json` with `data_dir` cut to the instance name;
every `lp/iter_000NN_0.lp` is reduced to its `paths(`, `config(` and `chosen_path(` lines (what xai/reasons.py reads
from an instance), so these lp files cannot be solved. `keep_contrasts.jsonl` was computed by xai/keep.py from the
full lp files of the trace (study trace of 2026-10-07); the row reasons of test_row_reasons.py read it.
