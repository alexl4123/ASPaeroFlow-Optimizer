Reduced June study trace: CENTRAL-EUROPE-7x7, 13 kept steps (the replay of the June 2026 study run).
`trace.jsonl`, `run.json` and `encoding.lp` are copied unchanged; every `lp/iter_000NN_0.lp` is reduced to its
`paths(`, `config(` and `chosen_path(` lines (what xai/reasons.py reads from an instance). The June trace has no
`hotspot.flights`/`hotspot.taken` and no `evaluation_window`; tests add them where needed and say so.
`step06_facts.lp` holds the reasons.lp input facts of step 6 (hand-reviewed), `step06_expected.lp` the shown atoms
of reasons.lp + reasons_text.lp on them.
`keep_contrasts.jsonl` was computed by xai/keep.py from the full lp files of the June trace (the reduced ones here
cannot be solved); the row reasons of test_row_reasons.py read it.
`CENTRAL-EUROPE-7x7/` holds the instance's `flights.csv`, `airplane_flight_assignment.csv` and `airports.csv`
(copied unchanged), what xai/plan.py needs besides the trace; `run.json`'s `data_dir` points outside the repository,
so plan.py finds them as the sibling folder of that name.
