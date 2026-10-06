Reduced June study trace: CENTRAL-EUROPE-7x7, 13 kept steps (the replay of the June 2026 study run).
`trace.jsonl`, `run.json` and `encoding.lp` are copied unchanged; every `lp/iter_000NN_0.lp` is reduced to its
`paths(`, `config(` and `chosen_path(` lines (what xai/reasons.py reads from an instance). The June trace has no
`hotspot.flights`/`hotspot.taken` and no `evaluation_window`; tests add them where needed and say so.
`step06_facts.lp` holds the reasons.lp input facts of step 6 (hand-reviewed), `step06_expected.lp` the shown atoms
of reasons.lp + reasons_text.lp on them.
