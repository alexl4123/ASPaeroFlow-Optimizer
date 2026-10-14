Practice study trace CE-NAVPT-PRACTICE_S15_2 (real-navpoint instance, 15 flights, timestep granularity 4), 3 kept
steps, every step with later flights of the same aircraft (step 2: 10 after 3, 8 after 4; step 3: 14 after 11).
`trace.jsonl`, `encoding.lp` and `lp/iter_00002_0.lp`, `lp/iter_00003_0.lp` are copied unchanged, so steps 2 and 3
can be solved again; step 1's instance (220 KB) is left out, so step 1 cannot be explained. `run.json` has `data_dir`
cut to the instance name. `CE-NAVPT-PRACTICE/` holds the instance's `airports.csv` and `mappings/vertex_map.csv`
(copied unchanged), what xai/session.py display_names needs for the fix and airport names.
Used by test_avoid_per_flight.py (S8 4a, per-flight "avoid navpoint").
