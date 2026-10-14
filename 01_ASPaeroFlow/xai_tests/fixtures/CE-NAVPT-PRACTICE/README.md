The practice study instance CE-NAVPT-PRACTICE (real-navpoint network of CE-NAVPT-TG4, 15 flights, timestep granularity
4; analysis/xai_study/study_instances/CE-NAVPT-PRACTICE, built 2026-10-12), every file copied unchanged except
`navgraph/` (coordinates for the map; the optimizer does not read it) and `README.txt`. With the study options
(--timestep-granularity=4 --max-number-sectors=100000 --seed=11904657 --max-delay-per-iteration=9 --number-threads=4)
the optimizer reproduces the study trace in `../NAVPT_PRACTICE_TRACE` (3 kept steps; step 2 splits sector 10 at time
step 35). Used by test_load.py (K08).
