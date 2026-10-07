% reasons.lp: why one ASPaeroFlow step worked on its spot, what its sub-problem allowed and what changed.
%
% Runs in its own clingo Control, together with reasons_text.lp only, NEVER with encoding.lp (some input
% names resemble the encoding's). Plain stratified program: no choice rules, no optimisation, no
% integrity constraints. Exactly one model. A record that breaks a documented rule of the optimizer gives
% error/1 atoms (never an unsatisfiable program). xai/reasons.py writes the input facts and renders the text.
%
% Input (integers and constants only; objective keys overload, arrival_delay, sector_number, reroute,
% sector_diff, reconfig; W in before/after):
%   step(N)  kept(N) | rejected(N)
%   hotspot(N,S,T)  hotspot_load(N,S,T,Demand,Capacity)  hotspot_overload(N,S,T,O)  hotspot_navpoints(N,S,K)
%   occupants_recorded(N)  occupant(N,F,Duration,Rank)  taken_count(N,K)
%   current_span(N,F,D)   (last minus first time + 1 of the recorded current trajectory of occupant F)
%   param(N,P,V)    P in max_aircraft, additional_time_increase, max_delay_per_iteration,
%                   number_capacity_management_configs, failed_attempts_before
%   instance_read(N)  subproblem_flight(N,F)  later_leg(N,Leg,F)  layout_option(N,C)
%   offered_paths(N,F,P)  offered_starts(N,F,K)   (lp file: candidate paths of solver flight F, and the
%                   number of distinct start times among them; the latter only when the lp states start times)
%   offered_delay(N,F,Min,Max)   (lp file: smallest and largest start time of F minus its current departure,
%                   only when the lp states start times and the current departure is known)
%   flights_recorded(N)  changed_flight(N,F)
%   sector_record(N)  part_before(N,S,V)  part_after(N,P,V)
%   objective(N,W,K,V)  recorded_before(N,overload,V)
%   rejected_before(N,R) with hotspot(R,S,T) and param(R,P,V)
%   run_option(O,V) | option_assumed(O)   O in sequential_execution, minimize_number_sectors (true/false),
%                                         composite_sector_function (max/other),
%                                         flight_duration_rule (occupied_steps/other)
%   filed_layout_static  evaluation_window(W)  run_length(L)  run_kept(K)
%   start_overload(O)  start_sector_overload(S,O)
% Row reasons (K12; only with the keep contrasts of the step, xai/keep.py, written by reasons.py keep_facts):
%   keep_recorded(N)  subproblem_count(N,K)  factual_clingo(N,I,V)  recorded_clingo(N,I,V)  (clingo vectors, by
%   position)  factual_cost(N,L,V)  (by level L in overload, delay, sectors, changed, config: the priorities 10..6
%   of encoding.lp, read off the atoms)  flight_entry(N,F)  current_recorded(N,F)  chosen_checked(N,F)
%   chosen_current(N,F)  keep_offered(N,F)  keep_cost(N,F,L,V) | keep_infeasible(N,F)  keep_unproven(N,F)
%   keep_cost_mismatch(N,F)  keep_moves(N,F,G)  keep_other(N,F,G)  keep_layout_differs(N,F)
%   keep_both_flight(N,F,G)  keep_both_cost(N,F,L,V) | keep_both_infeasible(N,F)  keep_both_unproven(N,F)
%   keep_both_cost_mismatch(N,F)  alt_cost(N,F,L,V) | alt_infeasible(N,F)  alt_unproven(N,F)  alt_cost_mismatch(N,F)
%   layout_choice(N,C)  keep_layout_cost(N,L,V) | keep_layout_infeasible(N)  keep_layout_unproven(N)
%   keep_layout_cost_mismatch(N)  alt_layout_cost(N,L,V) | alt_layout_infeasible(N)  alt_layout_unproven(N)
%   alt_layout_cost_mismatch(N)  run_option(arrival_delay_metric,M) | option_assumed(arrival_delay_metric)
% Evidence facts (flights_recorded, sector_record, instance_read, occupants_recorded) guard every
% sentence that says "no ..." or "unchanged": a missing fact never produces such a sentence.
%
% Output (#show): reason/1, about/2, reason_kind/2, reason_class/2, reason_arg/4 (Key, I orders list items,
% scalars I = 0), reason_source/2, reason_template/2 (R, template replacing the one of its kind), show/4
% (R, audience, visible|detail|row, order), delta/5, error/1, row_error/1.
% Reason terms: step N: hotspot(N), retry(N), not_kept(N) (visible); rule(N), capacity(N), subproblem(N),
% legs(N), delays(N), layouts(N), changed(N), unchanged(N), layout_result(N), scope(N) (detail);
% filed plan: start, start_rule. Row reasons (level row, one line under a row of the change list):
% flight_reason(N,F) with about(R, flight(N,F)), layout_reason(N) with about(R, sector(N,S,T,before)).
% error/1 is about the header and the record; row_error/1 drops the row lines of the step only.

#defined step/1.             #defined kept/1.             #defined rejected/1.
#defined hotspot/3.          #defined hotspot_load/5.     #defined hotspot_overload/4.
#defined hotspot_navpoints/3.
#defined occupants_recorded/1. #defined occupant/4.       #defined taken_count/2.
#defined current_span/3.
#defined param/3.
#defined instance_read/1.    #defined subproblem_flight/2. #defined later_leg/3.  #defined layout_option/2.
#defined offered_paths/3.    #defined offered_starts/3.    #defined offered_delay/4.
#defined flights_recorded/1. #defined changed_flight/2.
#defined sector_record/1.    #defined part_before/3.      #defined part_after/3.
#defined objective/4.        #defined recorded_before/3.  #defined rejected_before/2.
#defined run_option/2.       #defined option_assumed/1.   #defined filed_layout_static/0.
#defined evaluation_window/1. #defined run_length/1.      #defined run_kept/1.
#defined start_overload/1.   #defined start_sector_overload/2.
#defined template/3.         #defined error_text/2.
#defined keep_recorded/1.    #defined subproblem_count/2. #defined factual_clingo/3.  #defined recorded_clingo/3.
#defined factual_cost/3.     #defined flight_entry/2.     #defined current_recorded/2. #defined chosen_checked/2.
#defined chosen_current/2.   #defined keep_offered/2.     #defined keep_cost/4.       #defined keep_infeasible/2.
#defined keep_unproven/2.    #defined keep_cost_mismatch/2. #defined keep_moves/3.    #defined keep_other/3.
#defined keep_layout_differs/2. #defined keep_both_flight/3. #defined keep_both_cost/4. #defined keep_both_infeasible/2.
#defined keep_both_unproven/2. #defined keep_both_cost_mismatch/2. #defined alt_cost/4. #defined alt_infeasible/2.
#defined alt_unproven/2.     #defined alt_cost_mismatch/2. #defined layout_choice/2.  #defined keep_layout_cost/3.
#defined keep_layout_infeasible/1. #defined keep_layout_unproven/1. #defined keep_layout_cost_mismatch/1.
#defined alt_layout_cost/3.  #defined alt_layout_infeasible/1. #defined alt_layout_unproven/1.
#defined alt_layout_cost_mismatch/1.

when(before;after).
objective_key(overload;arrival_delay;sector_number;reroute;sector_diff;reconfig).

% ------------------------------------------------------------------ run options
sequential :- run_option(sequential_execution,true).
merging_on :- run_option(minimize_number_sectors,true).
merging_off :- run_option(minimize_number_sectors,false).
merging_off :- option_assumed(minimize_number_sectors).
capacity_is_max :- run_option(composite_sector_function,max).
capacity_is_max :- option_assumed(composite_sector_function).
option_source(sequential, run_option(sequential_execution,V)) :- run_option(sequential_execution,V).
option_source(sequential, option_assumed(sequential_execution)) :- option_assumed(sequential_execution).
option_source(merging, run_option(minimize_number_sectors,V)) :- run_option(minimize_number_sectors,V).
option_source(merging, option_assumed(minimize_number_sectors)) :- option_assumed(minimize_number_sectors).
option_source(composite, run_option(composite_sector_function,V)) :- run_option(composite_sector_function,V).
option_source(composite, option_assumed(composite_sector_function)) :- option_assumed(composite_sector_function).
% Runs that record flight_duration_rule = occupied_steps sort the cell by the time steps each flight occupies,
% at setup and after every kept step (computation_helpers.flight_spans_contiguous). Runs that do not record
% the rule may hold a stored duration one time period short for flights re-planned in a kept step.
duration_exact :- run_option(flight_duration_rule,occupied_steps).
option_source(duration, run_option(flight_duration_rule,V)) :- run_option(flight_duration_rule,V).
option_source(duration, option_assumed(flight_duration_rule)) :- option_assumed(flight_duration_rule).

% ------------------------------------------------------------------ objectives and deltas
has_objective(N,W,K) :- objective(N,W,K,_).
has_before(N) :- objective(N,before,_,_).
delta(N,K,B,A,lower)  :- step(N), objective(N,before,K,B), objective(N,after,K,A), A < B.
delta(N,K,B,A,higher) :- step(N), objective(N,before,K,B), objective(N,after,K,A), A > B.
delta(N,K,B,A,same)   :- step(N), objective(N,before,K,B), objective(N,after,K,A), A = B.

% ------------------------------------------------------------------ checks
kept_by_rule(N) :- objective(N,before,overload,B), objective(N,after,overload,A), A < B.
overload_known(N) :- has_objective(N,before,overload), has_objective(N,after,overload).
error(kept_rule(N)) :- kept(N), overload_known(N), not kept_by_rule(N).
error(not_kept_rule(N)) :- rejected(N), objective(N,before,overload,B), objective(N,after,overload,A), A != B.
error(overload_mismatch(N)) :- step(N), hotspot(N,S,T), hotspot_load(N,S,T,D,C), hotspot_overload(N,S,T,O), O != D-C.
error(before_mismatch(N)) :- step(N), recorded_before(N,overload,R), objective(N,before,overload,B), R != B.
error(missing_previous(N)) :- step(N), not has_before(N).
error(missing_objective(N,before,K)) :- step(N), has_before(N), objective_key(K), not has_objective(N,before,K).
error(missing_objective(N,after,K)) :- step(N), objective_key(K), not has_objective(N,after,K).

error(not_unique(objective(N,W,K))) :- objective(N,W,K,_), #count{ V : objective(N,W,K,V) } > 1.
error(not_unique(hotspot(N))) :- hotspot(N,_,_), #count{ S,T : hotspot(N,S,T) } > 1.
error(not_unique(hotspot_load(N))) :- hotspot_load(N,_,_,_,_), #count{ S,T,D,C : hotspot_load(N,S,T,D,C) } > 1.
error(not_unique(hotspot_overload(N))) :- hotspot_overload(N,_,_,_), #count{ S,T,O : hotspot_overload(N,S,T,O) } > 1.
error(not_unique(hotspot_navpoints(N))) :- hotspot_navpoints(N,_,_), #count{ S,K : hotspot_navpoints(N,S,K) } > 1.
error(not_unique(taken_count(N))) :- taken_count(N,_), #count{ K : taken_count(N,K) } > 1.
error(not_unique(run_length)) :- run_length(_), #count{ L : run_length(L) } > 1.

% How many flights the code passes to the solver (iteration_step.py build_job, one processor):
% min(max_aircraft, max(2, overload)), and never more than the flights in the cell.
need(N,2) :- step(N), hotspot(N,S,T), hotspot_overload(N,S,T,O), O < 2.
need(N,O) :- step(N), hotspot(N,S,T), hotspot_overload(N,S,T,O), O >= 2.
limit(N,X) :- need(N,X).
limit(N,M) :- need(N,_), param(N,max_aircraft,M).
limit(N,D) :- need(N,_), hotspot(N,S,T), hotspot_load(N,S,T,D,_).
lower_limit(N,K) :- limit(N,K), limit(N,K2), K2 < K.
take(N,K) :- limit(N,K), not lower_limit(N,K).
cap(N,X) :- need(N,X), not param(N,max_aircraft,_).
cap(N,X) :- need(N,X), param(N,max_aircraft,M), X <= M.
cap(N,M) :- need(N,X), param(N,max_aircraft,M), M < X.
error(taken_rule(N)) :- take(N,K), taken_count(N,K2), K != K2.

% The first K flights of the cell by recorded duration (ties: lower flight number) are the solver's flights.
selected(N,F) :- occupant(N,F,_,I), taken_count(N,K), I <= K.
check_candidates(N) :- step(N), instance_read(N), occupants_recorded(N), taken_count(N,_).
error(candidate_rule(N)) :- check_candidates(N), selected(N,F), not subproblem_flight(N,F).
error(candidate_rule(N)) :- check_candidates(N), subproblem_flight(N,F), not selected(N,F).
in_subproblem(N,F) :- subproblem_flight(N,F).
in_subproblem(N,L) :- later_leg(N,L,_).
error(changed_outside(N,F)) :- instance_read(N), flights_recorded(N), changed_flight(N,F), not in_subproblem(N,F).
% With the rule recorded, the duration of every occupant with a recorded trajectory (the solver flights) equals
% the time periods of that trajectory.
error(duration_rule(N)) :- duration_exact, occupant(N,F,D,_), current_span(N,F,D2), D != D2.

% Departure options in the lp file (optimize_flights.py): every candidate route of a solver flight comes with
% W delays (W = max_delay_per_iteration), so its paths have W distinct start times; in sequential mode the
% loops stop after the first path, so a solver flight has one path and one start time.
error(delay_options(N)) :- not sequential, subproblem_flight(N,F), offered_starts(N,F,K),
                           param(N,max_delay_per_iteration,W), K != W.
% The delays offered (start time minus the current departure, offered_delay(N,F,Min,Max)) are the delay range of the
% step outside sequential mode, and 0 in sequential mode (k = 0, see delays(N) below).
error(delay_options(N)) :- not sequential, subproblem_flight(N,F), offered_delay(N,F,Min,Max), delay_range(N,Lo,Hi),
                           (Min,Max) != (Lo,Hi).
error(delay_options(N)) :- sequential, subproblem_flight(N,F), offered_starts(N,F,K), K != 1.
error(delay_options(N)) :- sequential, subproblem_flight(N,F), offered_paths(N,F,P), P != 1.

% Sector layout at the hotspot time before and after the step.
before_navpoint(N,V) :- part_before(N,_,V).
after_navpoint(N,V) :- part_after(N,_,V).
error(parts_mismatch(N)) :- sector_record(N), merging_off, before_navpoint(N,V), not after_navpoint(N,V).
error(parts_mismatch(N)) :- sector_record(N), merging_off, after_navpoint(N,V), not before_navpoint(N,V).
part(N,P) :- part_after(N,P,_).
split(N,S,T,Parts) :- sector_record(N), hotspot(N,S,T), Parts = #count{ P : part(N,P) }.
% A split chosen at time T is written from T to the end (optimize_flights.py), so it adds at most
% (Parts-1) sectors per time of the evaluation window from T on. Holds only without merging and with a
% filed layout that does not change over time.
split_bound(N,(Parts-1)*(W-T)) :- split(N,S,T,Parts), evaluation_window(W), T < W.
split_bound(N,0) :- split(N,S,T,Parts), evaluation_window(W), T >= W.
error(split_extent(N)) :- filed_layout_static, merging_off, split_bound(N,X),
                          objective(N,before,sector_number,B), objective(N,after,sector_number,A), A-B > X.

% Rejected attempts right before this step (same spot, counter of consecutive rejections).
retried(N) :- step(N), rejected_before(N,_).
error(retry_spot(N,R)) :- not sequential, rejected_before(N,R), hotspot(N,S,T), hotspot(R,S2,T2), (S,T) != (S2,T2).
error(retry_count(N)) :- not sequential, step(N), param(N,failed_attempts_before,F),
                         F != #count{ R : rejected_before(N,R) }.

% ------------------------------------------------------------------ reasons of a step
% hotspot(N): the spot this step worked on (visible).
reason(hotspot(N)) :- step(N), hotspot(N,S,T), hotspot_load(N,S,T,D,C).
reason_kind(hotspot(N),hotspot_rule) :- reason(hotspot(N)), not sequential, hotspot(N,S,T), hotspot_load(N,S,T,D,_), D != 1.
reason_kind(hotspot(N),hotspot_rule_one) :- reason(hotspot(N)), not sequential, hotspot(N,S,T), hotspot_load(N,S,T,1,_).
reason_kind(hotspot(N),hotspot) :- reason(hotspot(N)), sequential, hotspot(N,S,T), hotspot_load(N,S,T,D,_), D != 1.
reason_kind(hotspot(N),hotspot_one) :- reason(hotspot(N)), sequential, hotspot(N,S,T), hotspot_load(N,S,T,1,_).
reason_class(hotspot(N),capacity) :- reason(hotspot(N)).
reason_arg(hotspot(N),sector,0,S) :- reason(hotspot(N)), hotspot(N,S,T).
reason_arg(hotspot(N),time,0,T) :- reason(hotspot(N)), hotspot(N,S,T).
reason_arg(hotspot(N),demand,0,D) :- reason(hotspot(N)), hotspot(N,S,T), hotspot_load(N,S,T,D,_).
reason_arg(hotspot(N),capacity,0,C) :- reason(hotspot(N)), hotspot(N,S,T), hotspot_load(N,S,T,_,C).
reason_source(hotspot(N),hotspot(N,S,T)) :- reason(hotspot(N)), hotspot(N,S,T).
reason_source(hotspot(N),hotspot_load(N,S,T,D,C)) :- reason(hotspot(N)), hotspot(N,S,T), hotspot_load(N,S,T,D,C).
reason_source(hotspot(N),A) :- reason(hotspot(N)), option_source(sequential,A).
show(hotspot(N),developer,visible,1) :- reason(hotspot(N)).

% Delay range of a step: k*W .. (k+1)*W-1 (optimize_flights.py), k = additional_time_increase.
delay_range(N,K*W,(K+1)*W-1) :- step(N), param(N,additional_time_increase,K), param(N,max_delay_per_iteration,W).
% ... of the next attempt after a rejected step N: k+1.
next_range(N,(K+1)*W,(K+2)*W-1) :- rejected(N), param(N,additional_time_increase,K), param(N,max_delay_per_iteration,W).
% After 5 consecutive rejections only the current layout is offered, after 10 only one flight
% (evaluate_solution.py). For retry(N) the counter is failed_attempts_before of N, for not_kept(N) one more.
counter(retry(N),F) :- step(N), param(N,failed_attempts_before,F).
counter(not_kept(N),F+1) :- rejected(N), param(N,failed_attempts_before,F).

% retry(N): the rejected attempts at this spot right before this kept step (visible).
reason(retry(N)) :- kept(N), not sequential, retried(N), delay_range(N,_,_).
reason_kind(retry(N),retry) :- reason(retry(N)), #count{ R : rejected_before(N,R) } > 1.
reason_kind(retry(N),retry_one) :- reason(retry(N)), #count{ R : rejected_before(N,R) } = 1.
reason_class(retry(N),rule) :- reason(retry(N)).
reason_arg(retry(N),steps,I,R) :- reason(retry(N)), rejected_before(N,R), I = #count{ R2 : rejected_before(N,R2), R2 <= R }.
reason_arg(retry(N),lo,0,Lo) :- reason(retry(N)), delay_range(N,Lo,_).
reason_arg(retry(N),hi,0,Hi) :- reason(retry(N)), delay_range(N,_,Hi).
reason_source(retry(N),rejected_before(N,R)) :- reason(retry(N)), rejected_before(N,R).
reason_source(retry(N),param(N,P,V)) :- reason(retry(N)), param(N,P,V),
    P = (additional_time_increase;max_delay_per_iteration;failed_attempts_before).
show(retry(N),developer,visible,2) :- reason(retry(N)).

% not_kept(N): a rejected attempt (visible).
% The line names the attempt's own spot: it is shown under the title of the kept step before it, whose spot differs.
reason(not_kept(N)) :- rejected(N), objective(N,after,overload,_), hotspot(N,_,_), sequential.
reason(not_kept(N)) :- rejected(N), objective(N,after,overload,_), hotspot(N,_,_), next_range(N,_,_).
reason_kind(not_kept(N),not_kept) :- reason(not_kept(N)), not sequential, next_range(N,_,_).
reason_kind(not_kept(N),not_kept_sequential) :- reason(not_kept(N)), sequential.
reason_class(not_kept(N),outcome) :- reason(not_kept(N)).
reason_arg(not_kept(N),iteration,0,N) :- reason(not_kept(N)).
reason_arg(not_kept(N),sector,0,S) :- reason(not_kept(N)), hotspot(N,S,T).
reason_arg(not_kept(N),time,0,T) :- reason(not_kept(N)), hotspot(N,S,T).
reason_arg(not_kept(N),overload,0,A) :- reason(not_kept(N)), objective(N,after,overload,A).
reason_arg(not_kept(N),lo,0,Lo) :- reason(not_kept(N)), not sequential, next_range(N,Lo,_).
reason_arg(not_kept(N),hi,0,Hi) :- reason(not_kept(N)), not sequential, next_range(N,_,Hi).
reason_source(not_kept(N),rejected(N)) :- reason(not_kept(N)).
reason_source(not_kept(N),hotspot(N,S,T)) :- reason(not_kept(N)), hotspot(N,S,T).
reason_source(not_kept(N),objective(N,after,overload,A)) :- reason(not_kept(N)), objective(N,after,overload,A).
reason_source(not_kept(N),param(N,P,V)) :- reason(not_kept(N)), not sequential, param(N,P,V),
    P = (additional_time_increase;max_delay_per_iteration;failed_attempts_before).
reason_source(not_kept(N),A) :- reason(not_kept(N)), option_source(sequential,A).
show(not_kept(N),developer,visible,2) :- reason(not_kept(N)).

% The optional {limits} part of the retry and not-kept lines.
limited(R) :- reason(R), counter(R,_).
reason_arg(R,limits,1,clause_layout) :- limited(R), not sequential, counter(R,F), F >= 5.
reason_arg(R,limits,2,clause_flight) :- limited(R), not sequential, counter(R,F), F >= 10.

% rule(N): how the spot is chosen and when a step is kept (detail; not in sequential mode).
reason(rule(N)) :- kept(N), not sequential.
reason_kind(rule(N),rule) :- reason(rule(N)).
reason_class(rule(N),rule) :- reason(rule(N)).
reason_source(rule(N),A) :- reason(rule(N)), option_source(sequential,A).
show(rule(N),developer,detail,1) :- reason(rule(N)).

% capacity(N): capacity of a composite sector (detail; composite function max only).
reason(capacity(N)) :- kept(N), capacity_is_max, hotspot(N,S,T), hotspot_navpoints(N,S,K).
reason_kind(capacity(N),capacity) :- reason(capacity(N)), hotspot(N,S,T), hotspot_navpoints(N,S,K), K != 1.
reason_kind(capacity(N),capacity_one) :- reason(capacity(N)), hotspot(N,S,T), hotspot_navpoints(N,S,1).
reason_class(capacity(N),capacity) :- reason(capacity(N)).
reason_arg(capacity(N),sector,0,S) :- reason(capacity(N)), hotspot(N,S,T).
reason_arg(capacity(N),navpoints,0,K) :- reason(capacity(N)), hotspot(N,S,T), hotspot_navpoints(N,S,K).
reason_arg(capacity(N),time,0,T) :- reason(capacity(N)), hotspot(N,S,T).
reason_source(capacity(N),hotspot_navpoints(N,S,K)) :- reason(capacity(N)), hotspot(N,S,T), hotspot_navpoints(N,S,K).
reason_source(capacity(N),A) :- reason(capacity(N)), option_source(composite,A).
show(capacity(N),developer,detail,2) :- reason(capacity(N)).

% subproblem(N): which flights of the cell went to the solver (detail).
occupant_rank(N,R,F,D) :- occupant(N,F,D,R).
cell_demand(N,D) :- step(N), hotspot(N,S,T), hotspot_load(N,S,T,D,_).
with_occupants(N) :- occupants_recorded(N), taken_count(N,_).
reason(subproblem(N)) :- kept(N), instance_read(N), cell_demand(N,_).
% all flights of the cell went to the solver
subproblem_all(N) :- reason(subproblem(N)), with_occupants(N), taken_count(N,K), cell_demand(N,K).
reason_kind(subproblem(N),subproblem_only) :- subproblem_all(N), cell_demand(N,1).
reason_kind(subproblem(N),subproblem_both) :- subproblem_all(N), cell_demand(N,2).
reason_kind(subproblem(N),subproblem_all) :- subproblem_all(N), cell_demand(N,D), D > 2.
% some were left out: a tie at the cut, or the shortest durations
cut(N,K) :- reason(subproblem(N)), with_occupants(N), taken_count(N,K), cell_demand(N,D), K < D.
tie_at_cut(N) :- cut(N,K), occupant_rank(N,K,_,D), occupant_rank(N,K+1,_,D).
tied(N,F,R) :- tie_at_cut(N), cut(N,K), occupant_rank(N,K,_,D), occupant_rank(N,R,F,D), R > K.
longer(N,F,R) :- tie_at_cut(N), cut(N,K), occupant_rank(N,K,_,D), occupant_rank(N,R,F,D2), R > K, D2 > D.
reason_kind(subproblem(N),subproblem_tie) :- tie_at_cut(N), #count{ F : tied(N,F,_) } = 1.
reason_kind(subproblem(N),subproblem_tie_many) :- tie_at_cut(N), #count{ F : tied(N,F,_) } > 1.
reason_kind(subproblem(N),subproblem_shortest) :- cut(N,K), not tie_at_cut(N), K != 1.
reason_kind(subproblem(N),subproblem_shortest_one) :- cut(N,1), not tie_at_cut(N).
% without recorded occupants: the rule only (a cell with one flight needs no rule)
reason_kind(subproblem(N),subproblem_rule) :- reason(subproblem(N)), not with_occupants(N), cell_demand(N,D), D != 1.
reason_kind(subproblem(N),subproblem_only) :- reason(subproblem(N)), not with_occupants(N), cell_demand(N,1).
reason_class(subproblem(N),rule) :- reason(subproblem(N)).
reason_arg(subproblem(N),demand,0,D) :- reason(subproblem(N)), cell_demand(N,D).
reason_arg(subproblem(N),sector,0,S) :- reason(subproblem(N)), hotspot(N,S,T).
reason_arg(subproblem(N),time,0,T) :- reason(subproblem(N)), hotspot(N,S,T).
reason_arg(subproblem(N),taken,I,F) :- reason(subproblem(N)), subproblem_flight(N,F),
                                       I = #count{ F2 : subproblem_flight(N,F2), F2 <= F }.
reason_arg(subproblem(N),k,0,K) :- cut(N,K).
reason_arg(subproblem(N),limit,0,X) :- reason(subproblem(N)), not with_occupants(N), cap(N,X).
reason_arg(subproblem(N),taken_durations,R,(F,D)) :- cut(N,K), occupant_rank(N,R,F,D), R <= K.
reason_arg(subproblem(N),left_out,R,F) :- cut(N,K), occupant_rank(N,R,F,_), R > K.
reason_arg(subproblem(N),left_out_durations,R,(F,D)) :- cut(N,K), occupant_rank(N,R,F,D), R > K.
reason_arg(subproblem(N),tied,R,F) :- tied(N,F,R).
% the optional {longer} part of the tie lines: left-out flights with a longer duration than the tie
reason_arg(subproblem(N),longer,R,F) :- longer(N,F,R).
reason_source(subproblem(N),subproblem_flight(N,F)) :- reason(subproblem(N)), subproblem_flight(N,F).
reason_source(subproblem(N),hotspot_load(N,S,T,D,C)) :- reason(subproblem(N)), hotspot(N,S,T), hotspot_load(N,S,T,D,C).
reason_source(subproblem(N),occupant(N,F,D,R)) :- reason(subproblem(N)), with_occupants(N), occupant(N,F,D,R).
reason_source(subproblem(N),taken_count(N,K)) :- reason(subproblem(N)), taken_count(N,K).
reason_source(subproblem(N),param(N,max_aircraft,M)) :- reason(subproblem(N)), not with_occupants(N), param(N,max_aircraft,M).
reason_source(subproblem(N),hotspot_overload(N,S,T,O)) :- reason(subproblem(N)), not with_occupants(N), hotspot(N,S,T), hotspot_overload(N,S,T,O).
show(subproblem(N),developer,detail,3) :- reason(subproblem(N)).
% the lines that state the sort rule say stored flight duration unless the run records the rule
stored_wording(subproblem_shortest,subproblem_shortest_stored).
stored_wording(subproblem_shortest_one,subproblem_shortest_one_stored).
stored_wording(subproblem_tie,subproblem_tie_stored).
stored_wording(subproblem_tie_many,subproblem_tie_many_stored).
stored_wording(subproblem_rule,subproblem_rule_stored).
reason_template(R,T) :- reason_kind(R,K), stored_wording(K,T), not duration_exact.
reason_source(R,A) :- reason_kind(R,K), stored_wording(K,_), option_source(duration,A).

% legs(N): later flights of the same aircraft that went with the solver's flights (detail).
leg_group(N,P,M) :- later_leg(N,_,P), M = #min{ L : later_leg(N,L,P) }.
leg_key(N,L,P,M) :- later_leg(N,L,P), leg_group(N,P,M).
reason(legs(N)) :- kept(N), instance_read(N), later_leg(N,_,_).
reason_kind(legs(N),legs) :- reason(legs(N)).
reason_class(legs(N),rule) :- reason(legs(N)).
reason_arg(legs(N),legs,I,(L,P)) :- reason(legs(N)), leg_key(N,L,P,M),
                                    I = #count{ M2,L2 : leg_key(N,L2,_,M2), (M2,L2) <= (M,L) }.
reason_source(legs(N),later_leg(N,L,P)) :- reason(legs(N)), later_leg(N,L,P).
show(legs(N),developer,detail,4) :- reason(legs(N)).

% delays(N): the departure delays the solver could choose (detail). In sequential mode each solver flight
% gets its first candidate route only, with delay k*W; k is raised only outside sequential mode
% (evaluate_solution.py), so the line is derived for k = 0 and only when the lp file shows one path and one
% start time for every solver flight.
reason(delays(N)) :- kept(N), not sequential, instance_read(N), delay_range(N,_,_), subproblem_flight(N,_).
one_option(N,F) :- subproblem_flight(N,F), offered_paths(N,F,1), offered_starts(N,F,1).
more_options(N) :- subproblem_flight(N,F), not one_option(N,F).
reason(delays(N)) :- kept(N), sequential, instance_read(N), param(N,additional_time_increase,0),
                     subproblem_flight(N,_), not more_options(N).
reason_kind(delays(N),delays) :- reason(delays(N)), not sequential, #count{ F : subproblem_flight(N,F) } > 1.
reason_kind(delays(N),delays_one) :- reason(delays(N)), not sequential, #count{ F : subproblem_flight(N,F) } = 1.
reason_kind(delays(N),delays_sequential) :- reason(delays(N)), sequential, #count{ F : subproblem_flight(N,F) } > 1.
reason_kind(delays(N),delays_sequential_one) :- reason(delays(N)), sequential, #count{ F : subproblem_flight(N,F) } = 1.
reason_class(delays(N),rule) :- reason(delays(N)).
reason_arg(delays(N),taken,I,F) :- reason(delays(N)), subproblem_flight(N,F),
                                   I = #count{ F2 : subproblem_flight(N,F2), F2 <= F }.
reason_arg(delays(N),lo,0,Lo) :- reason(delays(N)), not sequential, delay_range(N,Lo,_).
reason_arg(delays(N),hi,0,Hi) :- reason(delays(N)), not sequential, delay_range(N,_,Hi).
reason_source(delays(N),param(N,P,V)) :- reason(delays(N)), param(N,P,V),
    P = (additional_time_increase;max_delay_per_iteration).
reason_source(delays(N),subproblem_flight(N,F)) :- reason(delays(N)), subproblem_flight(N,F).
reason_source(delays(N),offered_paths(N,F,P)) :- reason(delays(N)), sequential, subproblem_flight(N,F), offered_paths(N,F,P).
reason_source(delays(N),offered_starts(N,F,K)) :- reason(delays(N)), sequential, subproblem_flight(N,F), offered_starts(N,F,K).
reason_source(delays(N),A) :- reason(delays(N)), option_source(sequential,A).
show(delays(N),developer,detail,5) :- reason(delays(N)).

% layouts(N): the sector layouts the solver could choose (detail).
layout_count(N,L) :- kept(N), instance_read(N), L = #count{ C : layout_option(N,C) }.
reason(layouts(N)) :- layout_count(N,L), L > 0, hotspot(N,_,_).
reason_kind(layouts(N),layouts) :- reason(layouts(N)), layout_count(N,L), L > 2.
reason_kind(layouts(N),layouts_one_split) :- reason(layouts(N)), layout_count(N,2).
reason_kind(layouts(N),layouts_current_only) :- reason(layouts(N)), layout_count(N,1).
reason_class(layouts(N),rule) :- reason(layouts(N)).
reason_arg(layouts(N),n,0,L) :- reason(layouts(N)), layout_count(N,L).
reason_arg(layouts(N),m,0,L-1) :- reason(layouts(N)), layout_count(N,L).
reason_arg(layouts(N),sector,0,S) :- reason(layouts(N)), hotspot(N,S,T).
reason_source(layouts(N),layout_option(N,C)) :- reason(layouts(N)), layout_option(N,C).
show(layouts(N),developer,detail,6) :- reason(layouts(N)).

% changed(N): the flights with a new trajectory in this step (detail).
reason(changed(N)) :- kept(N), flights_recorded(N).
reason_kind(changed(N),changed) :- reason(changed(N)), changed_flight(N,_).
reason_kind(changed(N),changed_none) :- reason(changed(N)), not changed_flight(N,_).
reason_class(changed(N),outcome) :- reason(changed(N)).
reason_arg(changed(N),changed,I,F) :- reason(changed(N)), changed_flight(N,F),
                                      I = #count{ F2 : changed_flight(N,F2), F2 <= F }.
reason_source(changed(N),flights_recorded(N)) :- reason(changed(N)).
reason_source(changed(N),changed_flight(N,F)) :- reason(changed(N)), changed_flight(N,F).
show(changed(N),developer,detail,7) :- reason(changed(N)).

% unchanged(N): flights of the sub-problem that kept their trajectory (detail).
kept_trajectory(N,F) :- kept(N), instance_read(N), flights_recorded(N), in_subproblem(N,F), not changed_flight(N,F).
reason(unchanged(N)) :- kept_trajectory(N,_).
reason_kind(unchanged(N),unchanged) :- reason(unchanged(N)).
reason_class(unchanged(N),outcome) :- reason(unchanged(N)).
reason_arg(unchanged(N),unchanged,I,F) :- kept_trajectory(N,F), I = #count{ F2 : kept_trajectory(N,F2), F2 <= F }.
reason_source(unchanged(N),flights_recorded(N)) :- reason(unchanged(N)).
reason_source(unchanged(N),subproblem_flight(N,F)) :- kept_trajectory(N,F), subproblem_flight(N,F).
reason_source(unchanged(N),later_leg(N,F,P)) :- kept_trajectory(N,F), later_leg(N,F,P).
show(unchanged(N),developer,detail,8) :- reason(unchanged(N)).

% layout_result(N): what happened to the hotspot sector (detail).
reason(layout_result(N)) :- kept(N), merging_off, split(N,S,T,_).
reason(layout_result(N)) :- kept(N), merging_on, hotspot(N,_,_).
reason_kind(layout_result(N),layout_split) :- reason(layout_result(N)), merging_off, split(N,S,T,P), P > 1.
reason_kind(layout_result(N),layout_same) :- reason(layout_result(N)), merging_off, split(N,S,T,1).
reason_kind(layout_result(N),layout_merging_on) :- reason(layout_result(N)), merging_on.
reason_class(layout_result(N),outcome) :- reason(layout_result(N)).
reason_arg(layout_result(N),sector,0,S) :- reason(layout_result(N)), hotspot(N,S,T).
reason_arg(layout_result(N),time,0,T) :- reason(layout_result(N)), merging_off, hotspot(N,S,T).
reason_arg(layout_result(N),parts,0,P) :- reason(layout_result(N)), merging_off, split(N,S,T,P).
reason_arg(layout_result(N),part_ids,I,P) :- reason(layout_result(N)), merging_off, part(N,P),
                                             I = #count{ P2 : part(N,P2), P2 <= P }.
reason_source(layout_result(N),sector_record(N)) :- reason(layout_result(N)), sector_record(N).
reason_source(layout_result(N),part_after(N,P,V)) :- reason(layout_result(N)), merging_off, part_after(N,P,V).
reason_source(layout_result(N),A) :- reason(layout_result(N)), option_source(merging,A).
show(layout_result(N),developer,detail,9) :- reason(layout_result(N)).

% scope(N): what the explanation covers (detail).
reason(scope(N)) :- kept(N).
reason_kind(scope(N),scope) :- reason(scope(N)), not sequential.
reason_kind(scope(N),scope_sequential) :- reason(scope(N)), sequential.
reason_class(scope(N),rule) :- reason(scope(N)).
reason_source(scope(N),kept(N)) :- reason(scope(N)).
reason_source(scope(N),A) :- reason(scope(N)), option_source(sequential,A).
show(scope(N),developer,detail,10) :- reason(scope(N)).

about(hotspot(N),step(N)) :- reason(hotspot(N)).
about(retry(N),step(N)) :- reason(retry(N)).
about(not_kept(N),step(N)) :- reason(not_kept(N)).
about(rule(N),step(N)) :- reason(rule(N)).
about(capacity(N),step(N)) :- reason(capacity(N)).
about(subproblem(N),step(N)) :- reason(subproblem(N)).
about(legs(N),step(N)) :- reason(legs(N)).
about(delays(N),step(N)) :- reason(delays(N)).
about(layouts(N),step(N)) :- reason(layouts(N)).
about(changed(N),step(N)) :- reason(changed(N)).
about(unchanged(N),step(N)) :- reason(unchanged(N)).
about(layout_result(N),step(N)) :- reason(layout_result(N)).
about(scope(N),step(N)) :- reason(scope(N)).

% ------------------------------------------------------------------ the filed plan (step 0)
reason(start) :- start_overload(_).
reason_kind(start,start) :- start_overload(O), O > 0.
reason_kind(start,start_none) :- start_overload(0).
reason_class(start,capacity) :- reason(start).
reason_arg(start,overload,0,O) :- start_overload(O).
start_key(S,O) :- start_sector_overload(S,O), O > 0.
reason_arg(start,sectors,I,(S,O)) :- start_overload(X), X > 0, start_key(S,O),
                                     I = #count{ S2,O2 : start_key(S2,O2), (-O2,S2) <= (-O,S) }.
reason_source(start,start_overload(O)) :- start_overload(O).
reason_source(start,start_sector_overload(S,O)) :- start_overload(X), X > 0, start_key(S,O).
show(start,developer,visible,1) :- reason(start).
reason(start_rule) :- start_overload(_).
reason_kind(start_rule,start_rule) :- reason(start_rule).
reason_class(start_rule,rule) :- reason(start_rule).
reason_source(start_rule,start_overload(O)) :- start_overload(O).
show(start_rule,developer,visible,2) :- reason(start_rule).
about(start,step(0)) :- reason(start).
about(start_rule,step(0)) :- reason(start_rule).


% ------------------------------------------------------------------ row reasons (K12)
% One line under each row of the change list of a kept step: every changed flight and the split of the hotspot
% sector. The class of a solver flight's line comes from the keep contrast: the best answer of the step's
% sub-problem with the flight on its current trajectory (the foil), compared with the recorded answer (the factual)
% level by level in the priority order of encoding.lp; the first level that differs decides (lexicographic weak
% constraints). The recorded answer is optimal for its sub-problem (solver.py solves without a time limit), and the
% factual re-solve must reproduce its clingo cost vector (factual_mismatch otherwise). "Exactly as good" is said only
% when all five levels are equal, and only within this step.
level_rank(overload,1). level_rank(delay,2). level_rank(sectors,3). level_rank(changed,4). level_rank(config,5).
row_step(N) :- kept(N), keep_recorded(N).

% the contrasts X of a step N and their foil costs
contrast_of(flight(N,F),N) :- keep_cost(N,F,_,_).
contrast_of(both(N,F),N) :- keep_both_cost(N,F,_,_).
contrast_of(alt(N,F),N) :- alt_cost(N,F,_,_).
contrast_of(layout(N),N) :- keep_layout_cost(N,_,_).
contrast_of(alt_layout(N),N) :- alt_layout_cost(N,_,_).
foil_cost(flight(N,F),L,V) :- keep_cost(N,F,L,V).
foil_cost(both(N,F),L,V) :- keep_both_cost(N,F,L,V).
foil_cost(alt(N,F),L,V) :- alt_cost(N,F,L,V).
foil_cost(layout(N),L,V) :- keep_layout_cost(N,L,V).
foil_cost(alt_layout(N),L,V) :- alt_layout_cost(N,L,V).
foil_level(X,L) :- foil_cost(X,L,_).
factual_level(N,L) :- factual_cost(N,L,_).
complete(X) :- contrast_of(X,N), #count{ L : foil_level(X,L), level_rank(L,_) } = 5,
               #count{ L : factual_level(N,L), level_rank(L,_) } = 5.
differs(X,L) :- contrast_of(X,N), foil_cost(X,L,A), factual_cost(N,L,B), A != B.
some_difference(X) :- differs(X,_).
higher_difference(X,L) :- differs(X,L), level_rank(L,R), differs(X,L2), level_rank(L2,R2), R2 < R.
deciding(X,L) :- differs(X,L), not higher_difference(X,L).
tie(X) :- complete(X), not some_difference(X).
worse(X) :- deciding(X,L), contrast_of(X,N), foil_cost(X,L,A), factual_cost(N,L,B), A > B.
better(X) :- deciding(X,L), contrast_of(X,N), foil_cost(X,L,A), factual_cost(N,L,B), A < B.
% the price of the chosen answer: the first level below the deciding one on which the foil is lower (not the
% layout rank, a tie-breaker with no meaning to the user)
gain_level(X,L) :- deciding(X,D), level_rank(D,RD), contrast_of(X,N), foil_cost(X,L,A), factual_cost(N,L,B), A < B,
                   level_rank(L,R), R > RD, L != config.
earlier_gain(X,L) :- gain_level(X,L), level_rank(L,R), gain_level(X,L2), level_rank(L2,R2), R2 < R.
gain(X,L) :- gain_level(X,L), not earlier_gain(X,L).

% net arrival delay when the run scores signed delay (also when the fact is missing: encoding.lp defaults to signed)
signed_delay :- run_option(arrival_delay_metric,signed).
signed_delay :- option_assumed(arrival_delay_metric).
signed_delay :- not run_option(arrival_delay_metric,_).
delay_words(words_net) :- signed_delay.
delay_words(words_plain) :- not signed_delay.

% flight_reason(N,F): one per changed flight of a kept step with keep contrasts
row_flight(N,F) :- row_step(N), changed_flight(N,F).
solver_row(N,F) :- row_flight(N,F), subproblem_flight(N,F).
leg_row(N,F,P) :- row_flight(N,F), later_leg(N,F,P), not subproblem_flight(N,F).
row_reason(flight_reason(N,F)) :- row_flight(N,F).
% a solver flight whose current trajectory was not offered (its current trajectory is recorded and no path equals it)
row_kind(flight_reason(N,F),not_offered) :- solver_row(N,F), current_recorded(N,F), not keep_offered(N,F).
row_kind(flight_reason(N,F),keep_infeasible) :- solver_row(N,F), keep_offered(N,F), keep_infeasible(N,F).
keep_kind(overload,keep_overload). keep_kind(delay,keep_delay). keep_kind(sectors,keep_sectors).
keep_kind(changed,keep_changed). keep_kind(config,keep_config).
row_kind(flight_reason(N,F),K) :- solver_row(N,F), keep_offered(N,F), worse(flight(N,F)), deciding(flight(N,F),L),
                                  keep_kind(L,K).
% a tie that moves exactly one other solver flight G (unchanged in the recorded answer), changes nothing else, and
% keeping F and G together is worse or impossible: "flight F or flight G had to change"
tie_other(N,F,G) :- solver_row(N,F), tie(flight(N,F)), keep_moves(N,F,G), #count{ G2 : keep_moves(N,F,G2) } = 1,
                    #count{ G3 : keep_other(N,F,G3) } = 1, keep_other(N,F,G), not keep_layout_differs(N,F),
                    keep_both_flight(N,F,G), worse(both(N,F)).
tie_other(N,F,G) :- solver_row(N,F), tie(flight(N,F)), keep_moves(N,F,G), #count{ G2 : keep_moves(N,F,G2) } = 1,
                    #count{ G3 : keep_other(N,F,G3) } = 1, keep_other(N,F,G), not keep_layout_differs(N,F),
                    keep_both_flight(N,F,G), keep_both_infeasible(N,F).
has_tie_other(N,F) :- tie_other(N,F,_).
row_kind(flight_reason(N,F),keep_tie_other) :- has_tie_other(N,F).
row_kind(flight_reason(N,F),keep_tie) :- solver_row(N,F), keep_offered(N,F), tie(flight(N,F)), not has_tie_other(N,F).
% later legs follow the path index of the solver flight they fly after (chosen_path(L,P) :- chosen_path(F,P))
row_kind(flight_reason(N,F),leg) :- leg_row(N,F,P), changed_flight(N,P).
row_kind(flight_reason(N,F),leg_parent_kept) :- leg_row(N,F,P), not changed_flight(N,P).

% layout_reason(N): the split of the hotspot sector (the current layout is configuration 0)
row_reason(layout_reason(N)) :- row_step(N), split(N,S,T,P), P > 1, layout_choice(N,C), C != 0.
layout_kind(overload,layout_keep_overload). layout_kind(delay,layout_keep_delay).
layout_kind(changed,layout_keep_changed).
row_kind(layout_reason(N),K) :- row_reason(layout_reason(N)), worse(layout(N)), deciding(layout(N),L), layout_kind(L,K).
row_kind(layout_reason(N),layout_keep_tie) :- row_reason(layout_reason(N)), tie(layout(N)).

reason(R) :- row_reason(R).
reason_kind(R,K) :- row_kind(R,K).
row_class(keep_overload,capacity). row_class(layout_keep_overload,capacity).
row_class(keep_delay,cost). row_class(keep_sectors,cost). row_class(keep_changed,cost). row_class(keep_config,cost).
row_class(layout_keep_delay,cost). row_class(layout_keep_changed,cost).
row_class(keep_tie_other,tie). row_class(keep_tie,tie). row_class(layout_keep_tie,tie).
row_class(not_offered,rule). row_class(keep_infeasible,rule). row_class(leg,rule). row_class(leg_parent_kept,rule).
reason_class(R,C) :- row_kind(R,K), row_class(K,C).
show(R,developer,row,1) :- row_reason(R).
about(flight_reason(N,F),flight(N,F)) :- row_reason(flight_reason(N,F)).
about(layout_reason(N),sector(N,S,T,before)) :- row_reason(layout_reason(N)), hotspot(N,S,T).

% arguments: the contrast of a row and its numbers at the deciding level and at the gain level
row_contrast(flight_reason(N,F),flight(N,F)) :- solver_row(N,F).
row_contrast(layout_reason(N),layout(N)) :- row_reason(layout_reason(N)).
reason_arg(flight_reason(N,F),flight,0,F) :- row_reason(flight_reason(N,F)).
reason_arg(flight_reason(N,F),parent,0,P) :- leg_row(N,F,P).
reason_arg(flight_reason(N,F),other,0,G) :- tie_other(N,F,G).
reason_arg(layout_reason(N),sector,0,S) :- row_reason(layout_reason(N)), hotspot(N,S,T).
reason_arg(R,time,0,T) :- row_contrast(R,X), contrast_of(X,N), hotspot(N,S,T).
reason_arg(R,foil,0,A) :- row_contrast(R,X), worse(X), deciding(X,L), L != config, foil_cost(X,L,A).
reason_arg(R,chosen,0,B) :- row_contrast(R,X), worse(X), deciding(X,L), L != config, contrast_of(X,N),
                            factual_cost(N,L,B).
% only the lines that name the delay (the kind, the gain or the keep-both part)
delay_text(keep_delay). delay_text(layout_keep_delay).
uses_delay_words(R) :- row_kind(R,K), delay_text(K).
uses_delay_words(R) :- row_contrast(R,X), worse(X), gain(X,delay).
uses_delay_words(flight_reason(N,F)) :- tie_other(N,F,_), deciding(both(N,F),delay).
reason_arg(R,delay_words,0,W) :- uses_delay_words(R), delay_words(W).
gain_clause(delay,gain_delay). gain_clause(sectors,gain_sectors). gain_clause(changed,gain_changed).
reason_arg(R,gain,0,C) :- row_contrast(R,X), worse(X), gain(X,L), gain_clause(L,C).
reason_arg(R,gain_foil,0,A) :- row_contrast(R,X), worse(X), gain(X,L), foil_cost(X,L,A).
reason_arg(R,gain_chosen,0,B) :- row_contrast(R,X), worse(X), gain(X,L), contrast_of(X,N), factual_cost(N,L,B).
alt_equal(N,F) :- tie(alt(N,F)).
alt_layout_equal(N) :- tie(alt_layout(N)).
reason_arg(flight_reason(N,F),alt,0,alt_flight) :- solver_row(N,F), worse(flight(N,F)), alt_equal(N,F).
reason_arg(layout_reason(N),alt,0,alt_layout) :- row_reason(layout_reason(N)), worse(layout(N)), alt_layout_equal(N).
both_clause(overload,both_overload). both_clause(delay,both_delay). both_clause(sectors,both_sectors).
both_clause(changed,both_changed). both_clause(config,both_config).
reason_arg(flight_reason(N,F),both,0,C) :- tie_other(N,F,_), deciding(both(N,F),L), both_clause(L,C).
reason_arg(flight_reason(N,F),both,0,both_infeasible) :- tie_other(N,F,_), keep_both_infeasible(N,F).
reason_arg(flight_reason(N,F),both_foil,0,A) :- tie_other(N,F,_), deciding(both(N,F),L), foil_cost(both(N,F),L,A).
reason_arg(flight_reason(N,F),both_chosen,0,B) :- tie_other(N,F,_), deciding(both(N,F),L), factual_cost(N,L,B).

% sources: the numbers each line states and the facts that select its kind
reason_source(flight_reason(N,F),changed_flight(N,F)) :- row_reason(flight_reason(N,F)).
reason_source(flight_reason(N,F),later_leg(N,F,P)) :- leg_row(N,F,P).
reason_source(flight_reason(N,F),changed_flight(N,P)) :- leg_row(N,F,P), changed_flight(N,P).
reason_source(flight_reason(N,F),current_recorded(N,F)) :- solver_row(N,F), current_recorded(N,F).
reason_source(flight_reason(N,F),chosen_current(N,F)) :- solver_row(N,F), chosen_current(N,F).
reason_source(flight_reason(N,F),keep_offered(N,F)) :- solver_row(N,F), keep_offered(N,F).
reason_source(flight_reason(N,F),keep_infeasible(N,F)) :- solver_row(N,F), keep_infeasible(N,F).
reason_source(R,keep_cost(N,F,L,V)) :- row_contrast(R,flight(N,F)), deciding(flight(N,F),L), keep_cost(N,F,L,V).
reason_source(R,keep_cost(N,F,L,V)) :- row_contrast(R,flight(N,F)), gain(flight(N,F),L), keep_cost(N,F,L,V).
reason_source(R,keep_cost(N,F,L,V)) :- row_contrast(R,flight(N,F)), tie(flight(N,F)), keep_cost(N,F,L,V).
reason_source(R,keep_layout_cost(N,L,V)) :- row_contrast(R,layout(N)), deciding(layout(N),L), keep_layout_cost(N,L,V).
reason_source(R,keep_layout_cost(N,L,V)) :- row_contrast(R,layout(N)), gain(layout(N),L), keep_layout_cost(N,L,V).
reason_source(R,keep_layout_cost(N,L,V)) :- row_contrast(R,layout(N)), tie(layout(N)), keep_layout_cost(N,L,V).
reason_source(R,factual_cost(N,L,V)) :- row_contrast(R,X), contrast_of(X,N), deciding(X,L), factual_cost(N,L,V).
reason_source(R,factual_cost(N,L,V)) :- row_contrast(R,X), contrast_of(X,N), gain(X,L), factual_cost(N,L,V).
reason_source(R,factual_cost(N,L,V)) :- row_contrast(R,X), contrast_of(X,N), tie(X), factual_cost(N,L,V).
reason_source(flight_reason(N,F),keep_moves(N,F,G)) :- tie_other(N,F,G).
reason_source(flight_reason(N,F),keep_both_cost(N,F,L,V)) :- tie_other(N,F,_), deciding(both(N,F),L),
                                                             keep_both_cost(N,F,L,V).
reason_source(flight_reason(N,F),keep_both_infeasible(N,F)) :- tie_other(N,F,_), keep_both_infeasible(N,F).
reason_source(flight_reason(N,F),alt_cost(N,F,L,V)) :- reason_arg(flight_reason(N,F),alt,0,_), alt_cost(N,F,L,V).
reason_source(layout_reason(N),alt_layout_cost(N,L,V)) :- reason_arg(layout_reason(N),alt,0,_), alt_layout_cost(N,L,V).
reason_source(layout_reason(N),split(N,S,T,P)) :- row_reason(layout_reason(N)), split(N,S,T,P).
reason_source(layout_reason(N),layout_choice(N,C)) :- row_reason(layout_reason(N)), layout_choice(N,C).
reason_source(R,run_option(arrival_delay_metric,M)) :- reason_arg(R,delay_words,0,_), run_option(arrival_delay_metric,M).

% ------------------------------------------------------------------ row checks
% A failed check drops every row line of the step (the change list says so); the header is unchanged.
% the factual re-solve reproduces the recorded clingo cost vector (same ground weak constraints: bans are externals)
row_error(factual_mismatch(N)) :- row_step(N), recorded_clingo(N,I,V), not factual_clingo(N,I,V).
row_error(factual_mismatch(N)) :- row_step(N), factual_clingo(N,I,V), not recorded_clingo(N,I,V).
row_error(factual_mismatch(N)) :- row_step(N), not factual_level(N,overload).
% level 10 of the factual is the total overload after the step
row_error(keep_scope(N)) :- row_step(N), factual_cost(N,overload,V), objective(N,after,overload,A), V != A.
row_error(not_optimal(X)) :- better(X), contrast_of(X,N), row_step(N).
row_error(foil_unproven(flight(N,F))) :- row_step(N), keep_unproven(N,F).
row_error(foil_unproven(both(N,F))) :- row_step(N), keep_both_unproven(N,F).
row_error(foil_unproven(alt(N,F))) :- row_step(N), alt_unproven(N,F).
row_error(foil_unproven(layout(N))) :- row_step(N), keep_layout_unproven(N).
row_error(foil_unproven(alt_layout(N))) :- row_step(N), alt_layout_unproven(N).
row_error(foil_cost_mismatch(flight(N,F))) :- row_step(N), keep_cost_mismatch(N,F).
row_error(foil_cost_mismatch(both(N,F))) :- row_step(N), keep_both_cost_mismatch(N,F).
row_error(foil_cost_mismatch(alt(N,F))) :- row_step(N), alt_cost_mismatch(N,F).
row_error(foil_cost_mismatch(layout(N))) :- row_step(N), keep_layout_cost_mismatch(N).
row_error(foil_cost_mismatch(alt_layout(N))) :- row_step(N), alt_layout_cost_mismatch(N).
% the numbers a line needs are missing
row_error(incomplete(flight(N,F))) :- solver_row(N,F), not current_recorded(N,F).
row_error(incomplete(flight(N,F))) :- solver_row(N,F), keep_offered(N,F), not keep_infeasible(N,F),
                                      not complete(flight(N,F)).
alt_known(N,F) :- alt_cost(N,F,_,_).
alt_known(N,F) :- alt_infeasible(N,F).
row_error(incomplete(alt(N,F))) :- solver_row(N,F), worse(flight(N,F)), not alt_known(N,F).
row_error(incomplete(alt(N,F))) :- solver_row(N,F), alt_cost(N,F,_,_), not complete(alt(N,F)).
both_known(N,F) :- keep_both_cost(N,F,_,_).
both_known(N,F) :- keep_both_infeasible(N,F).
row_error(incomplete(both(N,F))) :- solver_row(N,F), tie(flight(N,F)), keep_moves(N,F,_),
                                    #count{ G : keep_moves(N,F,G) } = 1, not both_known(N,F).
row_error(incomplete(both(N,F))) :- solver_row(N,F), keep_both_cost(N,F,_,_), not complete(both(N,F)).
row_error(incomplete(layout(N))) :- row_reason(layout_reason(N)), not keep_layout_infeasible(N), not complete(layout(N)).
row_error(incomplete(alt_layout(N))) :- row_reason(layout_reason(N)), worse(layout(N)), alt_layout_cost(N,_,_),
                                        not complete(alt_layout(N)).
% a changed solver flight whose chosen trajectory is its current one, or the reverse
row_error(changed_current(N,F)) :- row_step(N), changed_flight(N,F), subproblem_flight(N,F), chosen_current(N,F).
row_error(changed_current(N,F)) :- row_step(N), flights_recorded(N), chosen_checked(N,F), subproblem_flight(N,F),
                                   not changed_flight(N,F), not chosen_current(N,F).
% every changed flight has exactly one kind
row_error(row_missing(N,F)) :- row_flight(N,F), #count{ K : row_kind(flight_reason(N,F),K) } != 1.
row_error(row_missing(N,F)) :- solver_row(N,F), not flight_entry(N,F).
% the recorded split and the chosen layout agree (configuration 0 is the current layout)
split_recorded(N) :- split(N,_,_,P), P > 1.
row_error(layout_mismatch(N)) :- row_step(N), merging_off, split_recorded(N), layout_choice(N,0).
row_error(layout_mismatch(N)) :- row_step(N), merging_off, layout_choice(N,C), C != 0, not split_recorded(N).
row_error(layout_mismatch(N)) :- row_step(N), split_recorded(N), not layout_choice(N,_).
% a layout contrast without a line: keeping the current layout impossible, or decided by the open sectors or the
% layout rank (no template; never seen on the study traces)
row_error(layout_unexpected(N)) :- row_reason(layout_reason(N)), keep_layout_infeasible(N).
row_error(layout_unexpected(N)) :- row_reason(layout_reason(N)), worse(layout(N)), deciding(layout(N),L),
                                   not layout_kind(L,_).
row_error(layout_unexpected(N)) :- row_reason(layout_reason(N)), #count{ K : row_kind(layout_reason(N),K) } != 1,
                                   complete(layout(N)).
% only sub-problem 0 is explained
row_error(multi_subproblem(N)) :- row_step(N), subproblem_count(N,K), K > 1.

% ------------------------------------------------------------------ output contract
% error/1 covers the header reasons (and the record); row_error/1 the row reasons
has_source(R) :- reason_source(R,_).
error(no_source(R)) :- reason(R), not row_reason(R), not has_source(R).
has_template(K) :- template(K,developer,_).
error(no_template(K)) :- reason_kind(R,K), not row_reason(R), not has_template(K).
error(no_template(K)) :- reason_arg(_,limits,_,K), not has_template(K).
error(no_template(T)) :- reason_template(_,T), not has_template(T).
row_error(no_source(R)) :- row_reason(R), not has_source(R).
row_error(no_template(K)) :- row_kind(_,K), not has_template(K).
row_clause_key(gain;alt;both;delay_words).
row_error(no_template(K)) :- row_reason(R), row_clause_key(Key), reason_arg(R,Key,_,K), not has_template(K).

#show reason/1.
#show about/2.
#show reason_kind/2.
#show reason_class/2.
#show reason_arg/4.
#show reason_source/2.
#show reason_template/2.
#show show/4.
#show delta/5.
#show error/1.
#show row_error/1.
