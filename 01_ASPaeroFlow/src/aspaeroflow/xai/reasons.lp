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
%   occupants_recorded(N)  occupant(N,F,StoredDuration,Rank)  taken_count(N,K)
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
%                                         composite_sector_function (max/other)
%   filed_layout_static  evaluation_window(W)  run_length(L)  run_kept(K)
%   start_overload(O)  start_sector_overload(S,O)
% Evidence facts (flights_recorded, sector_record, instance_read, occupants_recorded) guard every
% sentence that says "no ..." or "unchanged": a missing fact never produces such a sentence.
%
% Output (#show): reason/1, about/2, reason_kind/2, reason_class/2, reason_arg/4 (Key, I orders list items,
% scalars I = 0), reason_source/2, show/4 (R, audience, visible|detail, order), delta/5, error/1.
% Reason terms: step N: hotspot(N), retry(N), not_kept(N) (visible); rule(N), capacity(N), subproblem(N),
% legs(N), delays(N), layouts(N), changed(N), unchanged(N), layout_result(N), scope(N) (detail);
% filed plan: start, start_rule. Later audiences (operator) add about(R, flight(N,F)) and
% about(R, sector(N,S,T,Phase)) with Phase in initial/before/after.

#defined step/1.             #defined kept/1.             #defined rejected/1.
#defined hotspot/3.          #defined hotspot_load/5.     #defined hotspot_overload/4.
#defined hotspot_navpoints/3.
#defined occupants_recorded/1. #defined occupant/4.       #defined taken_count/2.
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

% The first K flights of the cell by stored duration (ties: lower flight number) are the solver's flights.
selected(N,F) :- occupant(N,F,_,I), taken_count(N,K), I <= K.
check_candidates(N) :- step(N), instance_read(N), occupants_recorded(N), taken_count(N,_).
error(candidate_rule(N)) :- check_candidates(N), selected(N,F), not subproblem_flight(N,F).
error(candidate_rule(N)) :- check_candidates(N), subproblem_flight(N,F), not selected(N,F).
in_subproblem(N,F) :- subproblem_flight(N,F).
in_subproblem(N,L) :- later_leg(N,L,_).
error(changed_outside(N,F)) :- instance_read(N), flights_recorded(N), changed_flight(N,F), not in_subproblem(N,F).

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
% some were left out: a tie at the cut, or the shortest stored durations
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
% the optional {longer} part of the tie lines: left-out flights with a longer stored duration than the tie
reason_arg(subproblem(N),longer,R,F) :- longer(N,F,R).
reason_source(subproblem(N),subproblem_flight(N,F)) :- reason(subproblem(N)), subproblem_flight(N,F).
reason_source(subproblem(N),hotspot_load(N,S,T,D,C)) :- reason(subproblem(N)), hotspot(N,S,T), hotspot_load(N,S,T,D,C).
reason_source(subproblem(N),occupant(N,F,D,R)) :- reason(subproblem(N)), with_occupants(N), occupant(N,F,D,R).
reason_source(subproblem(N),taken_count(N,K)) :- reason(subproblem(N)), taken_count(N,K).
reason_source(subproblem(N),param(N,max_aircraft,M)) :- reason(subproblem(N)), not with_occupants(N), param(N,max_aircraft,M).
reason_source(subproblem(N),hotspot_overload(N,S,T,O)) :- reason(subproblem(N)), not with_occupants(N), hotspot(N,S,T), hotspot_overload(N,S,T,O).
show(subproblem(N),developer,detail,3) :- reason(subproblem(N)).

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

% ------------------------------------------------------------------ output contract
has_source(R) :- reason_source(R,_).
error(no_source(R)) :- reason(R), not has_source(R).
has_template(K) :- template(K,developer,_).
error(no_template(K)) :- reason_kind(_,K), not has_template(K).
error(no_template(K)) :- reason_arg(_,limits,_,K), not has_template(K).

#show reason/1.
#show about/2.
#show reason_kind/2.
#show reason_class/2.
#show reason_arg/4.
#show reason_source/2.
#show show/4.
#show delta/5.
#show error/1.
