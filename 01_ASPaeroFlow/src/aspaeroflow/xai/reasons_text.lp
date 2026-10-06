% reasons_text.lp: the texts of reasons.lp (loaded into the same Control; never with encoding.lp).
%
% template(Kind, Audience, Text): {key} is filled by xai/reasons.py from reason_arg(R,key,I,V) (lists joined
% as "a, b and c"). error_text(Code, Text): plain text of an error/1 code, shown inside record_error.
% Rules for every text: ASCII only, none of the characters double quote, backslash and apostrophe (the texts
% travel inside data("DIFF","...") atoms and explanation payloads whose escaping breaks them).
% Words: "step N" is the optimizer step, "time 11" the time index, "N time periods" a duration.

% ------------------------------------------------------------------ visible
template(hotspot_rule, developer, "Sector {sector} at time {time} had {demand} flights for a capacity of {capacity}: the earliest overload, lowest sector number first.").
template(hotspot_rule_one, developer, "Sector {sector} at time {time} had 1 flight for a capacity of {capacity}: the earliest overload, lowest sector number first.").
template(hotspot, developer, "Sector {sector} at time {time} had {demand} flights for a capacity of {capacity}.").
template(hotspot_one, developer, "Sector {sector} at time {time} had 1 flight for a capacity of {capacity}.").
template(retry, developer, "Steps {steps} at this spot were not kept, so this step offered only departure delays of {lo} to {hi} time periods{limits}.").
template(retry_one, developer, "Step {steps} at this spot was not kept, so this step offered only departure delays of {lo} to {hi} time periods{limits}.").
template(not_kept, developer, "Step {iteration} was not kept: the total overload did not fall (stays at {overload}). The plan is unchanged; a next attempt at this spot would offer departure delays of {lo} to {hi} time periods{limits}.").
template(not_kept_sequential, developer, "Step {iteration} was not kept: the total overload did not fall (stays at {overload}). The plan is unchanged.").
template(clause_layout, developer, " and only the current layout").
template(clause_flight, developer, " and at most 1 flight").

% ------------------------------------------------------------------ details
template(rule, developer, "ASPaeroFlow works on the earliest time with an overload, and at that time on the overloaded sector with the lowest number. A step is kept only if the total overload falls.").
template(capacity, developer, "Sector {sector} has {navpoints} navpoints. The capacity of a sector is the largest capacity among its navpoints.").
template(capacity_one, developer, "Sector {sector} has 1 navpoint. The capacity of a sector is the largest capacity among its navpoints.").
template(subproblem_all, developer, "All {demand} flights in sector {sector} at time {time} went to the solver: {taken}.").
template(subproblem_both, developer, "Both flights in sector {sector} at time {time} went to the solver: {taken}.").
template(subproblem_only, developer, "The only flight in sector {sector} at time {time} went to the solver: {taken}.").
template(subproblem_shortest, developer, "Of the {demand} flights in sector {sector} at time {time}, the {k} with the shortest stored flight duration went to the solver: {taken_durations}; {left_out_durations}.").
template(subproblem_shortest_one, developer, "Of the {demand} flights in sector {sector} at time {time}, the one with the shortest stored flight duration went to the solver: {taken_durations}; {left_out_durations}.").
template(subproblem_tie, developer, "Of the {demand} flights in sector {sector} at time {time}, {k} went to the solver: {taken}; {tied} has the same stored duration and was left out by its higher flight number.").
template(subproblem_tie_many, developer, "Of the {demand} flights in sector {sector} at time {time}, {k} went to the solver: {taken}; {tied} have the same stored duration and were left out by their higher flight numbers.").
template(subproblem_rule, developer, "Of the {demand} flights in sector {sector} at time {time}, {taken} went to the solver (rule: at most {limit}, shortest stored flight duration first, lower flight number on ties).").
template(legs, developer, "The later flights of their aircraft went with them: {legs}.").
template(legs_one, developer, "The later flight of its aircraft went with it: {legs}.").
template(delays, developer, "{taken} could each depart {lo} to {hi} time periods later than in the current plan.").
template(delays_one, developer, "{taken} could depart {lo} to {hi} time periods later than in the current plan.").
template(layouts, developer, "The solver could choose among {n} layouts of sector {sector}: the current one and {m} splits.").
template(layouts_one_split, developer, "The solver could choose among 2 layouts of sector {sector}: the current one and 1 split.").
template(layouts_current_only, developer, "Only the current layout of sector {sector} was offered.").
template(changed, developer, "New trajectory in this step: {changed}.").
template(changed_none, developer, "No flight got a new trajectory in this step.").
template(unchanged, developer, "The solver left {unchanged} unchanged.").
template(layout_split, developer, "Sector {sector} was split into {parts} sectors ({part_ids}) from time {time} on.").
template(layout_same, developer, "The layout of sector {sector} did not change.").
template(layout_merging_on, developer, "Sector merging was on in this run; layout changes outside sector {sector} are not described.").
template(scope, developer, "This describes this step only. The spot, the flights and the delay range follow fixed rules; the new trajectories and the layout are the answer of the solver to the sub-problem of this step.").

% ------------------------------------------------------------------ the filed plan (step 0)
template(start, developer, "The filed plan has a total overload of {overload} (flights above capacity, all times together), in sectors {sectors}.").
template(start_none, developer, "The filed plan has no overload.").
template(start_rule, developer, "Each step works on one overloaded sector at one time and is kept only if the total overload falls.").

% ------------------------------------------------------------------ errors
template(record_error, developer, "The recorded numbers of this step do not match the documented rules: {text}. No explanation is shown.").
template(render_error, developer, "The explanation text of this step could not be produced.").

error_text(kept_rule, "the step was kept although the total overload did not fall").
error_text(not_kept_rule, "the step was not kept although the total overload changed").
error_text(overload_mismatch, "the overload is not flights minus capacity").
error_text(before_mismatch, "the overload before the step differs from the value of the previous step").
error_text(missing_previous, "a value is missing").
error_text(missing_objective, "a value is missing").
error_text(not_unique, "a value is recorded twice").
error_text(taken_rule, "the number of flights passed to the solver does not follow the rule").
error_text(candidate_rule, "the flights passed to the solver are not the ones the rule selects").
error_text(changed_outside, "a flight outside the sub-problem changed").
error_text(parts_mismatch, "the parts after the split do not cover the sector").
error_text(split_extent, "the sector count rose by more than a split from the hotspot time on allows").
error_text(retry_spot, "a rejected attempt before this step worked on another spot").
error_text(retry_count, "the number of rejected attempts before this step does not match the counter").
error_text(no_source, "a sentence has no recorded source").
error_text(no_template, "a sentence has no text template").
error_text(render_error, "the explanation text of this step could not be produced").
