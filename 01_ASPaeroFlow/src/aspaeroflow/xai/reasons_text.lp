% reasons_text.lp: the texts of reasons.lp (loaded into the same Control; never with encoding.lp).
%
% template(Kind, Audience, Text): {key} is filled by xai/reasons.py from reason_arg(R,key,I,V) (lists joined
% as "a, b and c"). Optional parts: {limits} (the clause_* texts, or empty) and {longer} ("; 11 has a longer
% one", "; 11 and 20 have longer ones", or empty).
% error_text(Code, Text): plain text of an error/1 code, shown inside record_error.
% Rules for every text: ASCII only, none of the characters double quote, backslash and apostrophe (the texts
% travel inside data("DIFF","...") atoms and explanation payloads whose escaping breaks them).
% Words: "step N" is the optimizer step, "time 11" the time index, "N time periods" a duration.

% ------------------------------------------------------------------ visible
template(hotspot_rule, developer, "Sector {sector} at time {time} had {demand} flights for a capacity of {capacity}: the earliest overload, lowest sector number first.").
template(hotspot_rule_one, developer, "Sector {sector} at time {time} had 1 flight for a capacity of {capacity}: the earliest overload, lowest sector number first.").
template(hotspot, developer, "Sector {sector} at time {time} had {demand} flights for a capacity of {capacity}.").
template(hotspot_one, developer, "Sector {sector} at time {time} had 1 flight for a capacity of {capacity}.").
template(retry, developer, "Steps {steps} at this spot were not kept, so this step offered departure delays of {lo} to {hi} time periods (no shorter ones){limits}.").
template(retry_one, developer, "Step {steps} at this spot was not kept, so this step offered departure delays of {lo} to {hi} time periods (no shorter ones){limits}.").
template(not_kept, developer, "Step {iteration} (sector {sector} at time {time}) was not kept: the total overload did not fall (stays at {overload}). The plan is unchanged; a next attempt at this spot would offer departure delays of {lo} to {hi} time periods{limits}.").
template(not_kept_sequential, developer, "Step {iteration} (sector {sector} at time {time}) was not kept: the total overload did not fall (stays at {overload}). The plan is unchanged.").
template(clause_layout, developer, " and only the current layout").
template(clause_flight, developer, " and at most 1 flight").

% ------------------------------------------------------------------ details
template(rule, developer, "ASPaeroFlow works on the earliest time with an overload, and at that time on the overloaded sector with the lowest number. A step is kept only if the total overload falls.").
template(capacity, developer, "Sector {sector} has {navpoints} navpoints at time {time}. The capacity of a sector is the largest capacity among its navpoints.").
template(capacity_one, developer, "Sector {sector} has 1 navpoint at time {time}. The capacity of a sector is the largest capacity among its navpoints.").
template(subproblem_all, developer, "All {demand} flights in sector {sector} at time {time} went to the solver: {taken}.").
template(subproblem_both, developer, "Both flights in sector {sector} at time {time} went to the solver: {taken}.").
template(subproblem_only, developer, "The only flight in sector {sector} at time {time} went to the solver: {taken}.").
template(subproblem_shortest, developer, "Of the {demand} flights in sector {sector} at time {time}, the {k} with the shortest flight duration went to the solver: {taken_durations}; {left_out_durations}.").
template(subproblem_shortest_one, developer, "Of the {demand} flights in sector {sector} at time {time}, the one with the shortest flight duration went to the solver: {taken_durations}; {left_out_durations}.").
template(subproblem_tie, developer, "Of the {demand} flights in sector {sector} at time {time}, {k} went to the solver: {taken}; {tied} has the same flight duration and was left out by its higher flight number{longer}.").
template(subproblem_tie_many, developer, "Of the {demand} flights in sector {sector} at time {time}, {k} went to the solver: {taken}; {tied} have the same flight duration and were left out by their higher flight numbers{longer}.").
template(subproblem_rule, developer, "Of the {demand} flights in sector {sector} at time {time}, {taken} went to the solver (rule: at most {limit}, shortest flight duration first, lower flight number on ties).").
% the same five lines for traces that do not record flight_duration_rule (made before the rule was fixed)
template(subproblem_shortest_stored, developer, "Of the {demand} flights in sector {sector} at time {time}, the {k} with the shortest stored flight duration went to the solver: {taken_durations}; {left_out_durations}.").
template(subproblem_shortest_one_stored, developer, "Of the {demand} flights in sector {sector} at time {time}, the one with the shortest stored flight duration went to the solver: {taken_durations}; {left_out_durations}.").
template(subproblem_tie_stored, developer, "Of the {demand} flights in sector {sector} at time {time}, {k} went to the solver: {taken}; {tied} has the same stored duration and was left out by its higher flight number{longer}.").
template(subproblem_tie_many_stored, developer, "Of the {demand} flights in sector {sector} at time {time}, {k} went to the solver: {taken}; {tied} have the same stored duration and were left out by their higher flight numbers{longer}.").
template(subproblem_rule_stored, developer, "Of the {demand} flights in sector {sector} at time {time}, {taken} went to the solver (rule: at most {limit}, shortest stored flight duration first, lower flight number on ties).").
template(legs, developer, "Later flights of the same aircraft also went to the solver: {legs}.").
template(delays, developer, "{taken} could each depart {lo} to {hi} time periods later than in the current plan.").
template(delays_one, developer, "{taken} could depart {lo} to {hi} time periods later than in the current plan.").
template(delays_sequential, developer, "{taken} could each take one candidate route only, departing as in the current plan (sequential mode).").
template(delays_sequential_one, developer, "{taken} could take one candidate route only, departing as in the current plan (sequential mode).").
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
template(scope_sequential, developer, "This describes this step only. The spot and the flights follow fixed rules; the new trajectories and the layout are the answer of the solver to the sub-problem of this step.").

% ------------------------------------------------------------------ the filed plan (step 0)
template(start, developer, "The filed plan has a total overload of {overload} (flights above capacity, all times together), in sectors {sectors}.").
template(start_none, developer, "The filed plan has no overload.").
template(start_rule, developer, "Each step works on one overloaded sector at a single time and is kept only if the total overload falls.").

% ------------------------------------------------------------------ row reasons (one line under a change-list row)
% Clauses: {delay_words} (words_net, words_plain), {gain} (gain_*: the first criterion below the deciding one on
% which the answer that keeps the flight or the layout is better), {alt} (alt_*), {both} (both_*); each clause is
% filled with the row's arguments first (gain_foil, gain_chosen, both_foil, both_chosen, time, sector, delay_words).
template(keep_overload, developer, "With flight {flight} kept as it was, the best answer of this step would leave a total overload of {foil} instead of {chosen}{gain}.{alt}").
template(keep_delay, developer, "With flight {flight} kept as it was, the best answer of this step would have the same total overload and a {delay_words} of {foil} instead of {chosen} time periods, summed over the flights of this step{gain}.{alt}").
template(keep_sectors, developer, "With flight {flight} kept as it was, the best answer of this step would have the same total overload and arrival delay and {foil} instead of {chosen} open sectors at time {time}{gain}.{alt}").
template(keep_changed, developer, "With flight {flight} kept as it was, the best answer of this step would be the same on overload, arrival delay and open sectors and would have {foil} instead of {chosen} flights sent to the solver off the filed route or the earliest offered departure{gain}.{alt}").
template(keep_config, developer, "With flight {flight} kept as it was, the best answer of this step would be the same on overload, arrival delay, open sectors and flights sent to the solver, and would use a sector layout that the optimizer ranks below the chosen one when all else is equal.{alt}").
template(keep_tie_other, developer, "Flight {flight} or flight {other} had to change: keeping both as they were {both}. Keeping flight {flight} as it was and changing flight {other} is exactly as good; the solver picked one of the equally good answers.").
template(keep_tie, developer, "Keeping flight {flight} as it was is exactly as good on every criterion of this step; the solver picked one of the equally good answers.").
template(not_offered, developer, "The current trajectory of flight {flight} was not among the options of this step, so every answer changed it.").
template(keep_infeasible, developer, "No answer of this step keeps flight {flight} on its current trajectory.").
template(leg, developer, "Flies after flight {parent} on the same aircraft and moved with it.").
template(leg_parent_kept, developer, "Flies after flight {parent} on the same aircraft; its trajectory follows the option chosen for flight {parent}.").
template(layout_keep_overload, developer, "If sector {sector} had not been split, the best answer of this step would leave a total overload of {foil} instead of {chosen}{gain}.{alt}").
template(layout_keep_delay, developer, "If sector {sector} had not been split, the best answer of this step would have the same total overload and a {delay_words} of {foil} instead of {chosen} time periods, summed over the flights of this step{gain}.{alt}").
template(layout_keep_changed, developer, "If sector {sector} had not been split, the best answer of this step would be the same on overload, arrival delay and open sectors and would have {foil} instead of {chosen} flights sent to the solver off the filed route or the earliest offered departure{gain}.{alt}").
template(layout_keep_tie, developer, "If sector {sector} had not been split, the best answer of this step would be exactly as good on every criterion; the solver picked one of the equally good answers.").
template(words_net, developer, "net arrival delay").
template(words_plain, developer, "arrival delay").
template(gain_delay, developer, ", but its {delay_words} would be {gain_foil} instead of {gain_chosen} time periods").
template(gain_sectors, developer, ", but it would have {gain_foil} instead of {gain_chosen} open sectors at time {time}").
template(gain_changed, developer, ", but it would have {gain_foil} instead of {gain_chosen} flights sent to the solver off the filed route or the earliest offered departure").
template(alt_flight, developer, " Another new trajectory was exactly as good.").
template(alt_layout, developer, " Another split of sector {sector} was exactly as good.").
template(both_overload, developer, "would leave a total overload of {both_foil} instead of {both_chosen}").
template(both_delay, developer, "would have the same total overload and a {delay_words} of {both_foil} instead of {both_chosen} time periods").
template(both_sectors, developer, "would have the same total overload and arrival delay and {both_foil} instead of {both_chosen} open sectors at time {time}").
template(both_changed, developer, "would be the same on overload, arrival delay and open sectors and would have {both_foil} instead of {both_chosen} flights sent to the solver off the filed route or the earliest offered departure").
template(both_config, developer, "would use a sector layout that the optimizer ranks below the chosen one, all else being equal").
template(both_infeasible, developer, "is not possible in this step").
template(row_unchecked, developer, "The reasons of this step could not be checked.").

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
error_text(duration_rule, "a recorded flight duration differs from the time periods of its trajectory").
error_text(delay_options, "the departure options in the sub-problem do not match the delay rule").
error_text(parts_mismatch, "the parts after the split do not cover the sector").
error_text(split_extent, "the sector count rose by more than a split from the hotspot time on allows").
error_text(retry_spot, "a rejected attempt before this step worked on another spot").
error_text(retry_count, "the number of rejected attempts before this step does not match the counter").
error_text(no_source, "a sentence has no recorded source").
error_text(no_template, "a sentence has no text template").
error_text(render_error, "the explanation text of this step could not be produced").
