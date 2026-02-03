% Encoding in the ASPaeroFlow Optimizer:
% GET ALL FLIGHT IDS:

flightID(ID) :- flightPlan(ID,_,_).

all_flightID(ID) :- flightID(ID).
all_flightID(ID) :- sector_flight(ID,_,_).
all_flightID(ID) :- navpoint_flight(ID,_,_).

% GUESSES:
1{chosen_config(CONFIG):config(CONFIG,_)}1.
1{chosen_path(ID,P):paths(ID,P)}1 :- flightID(ID).

reroute(ID) :- flightID(ID), not chosen_path(ID,0).
reconfigure_sectors(CONFIG) :- chosen_config(CONFIG), CONFIG > 0.

% GET CHOSEN SECTORS:
navpoint_sector_assignment(NAV,SEC,TIME) :- chosen_config(CONFIG), possible_assignment(NAV,SEC,TIME,CONFIG).
sector_capacity(SEC,CAPACITY,TIME) :- chosen_config(CONFIG), possible_sector_capacity(SEC,CAPACITY,TIME,CONFIG).


time_minimum(T) :- T = #min{T':possible_assignment(_,_,T',_);T':next_pos(_,_,_,T',_,_);T':single_pos(_,_,_,T')}.
time_maximum(T) :- T = #max{T':possible_assignment(_,_,T',_);T':next_pos(_,_,_,_,_,T');T':single_pos(_,_,_,T')}.
time(TMIN..TMAX) :- time_minimum(TMIN), time_maximum(TMAX).

% FOR STARTING AIRPORT:
sector_flight(ID,SEC0,T) :- actual_flight_operations_start_time(ID,TSTART,P), chosen_path(ID,P), single_pos(ID,P,NAV0,T0), navpoint_sector_assignment(NAV0,SEC0,T), T = T0 + TSTART.
sector_flight(ID,SEC0,T) :- actual_flight_operations_start_time(ID,TSTART,P), chosen_path(ID,P), time(T), next_pos(ID,P,NAV0,T0,NAV0,T1), T >= T0 + TSTART, T <= T1 + TSTART, navpoint_sector_assignment(NAV0,SEC0,T).
% EN-ROUTE:
sector_flight(ID,SEC0,T) :- actual_flight_operations_start_time(ID,TSTART,P), chosen_path(ID,P), time(T), next_pos(ID,P,NAV0,T0,NAV1,T1), NAV0 != NAV1, T >= T0 + TSTART, T <= T1 + TSTART, DT = (T1 - T0)/2, T <= DT + T0 + TSTART, navpoint_sector_assignment(NAV0,SEC0,T).
sector_flight(ID,SEC1,T) :- actual_flight_operations_start_time(ID,TSTART,P), chosen_path(ID,P), time(T), next_pos(ID,P,NAV0,T0,NAV1,T1), NAV0 != NAV1, T >= T0 + TSTART, T <= T1 + TSTART, DT = (T1 - T0)/2, T > DT + T0 + TSTART, navpoint_sector_assignment(NAV1,SEC1,T).

navpoint_flight(ID,NAV0,T) :- actual_flight_operations_start_time(ID,TSTART,P), chosen_path(ID,P), single_pos(ID,P,NAV0,T0), T = T0 + TSTART.
navpoint_flight(ID,NAV0,T) :- actual_flight_operations_start_time(ID,TSTART,P), chosen_path(ID,P), next_pos(ID,P,NAV0,T0,_,_), T = T0 + TSTART.
navpoint_flight(ID,NAV1,T) :- actual_flight_operations_start_time(ID,TSTART,P), chosen_path(ID,P), next_pos(ID,P,_,_,NAV1,T1), T = T1 + TSTART.

% CAPACITY CONSTRAINT:
overload(SEC, T, LOAD-CAPACITY) :- sector_capacity(SEC,CAPACITY,T), #count{ID:sector_flight(ID,SEC,T)} = LOAD, LOAD > CAPACITY.
% WITH DIFFS:
%overload(SEC1,T,LOAD-CAPACITY) :- sector_capacity(SEC1,CAPACITY,T), #count{ID:sector_flight(ID,SEC1,T),sector_flight(ID,SEC2,T-1), SEC1!=SEC2} = LOAD, LOAD > CAPACITY.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% FLIGHT MUST OCCUR:
flight_occurs(ID) :- sector_flight(ID,_,_).
:- flightID(ID), not flight_occurs(ID).


arrival_delay(ID,Y-T) :- chosen_path(ID,P), planned_arrival_time(ID,T), actual_arrival_time(ID,Y,P).

% ------------- OPTIMIZATION ----------------
% ------------------------------------------------------------------------------
% PRIMARY WEAK CONSTRAINT:
:~ overload(X,T,OVER). [OVER@10,X,T]
:~ chosen_config(CONFIG), config(CONFIG,TOTAL_OVER). [TOTAL_OVER@10,CONFIG]
% SECONDARY WEAK CONSTRAINT:
:~ arrival_delay(ID,DIFF). [DIFF@9,ID]
% TERTIARY WEAK CONSTRAINT:
:~ chosen_config(CONFIG), config_number_sectors(CONFIG,NUMBER). [NUMBER@8,CONFIG]
% QUINARY WEAK CONSTRAINT:
:~ reroute(ID). [1@7,ID]
% QUARTERNARY WEAK CONSTRAINT:
:~ chosen_config(CONFIG). [CONFIG@6,CONFIG]


%#show output_sec_cap/3.
%#show show_overload/3.
%#show all_flightID/1.
%#show navpoint_flight/3.
%#show sector_flight/3.
%#show reroute/1.
%#show actual_arrival_time/2.
%#show planned_arrival_time/2.
%#show arrival_delay/2.
%#show overload/3.
%#show arrival_delay/2.
%#show time/1.
%#show path/2.
#show chosen_config/1.
#show chosen_path/2.
%#show sector_flight/3.
%#show tmp_output/3.
%#show tmp_sector_flight/6.
%#show time/1.
