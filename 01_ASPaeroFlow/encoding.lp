% Encoding in the ASPaeroFlow Optimizer:


% GET ALL FLIGHT IDS:
flightID(ID) :- flightPlan(ID,_,_).

% GUESSES:
1{chosen_config(CONFIG):config(CONFIG)}1.
1{chosen_path(ID,P):paths(ID,P)}1 :- flightID(ID).

reroute(ID) :- flightID(ID), not chosen_path(ID,0).
reconfigure_sectors(CONFIG) :- chosen_config(CONFIG), CONFIG > 0.

% GET CHOSEN SECTORS:
navpoint_sector_assignment(NAV,SEC,TIME) :- chosen_config(CONFIG), possible_assignment(NAV,SEC,TIME,CONFIG).
sector_capacity(SEC,CAPACITY,TIME) :- chosen_config(CONFIG), possible_sector_capacity(SEC,CAPACITY,TIME,CONFIG).


time_minimum(T) :- T = #min{T':possible_assignment(_,_,T',_);T':next_pos(_,_,_,T',_,_)}.
time_maximum(T) :- T = #max{T':possible_assignment(_,_,T',_);T':next_pos(_,_,_,_,_,T')}.
time(TMIN..TMAX) :- time_minimum(TMIN), time_maximum(TMAX).

% FOR STARTING AIRPORT:
sector_flight(ID,SEC0,T) :- plannedDepartureTime(ID,TSTART), chosen_path(ID,P), time(T), next_pos(ID,P,NAV0,T0,NAV0,T1), T >= T0 + TSTART, T <= T1 + TSTART, navpoint_sector_assignment(NAV0,SEC0,T).
% EN-ROUTE:
sector_flight(ID,SEC0,T) :- plannedDepartureTime(ID,TSTART), chosen_path(ID,P), time(T), next_pos(ID,P,NAV0,T0,NAV1,T1), NAV0 != NAV1, T >= T0 + TSTART, T <= T1 + TSTART, DT = (T1 - T0)/2, T <= DT + T0 + TSTART, navpoint_sector_assignment(NAV0,SEC0,T).
sector_flight(ID,SEC1,T) :- plannedDepartureTime(ID,TSTART), chosen_path(ID,P), time(T), next_pos(ID,P,NAV0,T0,NAV1,T1), NAV0 != NAV1, T >= T0 + TSTART, T <= T1 + TSTART, DT = (T1 - T0)/2, T > DT + T0 + TSTART, navpoint_sector_assignment(NAV1,SEC1,T).

% CAPACITY CONSTRAINT:
overload(SEC, T, LOAD-CAPACITY) :- sector_capacity(SEC,CAPACITY,T), #count{ID:sector_flight(ID,SEC,T)} = LOAD, LOAD > CAPACITY.
% WITH DIFFS:
%overload(SEC1,T,LOAD-CAPACITY) :- sector_capacity(SEC1,CAPACITY,T), #count{ID:sector_flight(ID,SEC1,T),sector_flight(ID,SEC2,T-1), SEC1!=SEC2} = LOAD, LOAD > CAPACITY.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% REROUTE - YES/NO:
%{reroute(ID)} :- flightPlan(ID,_,_).
%reroute(ID) :- flightPlan(ID,_,_).
%1{path(ID,P):paths(ID,P)}1 :- reroute(ID).
%flight(ID,T + T',X) :- reroute(ID), path(ID,P), actualDepartureTime(ID, T), sectorFlight(ID,T',X,P).

% FLIGHT MUST OCCUR:
flightOccurs(ID) :- sector_flight(ID,_,_).
%:- flightID(ID), not flightOccurs(ID).


% ------------- OPTIMIZATION ----------------
arrivalDelay(ID,Y-T) :- chosen_path(ID,P), plannedArrivalTime(ID,T), actualArrivalTime(ID,Y,P).

%tmp_output(8,50,T) :- sector_flight(8,50,T).
%tmp_output(8,2,T) :- sector_flight(8,2,T).

% ------------------------------------------------------------------------------
% PRIMARY WEAK CONSTRAINT:
:~ overload(X,T,OVER). [OVER@10,X,T]
% SECONDARY WEAK CONSTRAINT:
:~ arrivalDelay(ID,DIFF). [DIFF@9,ID]
% TERTIARY WEAK CONSTRAINT:
:~ reconfigure_sectors(CONFIG). [1@8,CONFIG]
% QUARTIARY WEAK CONSTRAINT:
:~ reroute(ID). [1@7,ID]


%#show flight/3.
%#show reroute/1.
%#show actualArrivalTime/2.
%#show plannedArrivalTime/2.
%#show arrivalDelay/2.
%#show overload/3.
%#show arrivalDelay/2.
%#show time/1.
%#show path/2.
%#show chosen_config/1.
%#show chosen_path/2.
#show sector_flight/3.
%#show tmp_output/3.
%#show tmp_sector_flight/6.
%#show time/1.
