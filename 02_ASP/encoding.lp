% FACTS:
% navpoint_flight_plan(<ID>,<NAVPOINT-LOCATION>,<TIME>)
% navpoint_sector(<NAVPOINT-ID>,<SECTOR-ID>,<TIME=0 for FACTS>)
% navpoint_edge(<X0>,<X1>,<PLANE-SPEED>,<UNIT-DIST>)
% airport(<X>)
% aircraft(<ID>,<PLANE-SPEED>)
% aircraft_flight(<PLANE-ID>,<FLIGHT-ID>)
% atomic_sector(<NAVPOINT-ID>,<CAPACITY>,<TIME>)

navpoint(NAV) :- navpoint_sector(NAV,_,_).



% Dynamic Sector Allocation:
navpoint_sector(SEC,SEC,T) :- navpoint(SEC), time(T), airport(SEC).
1{navpoint_sector(NAV,SEC,T):navpoint_sector(_,SEC,0)}1:- navpoint(NAV), time(T), not airport(NAV), regulation_dynamic_sector_allocation.
initial_sector(SEC) :- navpoint_sector(_,SEC,0), regulation_restricted_dynamic_sector_allocation.
1{navpoint_sector_decision(SEC,0..3,T)}1 :- initial_sector(SEC), time(T), not airport(SEC), regulation_restricted_dynamic_sector_allocation.
navpoint_sector(NAV1,SEC1,T) :- navpoint_sector_decision(SEC,DEC,T), navpoint_sector_restricted_sector_allocation(SEC,DEC,NAV1,SEC1), regulation_restricted_dynamic_sector_allocation.
navpoint_sector(NAV,SEC,T) :- time(T), navpoint_sector(NAV,SEC,0), -regulation_dynamic_sector_allocation.



:- navpoint_sector(NAV,SEC1,T), navpoint_sector(NAV,SEC2,T), SEC1 != SEC2.
:- navpoint_sector(NAV,SEC,T), not airport(NAV), airport(SEC).
:- navpoint_sector(NAV,SEC,T), airport(NAV), not airport(SEC).
:- navpoint_sector(NAV1,SEC,T), NAV1 !=SEC, not navpoint_sector(SEC,SEC,T).

% MINIMIZE SECTOR-NUMBER:
sector_len(SEC,T,CAP_LEN) :- time(T), navpoint(SEC), CAP_LEN = #count{NAV:navpoint_sector(NAV,SEC,T)}.
sector(SEC,T,0) :- sector_len(SEC,T,CAP_LEN), CAP_LEN = 0. 
sector(SEC,T,C) :- time(T), navpoint(SEC), C = #max{C',NAV:navpoint_sector(NAV,SEC,T),atomic_sector(NAV,C',T)}, sector_len(SEC,T,CAP_LEN), CAP_LEN > 0.
sector_number(T,NUM) :- time(T), NUM = #count{SEC:sector(SEC,T,C), C > 0}.

% MINIMIZE NAVPOINT-SECTOR ASSIGNMENT CHANGES:
navpoint_changed(NAV,T,1) :- navpoint_sector(NAV,SEC0,T), navpoint_sector(NAV,SEC1,T+1), SEC0 != SEC1.
navpoint_changed(NAV,T,0) :- navpoint_sector(NAV,SEC0,T), navpoint_sector(NAV,SEC0,T+1).
sector_diff(T,DIFF) :- time(T), DIFF = #sum{V,NAV: navpoint_changed(NAV,T,V)}.

% MINIMIZE RECONFIG:
reconfig(NAV,T) :- navpoint_sector(NAV,SEC,0), navpoint_sector(NAV,SEC2,T), SEC != SEC2, T > 0.

% GET ALL AIRPLANE SPEEDS
aircraft_speed(S) :- aircraft(_,S).

% UNDIRECTED GRAPH:
navpoint_edge(Y,X,S,D) :- navpoint_edge(X,Y,S,D).
navpoint_edge(X,X,S,1) :- airport(X), aircraft_speed(S).

% GET NAVPOINTS
navpoint(X) :- navpoint_edge(X,_,_,_).
navpoint(Y) :- navpoint_edge(_,Y,_,_).

% PRELIMINARIES:
% GET ALL TIME STEPS:
time(T) :- navpoint_flight_plan(_,_,T).
% GET ALL FLIHT IDS:
flightID(ID) :- navpoint_flight_plan(ID,_,_).

% MORE TIME HEURISTIC:
time(0).
time(T+1) :- time(T), max_time(TMAX), T < TMAX.
time(T+1) :- time(T), induced_max_time(TMAX), T < TMAX.
induced_max_time(T) :- T = #max{T': navpoint_flight_plan(_,_,T')}.

% NORMAL FLIGHT:
navpoint_flight(ID,X,T) :- navpoint_flight_plan(ID,X,T), not reroute(ID).

% LIMIT SECTOR CAPACITY (hard constraint modelling):
%overload(X,T,LOAD-C) :- sector(X,T,C), #count{ID:flight(ID,X,T),flight(ID,X2,T-1), X!=X2} = LOAD, LOAD > C.
% CURRENT METHOD -> LIMIT NUMBER FLIGHTS PER TIMESTEP IN SECTOR:
overload(X,T,LOAD-C) :- sector(X,T,C), #count{ID:flight(ID,X,T)} = LOAD, LOAD > C.

% REROUTE - YES/NO:
{reroute(ID)} :- navpoint_flight_plan(ID,_,_).
reroute(ID) :- planned_departure_time(ID,T), actual_departure_time(ID,T'), T' > T.
{navpoint_flight(ID,X,T):navpoint(X)}1 :- reroute(ID), time(T), actual_departure_time(ID,T'), T >= T', regulation_rerouting.
navpoint_flight(ID,X,T + TT) :- navpoint_flight_plan(ID,X,T), actual_departure_time(ID,TACT),planned_departure_time(ID,TPLAN), TT = TACT - TPLAN, -regulation_rerouting.
1{navpoint_flight_chosen_path(ID,P):regulation_restricted_rerouting_paths(ID,_,_,P)}1 :- reroute(ID), regulation_restricted_rerouting.
navpoint_flight(ID,X,T+T_DELTA) :- regulation_restricted_rerouting_paths(ID,X,T_DELTA,P), reroute(ID), actual_departure_time(ID,T), navpoint_flight_chosen_path(ID,P), regulation_restricted_rerouting.


% ----------------- ONLY FOR REROUTED PLANES --------------------
% NO FLIGHT BEFORE PLANNED START TIME:
planned_departure_time(ID,T) :- flightID(ID), T = #min{T':navpoint_flight_plan(ID,_,T')}.
:- reroute(ID), navpoint_flight(ID,_,T), planned_departure_time(ID,T'), T < T'.


% FLIGHT STARTS AT ORIGIN:
planned_origin(ID,X) :- reroute(ID),planned_departure_time(ID,T),navpoint_flight_plan(ID,X,T).
navpoint_flight(ID,X,T) :- reroute(ID), actual_departure_time(ID,T), planned_origin(ID,X).

% FLIGHT ENDS AT DESTINATION:
planned_destination(ID,X) :- reroute(ID), T=#max{T':navpoint_flight_plan(ID,_,T')}, navpoint_flight_plan(ID,X,T).
actual_arrival_time(ID,T) :- flightID(ID), T = #max{T':navpoint_flight(ID,_,T')}.
:- reroute(ID), actual_arrival_time(ID,T), navpoint_flight(ID,X,T), planned_destination(ID,X'), X != X'.
:- flightID(ID), not actual_arrival_time(ID,_).
:- flightID(ID), planned_destination(ID,X), actual_arrival_time(ID,T), navpoint_flight(ID,X',T), X != X'.

% NO FLIGHT AFTER DESTINATION REACHED:
actual_destination_first_reached_time(ID,T) :- reroute(ID), planned_destination(ID,X), T = #min{T':navpoint_flight(ID,X,T')}.
:- reroute(ID), navpoint_flight(ID,_,T), actual_destination_first_reached_time(ID,T'), T > T'.

% MUST BE PATH:
%navpoint_seq(ID,T,X,X'') :- navpoint_flight(ID,X,T), T'' = #min{T',X': navpoint_flight(ID,X',T'), T' > T}, navpoint_flight(ID,X'',T'').
%:- navpoint_flight(ID,X,T), navpoint_flight(ID,X',T'), T' > T, navpoint_seq(ID,T,X,X'), aircraft_flight(AID,ID), aircraft(AID,S), D = T'-T, not navpoint_edge(X,X',S,D).

% MUST BE PATH:
pot_navpoint_seq(ID,T,TT) :- navpoint_flight(ID,_,T), navpoint_flight(ID,_,TT), T < TT.
not_navpoint_seq(ID,T,TT) :- pot_navpoint_seq(ID,T,TT), T < TTT, TTT < TT, pot_navpoint_seq(ID,T,TTT).
navpoint_seq(ID,T,TT) :- pot_navpoint_seq(ID,T,TT), not not_navpoint_seq(ID,T,TT).
:- navpoint_seq(ID,T,TT), navpoint_flight(ID,X,T), navpoint_flight(ID,XX,TT), aircraft_flight(AID,ID), aircraft(AID,S), D = TT - T, not navpoint_edge(X,XX,S,D).


% NEVER TWICE AT SAME LOCATION:
:- navpoint_flight(ID,X,T), navpoint_flight(ID,X,T'), T != T', not airport(X).

% FLIGHT MUST OCCUR:
flightOccurs(ID) :- navpoint_flight(ID,_,_).
:- flightID(ID), not flightOccurs(ID).

% HELPER PREDS:
1{actual_departure_time(ID,T):time(T), T >= TP}1 :- planned_departure_time(ID,TP), regulation_delaying.
1{actual_departure_time(ID,T):time(T), T >= TP, T <= TP + MD}1 :- planned_departure_time(ID,TP), restricted_delaying_max_delay(MD), regulation_restricted_delaying.
actual_departure_time(ID,TP) :- planned_departure_time(ID,TP), -regulation_delaying.

planned_arrival_time(ID,T) :- flightID(ID), T = #max{T':navpoint_flight_plan(ID,_,T')}.
arrival_delay(ID,Y-T) :- planned_arrival_time(ID,T), actual_arrival_time(ID,Y).
actualFlightTime(ID,T) :- actual_departure_time(ID,TDA),actual_destination_first_reached_time(ID,TRA), T = TRA - TDA.
plannedFlightTime(ID,T) :- planned_departure_time(ID,TDP), planned_arrival_time(ID,TRP), T = TRP - TDP.

% NO GAPS:
:- time(T), actual_departure_time(ID,T'), T > T', actual_arrival_time(ID,T''), T < T'', not flight(ID,_,T).

% TRANSLATE navpoint_flight to flight
flight(ID,S,T) :- navpoint_flight(ID,X,T), navpoint_sector(X,S,T).
flightT(ID,T,T',0) :- navpoint_seq(ID,T,TT), D = TT - T, time(T'), T' > T, T' <= T + D/2.
flightT(ID,TT,T',1) :- navpoint_seq(ID,T,TT), D = TT - T, time(T'), T' < TT, T' > T + D/2.
flight(ID,S,T') :- flightT(ID,T,T',0), navpoint_flight(ID,X,T), navpoint_sector(X,S,T').
flight(ID,S,T') :- flightT(ID,TT,T',1), navpoint_flight(ID,XX,TT), navpoint_sector(XX,S,T').

%flight(ID,S,T') :- navpoint_flight(ID,X,T), navpoint_flight(ID,_,TT), navpoint_seq(ID,T,TT), D = TT - T, time(T'), T' > T, T' <= T + D/2, navpoint_sector(X,S,T').
%flight(ID,S',T') :- navpoint_flight(ID,_,T), navpoint_flight(ID,X',TT), navpoint_seq(ID,T,TT), D = TT - T,time(T'), T' < TT, T' > T + D/2, navpoint_sector(X',S',T').

% Multi-leg flights:
:- aircraft_flight(AID,FID0), aircraft_flight(AID,FID1), FID0 != FID1, planned_arrival_time(FID0,T0), planned_arrival_time(FID1,T1), T0 < T1, actual_arrival_time(FID0, T0'), actual_departure_time(FID1,T1'), T0' >= T1'.

% ------------------------------------------------------------------------------
% PRIMARY WEAK CONSTRAINT:
:~ overload(X,T,OVER). [OVER@10,X,T]
% SECONDARY WEAK CONSTRAINT:
:~ arrival_delay(ID,DIFF). [DIFF@9,ID]
% TERTIARY WEAK CONSTRAINT:
:~ sector_number(T,NUM). [NUM@8,T]
% OPTIMIZE FOR LEAST SECTOR CHANGES:
:~ sector_diff(T,DIFF). [DIFF@7,T]
% OPTIMIZE FOR LEAST REROUTES/DELAYS:
:~ reroute(ID). [1@6,ID]
% OPTIMIZE FOR LEAST RECONFIGS:
:~ reconfig(ID,T). [1@5,ID,T]

%#heuristic navpoint_sector(NAV,SEC,T) : time(T), navpoint_sector(NAV,SEC,0). [1,sign]
%#heuristic navpoint_sector(NAV,SEC,T) : time(T), navpoint_sector(NAV,SEC,0). [2,factor]
%#heuristic reroute(ID):navpoint_flight_plan(ID,_,_). [1,sign]
%#heuristic actual_departure_time(ID,TP) : planned_departure_time(ID,TP). [1,sign]



%output_sec_len(X,Y,Z) :- sector_len(X,Y,Z), Y = 12.
%output_sec_avg(X,Y,Z) :- sector_avg_cap(X,Y,Z), Y = 12.
%output_sec(X,Y,Z) :- sector(X,Y,Z), Y = 12.

#show flight/3.
#show navpoint_flight/3.
%#show navpoint_sector_decision/3.
%#show reroute/1.
%#show navpoint_seq/3.
%#show navpoint_seq/4.
%#show actual_arrival_time/2.
%#show planned_arrival_time/2.
%#show arrival_delay/2.
%#show overload/3.
%#show time/1.
%#show actual_destination_first_reached_time/2.
%#show planned_departure_time/2.
%#show actual_departure_time/2.
%#show navpoint_sector/3.
%#show output_sec_len/3.
%#show output_sec/3.
%#show output_sec_avg/3.
%#show navpoint_sector/3.

#show time/1.

