% NAIVE ENCODING - FIRST VERSION:
% Assumptions:
% Each time step = 1h
% Each airplane takes exactly 1h per sector, regardless of its type (uniform sectors and airplanes)
% Each airplane has a filed flight plan
% No environmental effects

% FACTS:
% navpointFlightPlan(<ID>,<TIME>,<NAVPOINT-LOCATION>)
% sector(<LOCATION>,<TIME>,<CAPACITY>)
% navpointEdge(<X0>,<X1>,<PLANE-SPEED>,<UNIT-DIST>)
% airport(<X>)
% airplane(<ID>,<PLANE-SPEED>)
% airplane_flight(<PLANE-ID>,<FLIGHT-ID>)
% navaid_sector(<NAVAID-ID>,<SECTOR-ID>,<TIME=0 for FACTS>)
% sectorCapacityFactor({sector_capactiy_factor}).

navaid(NAV) :- navaid_sector(NAV,_,_).

% Sector Guess:
% Airport Strict:
navaid_sector(SEC,SEC,T) :- navaid(SEC), time(T), airport(SEC).
% En-Route Guess:
1{navaid_sector(NAV,SEC,T):navaid_sector(_,SEC,0)}1:- navaid(NAV), time(T).
:- navaid_sector(NAV,SEC1,T), navaid_sector(NAV,SEC2,T), SEC1 != SEC2.
%:- navaid_sector(NAV1,SEC,T), navaid_sector(NAV2,SEC,T), NAV1 != NAV2.
:- navaid_sector(NAV,SEC,T), not airport(NAV), airport(SEC).

% MINIMIZE NAVAID-SECTOR ASSIGNMENT CHANGES:
navaid_changed(NAV,T,1) :- navaid_sector(NAV,SEC0,T), navaid_sector(NAV,SEC1,T+1), SEC0 != SEC1.
navaid_changed(NAV,T,0) :- navaid_sector(NAV,SEC0,T), navaid_sector(NAV,SEC0,T+1).
sector_diff(SEC,T,DIFF) :- sector_len(SEC,T,_), DIFF = #sum{V,NAV: navaid_changed(NAV,T,V)}.

% SECTOR CAP FUNCTION:
%sector_avg_cap(SEC,T,CAP_SUM / CAP_LEN) :- time(T), navaid(SEC), CAP_SUM = #sum{C,NAV:navaid_sector(NAV,SEC,T), atomic_sector(NAV,T,C)}, sector_len(SEC,T,CAP_LEN), CAP_LEN > 0.
%sector_avg_cap(SEC,T,0) :- time(T), navaid(SEC), sector_len(SEC,T,CAP_LEN), CAP_LEN = 0.
%sector_variables(SEC,T,CAP_LEN,Z,CAP_AVG) :- sectorCapacityFactor(Z), navaid_sector(_,SEC,T), sector_len(SEC,T,CAP_LEN), sector_avg_cap(SEC,T,CAP_AVG).
%sector(SEC,T,C) :- sector_variables(SEC,T,CAP_LEN,Z,CAP_AVG), CAP_LEN >= Z, C = (CAP_AVG * (Z+1)) / 2, CAP_LEN >= 1. 
%sector(SEC,T,C) :- sector_variables(SEC,T,CAP_LEN,Z,CAP_AVG), CAP_LEN < Z, C = (CAP_AVG * (2 * Z * CAP_LEN - CAP_LEN * CAP_LEN + CAP_LEN)) / (Z*2), CAP_LEN >= 1. 

sector_len(SEC,T,CAP_LEN) :- time(T), navaid(SEC), CAP_LEN = #count{NAV:navaid_sector(NAV,SEC,T)}.
sector(SEC,T,0) :- sector_len(SEC,T,CAP_LEN), CAP_LEN = 0. 
sector(SEC,T,C) :- time(T), navaid(SEC), C = #max{C',NAV:navaid_sector(NAV,SEC,T),atomic_sector(NAV,T,C')}, sector_len(SEC,T,CAP_LEN), CAP_LEN > 0.
sector_number(T,NUM) :- time(T), NUM = #count{SEC:sector(SEC,T,C), C > 0}.

% GET ALL AIRPLANE SPEEDS
airplane_speed(S) :- airplane(_,S).

% UNDIRECTED GRAPH:
navpointEdge(Y,X,S,D) :- navpointEdge(X,Y,S,D).
navpointEdge(X,X,S,1) :- airport(X), airplane_speed(S).

% GET NAVPOINTS
navpoint(X) :- navpointEdge(X,_,_,_).
navpoint(Y) :- navpointEdge(_,Y,_,_).

% PRELIMINARIES:
% GET ALL TIME STEPS:
time(T) :- navpointFlightPlan(_,T,_).
% GET ALL FLIHT IDS:
flightID(ID) :- navpointFlightPlan(ID,_,_).

% MORE TIME HEURISTIC:
time(0).
time(T+1) :- time(T), maxTime(TMAX), T < TMAX.
time(T+1) :- time(T), inducedMaxTime(TMAX), T < TMAX.
inducedMaxTime(T) :- T = #max{T': navpointFlightPlan(_,T',_)}.

% NORMAL FLIGHT:
navpointFlight(ID,T,X) :- navpointFlightPlan(ID,T,X), not reroute(ID).

% LIMIT SECTOR CAPACITY (hard constraint modelling):
%overload(X,T,LOAD-C) :- sector(X,T,C), #count{ID:flight(ID,T,X),flight(ID,T-1,X2), X!=X2} = LOAD, LOAD > C.
overload(X,T,LOAD-C) :- sector(X,T,C), #count{ID:flight(ID,T,X)} = LOAD, LOAD > C.

% REROUTE - YES/NO:
{reroute(ID)} :- navpointFlightPlan(ID,_,_).
reroute(ID) :- plannedDepartureTime(ID,T), actualDepartureTime(ID,T'), T' > T.
{navpointFlight(ID,T,X):navpoint(X)}1 :- reroute(ID), time(T), actualDepartureTime(ID,T'), T >= T'.

% ----------------- ONLY FOR REROUTED PLANES --------------------
% NO FLIGHT BEFORE PLANNED START TIME:
plannedDepartureTime(ID,T) :- flightID(ID), T = #min{T':navpointFlightPlan(ID,T',_)}.
:- reroute(ID), navpointFlight(ID,T,_), plannedDepartureTime(ID,T'), T < T'.


% FLIGHT STARTS AT ORIGIN:
plannedOrigin(ID,X) :- reroute(ID),plannedDepartureTime(ID,T),navpointFlightPlan(ID,T,X).
%navpointFlight(ID,T,X) :- reroute(ID), navpointFlightPlan(ID,T,X), plannedOrigin(ID,X).
navpointFlight(ID,T,X) :- reroute(ID), actualDepartureTime(ID,T), plannedOrigin(ID,X).
%:- navpointFlight(ID,T,_), actualDepartureTime(ID,T'), T < T'.
%navpointFlight(ID,T,X) :- reroute(ID), time(T), plannedDepartureTime(ID,T'), actualDepartureTime(ID,T''), T >= T', T <= T'', plannedOrigin(ID,X).

% FLIGHT ENDS AT DESTINATION:
plannedDestination(ID,X) :- reroute(ID), T=#max{T':navpointFlightPlan(ID,T',_)}, navpointFlightPlan(ID,T,X).
actualArrivalTime(ID,T) :- flightID(ID), T = #max{T':navpointFlight(ID,T',_)}.
:- reroute(ID), actualArrivalTime(ID,T), navpointFlight(ID,T,X), plannedDestination(ID,X'), X != X'.
:- flightID(ID), not actualArrivalTime(ID,_).
:- flightID(ID), plannedDestination(ID,X), actualArrivalTime(ID,T), navpointFlight(ID,T,X'), X != X'.

% NO FLIGHT AFTER DESTINATION REACHED:
actualDestinationFirstReachedTime(ID,T) :- reroute(ID), plannedDestination(ID,X), T = #min{T':navpointFlight(ID,T',X)}.
:- reroute(ID), navpointFlight(ID,T,_), actualDestinationFirstReachedTime(ID,T'), T > T'.

% MUST BE PATH:
navpointSeq(ID,T,X,X'') :- navpointFlight(ID,T,X), T'' = #min{T',X': navpointFlight(ID,T',X'), T' > T}, navpointFlight(ID,T'',X'').
:- navpointFlight(ID,T,X), navpointFlight(ID,T',X'), T' > T, navpointSeq(ID,T,X,X'), airplane_flight(AID,ID), airplane(AID,S), D = T'-T, not navpointEdge(X,X',S,D).


% NEVER TWICE AT SAME LOCATION:
:- navpointFlight(ID,T,X), navpointFlight(ID,T',X), T != T', not airport(X).

% FLIGHT MUST OCCUR:
flightOccurs(ID) :- navpointFlight(ID,_,_).
:- flightID(ID), not flightOccurs(ID).

% HELPER PREDS:
1{actualDepartureTime(ID,T):time(T), T >= TP}1 :- plannedDepartureTime(ID,TP).
plannedArrivalTime(ID,T) :- flightID(ID), T = #max{T':navpointFlightPlan(ID,T',_)}.
arrivalDelay(ID,Y-T) :- plannedArrivalTime(ID,T), actualArrivalTime(ID,Y).
actualFlightTime(ID,T) :- actualDepartureTime(ID,TDA),actualDestinationFirstReachedTime(ID,TRA), T = TRA - TDA.
plannedFlightTime(ID,T) :- plannedDepartureTime(ID,TDP), plannedArrivalTime(ID,TRP), T = TRP - TDP.

% NO GAPS:
:- time(T), actualDepartureTime(ID,T'), T > T', actualArrivalTime(ID,T''), T < T'', not flight(ID,T,_).

% TRANSLATE navpointFlight to flight
flight(ID,T,S) :- navpointFlight(ID,T,X), navaid_sector(X,S,T).
flight(ID,T',S) :- navpointFlight(ID,T,X), navpointFlight(ID,T+D,X'), navpointSeq(ID,T,X,X'), navaid_sector(X,S,T'), airplane_flight(AID,ID), airplane(AID,SPEED), navpointEdge(X,X',SPEED,D), time(T'), T' > T, T' <= T + D/2. 
flight(ID,T',S') :- navpointFlight(ID,T,X), navpointFlight(ID,T+D,X'), navpointSeq(ID,T,X,X'), navaid_sector(X',S',T'), airplane_flight(AID,ID),  airplane(ID,SPEED), navpointEdge(X,X',SPEED,D), time(T'), T' < T+D, T' > T + D/2. 

% Multi-leg flights:
:- airplane_flight(AID,FID0), airplane_flight(AID,FID1), FID0 != FID1, plannedArrivalTime(FID0,T0), plannedArrivalTime(FID1,T1), T0 < T1, actualArrivalTime(FID0, T0'), actualDepartureTime(FID1,T1'), T0' >= T1'.

% ------------------------------------------------------------------------------
% PRIMARY WEAK CONSTRAINT:
:~ overload(X,T,OVER). [OVER@10,X,T]
% SECONDARY WEAK CONSTRAINT:
:~ arrivalDelay(ID,DIFF). [DIFF@9,ID]
% TERTIARY WEAK CONSTRAINT:
:~ sector_number(T,NUM). [NUM@8,T]
% QUARTINARY OPTIMIZE FOR MINIMUM TRAVEL TIME:
:~ actualFlightTime(ID,TA), plannedFlightTime(ID,TP), DIFF = TA - TP. [DIFF@7,ID]
% OPTIMIZE FOR LEAST SECTOR CHANGES:
:~ sector_diff(SEC,T,DIFF). [DIFF@6,SEC,T]

%output_sec_len(X,Y,Z) :- sector_len(X,Y,Z), Y = 12.
%output_sec_avg(X,Y,Z) :- sector_avg_cap(X,Y,Z), Y = 12.
%output_sec(X,Y,Z) :- sector(X,Y,Z), Y = 12.

#show flight/3.
%#show navpointFlight/3.
#show reroute/1.
%#show navpointSeq/4.
%#show actualArrivalTime/2.
%#show plannedArrivalTime/2.
%#show arrivalDelay/2.
%#show overload/3.
%#show timeStepGranularities/1.
%#show time/1.
%#show actualDestinationFirstReachedTime/2.
%#show plannedDepartureTime/2.
%#show actualDepartureTime/2.
%#show lololol/8.
%#show navaid_sector/3.
%#show output_sec_len/3.
%#show output_sec/3.
%#show output_sec_avg/3.
%#show navaid_sector/3.

#show time/1.

