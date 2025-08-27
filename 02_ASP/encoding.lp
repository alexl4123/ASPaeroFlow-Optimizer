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
% navaid_sector(<NAVAID-ID>,<SECTOR-ID>)

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
time(T+1) :- time(T), maxTime(TMAX), T <= TMAX.

% NORMAL FLIGHT:
navpointFlight(ID,T,X) :- navpointFlightPlan(ID,T,X), not reroute(ID).

% LIMIT SECTOR CAPACITY (hard constraint modelling):
overload(X,T,LOAD-C) :- sector(X,T,C), #count{ID:flight(ID,T,X),flight(ID,T-1,X2), X!=X2} = LOAD, LOAD > C.

% REROUTE - YES/NO:
{reroute(ID)} :- navpointFlightPlan(ID,_,_).
{navpointFlight(ID,T,X):navpoint(X)}1 :- reroute(ID), time(T).

% ----------------- ONLY FOR REROUTED PLANES --------------------
% NO FLIGHT BEFORE PLANNED START TIME:
plannedDepartureTime(ID,T) :- reroute(ID), T = #min{T':navpointFlightPlan(ID,T',_)}.
:- reroute(ID), navpointFlight(ID,T,_), plannedDepartureTime(ID,T'), T < T'.

% FLIGHT STARTS AT ORIGIN:
plannedOrigin(ID,X) :- reroute(ID),plannedDepartureTime(ID,T),navpointFlightPlan(ID,T,X).
navpointFlight(ID,T,X) :- reroute(ID), navpointFlightPlan(ID,T,X), plannedOrigin(ID,X).
navpointFlight(ID,T,X) :- reroute(ID), time(T), plannedDepartureTime(ID,T'), actualDepartureTime(ID,T''), T >= T', T <= T'', plannedOrigin(ID,X).

% FLIGHT ENDS AT DESTINATION:
plannedDestination(ID,X) :- reroute(ID), T=#max{T':navpointFlightPlan(ID,T',_)}, navpointFlightPlan(ID,T,X).
actualArrivalTime(ID,T) :- flightID(ID), T = #max{T':navpointFlight(ID,T',_)}.
:- reroute(ID), actualArrivalTime(ID,T), navpointFlight(ID,T,X), plannedDestination(ID,X'), X != X'.

% NO FLIGHT AFTER DESTINATION REACHED:
actualDestinationFirstReachedTime(ID,T) :- reroute(ID), plannedDestination(ID,X), T = #min{T':navpointFlight(ID,T',X)}.
:- reroute(ID), navpointFlight(ID,T,_), actualDestinationFirstReachedTime(ID,T'), T > T'.

% MUST BE PATH:
navpointSeq(ID,T,X,X'') :- navpointFlight(ID,T,X), T'' = #min{T',X': navpointFlight(ID,T',X'), T' > T}, navpointFlight(ID,T'',X'').
:- navpointFlight(ID,T,X), navpointFlight(ID,T',X'), T' > T, navpointSeq(ID,T,X,X'), airplane(ID,S), D = T'-T, not navpointEdge(X,X',S,D).


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
flight(ID,T,S) :- navpointFlight(ID,T,X), navaid_sector(X,S).
flight(ID,T',S) :- navpointFlight(ID,T,X), navpointFlight(ID,T+D,X'), navpointSeq(ID,X,X'), navaid_sector(X,S), airplane(ID,SPEED), navpointEdge(X,X',SPEED,D), time(T'), T' > T, T' <= T + D/2. 
flight(ID,T',S') :- navpointFlight(ID,T,X), navpointFlight(ID,T+D,X'), navpointSeq(ID,X,X'), navaid_sector(X',S'), airplane(ID,SPEED), navpointEdge(X,X',SPEED,D), time(T'), T' < T+D, T' > T + D/2. 


% ------------------------------------------------------------------------------
% PRIMARY WEAK CONSTRAINT:
:~ overload(X,T,OVER). [OVER@3,X,T]
% SECONDARY WEAK CONSTRAINT:
:~ arrivalDelay(ID,DIFF). [DIFF@2,ID]
% OPTIMIZE FOR MINIMUM TRAVEL TIME:
:~ actualFlightTime(ID,TA), plannedFlightTime(ID,TP), DIFF = TA - TP. [DIFF@1,ID]

%#show flight/3.
#show navpointFlight/3.
%#show reroute/1.
%#show navpointSeq/3.
%#show actualArrivalTime/2.
%#show plannedArrivalTime/2.
%#show arrivalDelay/2.
%#show overload/3.
%#show timeStepGranularities/1.
%#show time/1.
%#show actualDestinationFirstReachedTime/2.
#show plannedDepartureTime/2.
#show actualDepartureTime/2.
