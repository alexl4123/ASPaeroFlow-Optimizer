% NAIVE ENCODING - FIRST VERSION:
% Assumptions:
% Each time step = 1h
% Each airplane takes exactly 1h per sector, regardless of its type (uniform sectors and airplanes)
% Each airplane has a filed flight plan
% No environmental effects

% FACTS:
% flightPlan(<ID>,<TIME>,<LOCATION>)
% sector(<LOCATION>,<TIME>,<CAPACITY>)
% sectorEdge(<LOCATION-1>,<LOCATION-2>)
% airport(<LOCATION>)

sectorEdge(X,Y) :- sectorEdge(Y,X).
sectorEdge(X,X) :- airport(X).

% PRELIMINARIES:
% GET ALL TIME STEPS:
time(T) :- flightPlan(_,T,_).
time(T) :- sector(_,T,_).

flightPlanMaxTime(TMAX) :- TMAX = #max{T:flightPlan(_,T,_);T:sector(_,T,_)}.
effectiveLastTime(T+TMAX) :- flightPlanMaxTime(TMAX), additionalTime(T).
time(T+1) :- time(T), effectiveLastTime(TMAX), T < TMAX.

% GET ALL FLIHT IDS:
flightID(ID) :- flightPlan(ID,_,_).

% MORE TIME:
%maxTimeOriginal(T) :- T = #max{T':flightPlan(_,T',_)}.
%time(T+1) :- time(T), maxTimeOriginal(TMAX), T <= 3*TMAX.

% NORMAL FLIGHT:
flight(ID,T,X) :- flightPlan(ID,T,X), not reroute(ID).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% WE ENFORCE THE OPTIMAL MODEL %%%%%%%%%%%%%%%%%%%
%:- reroute(1).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% XAI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assuming that before we got reroute(2).
% Now we want to ask, WHY should Airplane with ID=2 be rerouted?
% So we enforce, it will not be rerouted
%:- reroute(2).
% LIMIT SECTOR CAPACITY (soft constraint for XAI):
overload(X,T,LOAD-C) :- sector(X,T,C), time(T), #count{ID:flight(ID,T,X)} = LOAD, LOAD > C.
%:- overload(_,_).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% REROUTE - YES/NO:
{reroute(ID)} :- flightPlan(ID,_,_).
{flight(ID,T,X):sector(X,_,_)}1 :- reroute(ID), time(T).

% FLIGHT MUST OCCUR:
flightOccurs(ID) :- flight(ID,_,_).
:- flightID(ID), not flightOccurs(ID).

% ----------------- ONLY FOR REROUTED PLANES --------------------
% NO FLIGHT BEFORE PLANNED START TIME:
plannedDepartureTime(ID,T) :- reroute(ID), T = #min{T':flightPlan(ID,T',_)}.
:- reroute(ID), flight(ID,T,_), plannedDepartureTime(ID,T'), T < T'.

% FLIGHT STARTS AT ORIGIN:
plannedOrigin(ID,X) :- reroute(ID),plannedDepartureTime(ID,T),flightPlan(ID,T,X).
actualDepartureTime(ID,T) :- reroute(ID), T=#min{T':flight(ID,T',_)}.
%:- reroute(ID), actualDepartureTime(ID,T), flight(ID,T,X), plannedOrigin(ID,X'), X != X'.
flight(ID,T,X) :- reroute(ID), flightPlan(ID,T,X), plannedOrigin(ID,X).

% FLIGHT ENDS AT DESTINATION:
plannedDestination(ID,X) :- reroute(ID), T=#max{T':flightPlan(ID,T',_)}, flightPlan(ID,T,X).
actualArrivalTime(ID,T) :- flightID(ID), T = #max{T':flight(ID,T',_)}.
:- reroute(ID), actualArrivalTime(ID,T), flight(ID,T,X), plannedDestination(ID,X'), X != X'.

% NO FLIGHT AFTER DESTINATION REACHED:
actualDestiantionFirstReachedTime(ID,T) :- reroute(ID), plannedDestination(ID,X), T = #min{T':flight(ID,T',X)}.
:- reroute(ID), flight(ID,T,_), actualDestiantionFirstReachedTime(ID,T'), T > T'.

% MUST BE PATH:
:- flight(ID,T,X), flight(ID,T+1,X'), not sectorEdge(X,X').

% NEVER TWICE AT SAME LOCATION (EXCEPT AIRPORTS):
:- flight(ID,T,X), flight(ID,T',X), T != T', not airport(X).

% NO GAPS:
:- time(T), actualDepartureTime(ID,T'), T > T', actualArrivalTime(ID,T''), T < T'', not flight(ID,T,_).


% ------------- OPTIMIZATION ----------------
plannedArrivalTime(ID,T) :- flightID(ID), T = #max{T':flightPlan(ID,T',_)}.
arrivalDelay(ID,Y-T) :- plannedArrivalTime(ID,T), actualArrivalTime(ID,Y).

% ------------------------------------------------------------------------------
% PRIMARY WEAK CONSTRAINT:
:~ overload(X,T,OVER). [OVER@2,X,T]
% SECONDARY WEAK CONSTRAINT:
:~ arrivalDelay(ID,DIFF). [1@1,ID]

#show flight/3.
#show reroute/1.
%#show actualArrivalTime/2.
%#show plannedArrivalTime/2.
%#show arrivalDelay/2.
#show overload/3.
#show arrivalDelay/2.
#show time/1.
