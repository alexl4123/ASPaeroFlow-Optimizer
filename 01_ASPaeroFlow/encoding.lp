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
%sectorEdge(X,X) :- airport(X).

% GET ALL FLIGHT IDS:

flightID(ID) :- flightPlan(ID,_,_).

% NORMAL FLIGHT:
%flight(ID,T,X) :- flightPlan(ID,T,X), not reroute(ID).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% WE ENFORCE THE OPTIMAL MODEL %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LIMIT SECTOR CAPACITY (soft constraint for XAI):

overload(X,T,LOAD-C) :- sector(X,T,C), #count{ID:flight(ID,T,X),flight(ID,T-1,X2), X!=X2} = LOAD, LOAD > C.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% REROUTE - YES/NO:
%{reroute(ID)} :- flightPlan(ID,_,_).

reroute(ID) :- flightPlan(ID,_,_).

1{path(ID,P):paths(ID,P)}1 :- reroute(ID).

flight(ID,T + T',X) :- reroute(ID), path(ID,P), actualDepartureTime(ID, T), sectorFlight(ID,T',X,P).


% FLIGHT MUST OCCUR:
flightOccurs(ID) :- flight(ID,_,_).
%:- flightID(ID), not flightOccurs(ID).

% MUST BE PATH:
%:- flight(ID,T,X), flight(ID,T+1,X'), not sectorEdge(X,X').

% ------------- OPTIMIZATION ----------------
arrivalDelay(ID,Y-T) :- path(ID,P), plannedArrivalTime(ID,T), actualArrivalTime(ID,Y,P).

% ------------------------------------------------------------------------------
% PRIMARY WEAK CONSTRAINT:
:~ overload(X,T,OVER). [OVER@2,X,T]
% SECONDARY WEAK CONSTRAINT:
:~ arrivalDelay(ID,DIFF). [DIFF@1,ID]

%#show flight/3.
%#show reroute/1.
%#show actualArrivalTime/2.
%#show plannedArrivalTime/2.
%#show arrivalDelay/2.
#show overload/3.
%#show arrivalDelay/2.
%#show time/1.
#show path/2.