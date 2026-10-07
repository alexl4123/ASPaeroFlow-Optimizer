"""The two calls evaluate_solution.py makes into the trace: before the answer is applied, and after."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .trace import trajectory_of


def capture_before(evaluator, solutions, flight_ids, converted_navpoint_matrix, capacity_time_matrix,
                   system_loads, number_of_conflicts, controller_sector_diff_dict, optimization_dto) -> Dict[str, Any]:
    """State of the iteration before its answer is applied (the matrices still hold the old plan)."""
    time_index = int(controller_sector_diff_dict.get("time_index", -1))
    sector_index = int(controller_sector_diff_dict.get("sector_index", -1))
    hotspot: Dict[str, Any] = {"time": time_index, "sector": sector_index}
    if 0 <= sector_index < capacity_time_matrix.shape[0] and 0 <= time_index < capacity_time_matrix.shape[1]:
        hotspot["capacity"] = int(capacity_time_matrix[sector_index, time_index])
        hotspot["demand"] = int(system_loads[sector_index, time_index])
    prev = controller_sector_diff_dict.get("prev_sector_config", {}).get(sector_index, {})
    hotspot["vertices"] = list(prev.get("vertices", []))
    hotspot["overload"] = prev.get("overload")
    # The flights in the hotspot cell with the flight durations the candidate sort used
    # (iteration_step.py build_job: stable sort by duration, so ties fall to the lower flight number),
    # and how many of them were passed to the solver. Both matrices still hold the plan before the answer.
    instance_matrix = optimization_dto.get("converted_instance_matrix")
    durations = optimization_dto.get("flight_durations")
    if (instance_matrix is not None and durations is not None and 0 <= sector_index
            and 0 <= time_index < instance_matrix.shape[1]):
        cell = np.flatnonzero(instance_matrix[:, time_index] == sector_index)
        flights = [{"id": int(f), "duration": int(durations[f])} for f in cell]
        hotspot["flights"] = sorted(flights, key=lambda x: (x["duration"], x["id"]))
    if optimization_dto.get("xai_taken") is not None:
        hotspot["taken"] = len(optimization_dto["xai_taken"])

    subproblems = []
    for model, _restore, instance in solutions:
        chosen_paths = model.get_chosen_paths() if hasattr(model, "get_chosen_paths") else {}
        flights = sorted(set(int(f) for f in flight_ids) | set(chosen_paths))
        subproblems.append({
            "instance": instance,
            "chosen_config": int(str(model.get_sector_config().arguments[0])),
            "chosen_paths": chosen_paths,
            "clingo_cost": list(getattr(model, "cost", [])),
            "solve_seconds": getattr(model, "computation_time", None),
            "current_trajectories": {f: trajectory_of(converted_navpoint_matrix[f, :]) for f in flights},
        })

    parameters = {
        "additional_time_increase": optimization_dto.get("additional_time_increase"),
        "max_aircraft": optimization_dto.get("max_number_airplanes_considered_in_ASP"),
        "max_explored_vertices": getattr(evaluator, "_max_explored_vertices", None),
        "max_delay_per_iteration": getattr(evaluator, "_max_delay_per_iteration", None),
        "number_capacity_management_configs": getattr(evaluator, "number_capacity_management_configs", None),
        "failed_attempts_before": optimization_dto.get("counter_equal_solutions"),
    }
    return {"hotspot": hotspot, "subproblems": subproblems, "parameters": parameters,
            "overload_before": int(number_of_conflicts) if number_of_conflicts is not None else None}


def record_iteration(evaluator, pre: Dict[str, Any], iteration: int, accepted, output_dict: Dict[str, Any],
                     flight_changes: Dict[int, Any], sector_changes: Dict[str, Any]) -> None:
    trace = evaluator._xai_trace
    trace.write_encoding(evaluator.encoding)
    objectives = {k: v for k, v in output_dict.items() if k not in ("DIFF", "ITERATION-BACKUP")}
    objectives["OVERLOAD-BEFORE"] = pre["overload_before"]
    sector_changes = {k: v for k, v in (sector_changes or {}).items() if k != "accepted_solution"}
    trace.record(iteration=int(iteration), hotspot=pre["hotspot"], accepted=bool(accepted),
                 objectives=objectives, subproblems=pre["subproblems"],
                 flight_changes=flight_changes if accepted else {},
                 sector_changes=sector_changes if accepted else {},
                 parameters=pre["parameters"])
