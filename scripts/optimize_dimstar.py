"""Constrained mixed-variable L/D optimization for Project Dimstar.

The script enumerates wing/tail airfoil pairs and uses differential evolution
for the continuous geometry. It never modifies the input project. Completed
airfoil pairs are checkpointed, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from flightlab.project import AircraftProject
from flightlab.project_analysis import (
    TrimNotPossibleError,
    run_design_point,
    surface_section_cl_max,
)


VARIABLE_NAMES = (
    "wing_chord_m",
    "outer_taper_ratio",
    "tip_elevation_m",
    "tip_twist_deg",
    "tail_chord_m",
    "tail_twist_deg",
    "cg_x_m",
)


@dataclass(frozen=True)
class Targets:
    static_margin_min: float = 0.10
    static_margin_max: float = 0.20
    span_efficiency_min: float = 0.97
    control_tolerance_deg: float = 0.05
    penalty_weight: float = 1_000.0


@dataclass(frozen=True)
class Metrics:
    static_margin: float
    span_efficiency: float
    trim_deflection_deg: float
    alpha_deg: float
    cd_total: float
    lift_to_drag: float
    stall_speed_estimate_mps: float
    x_cg_m: float
    x_np_m: float
    mac_m: float
    mass_kg: float
    adjustable_mass_x_m: float


def comma_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("give at least one comma-separated value")
    return values


def numeric_pair(value: str) -> tuple[float, float]:
    try:
        lower, upper = (float(item.strip()) for item in value.split(",", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("bounds must be LOWER,UPPER") from exc
    if not lower < upper:
        raise argparse.ArgumentTypeError("the lower bound must be less than the upper bound")
    return lower, upper


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def find_surfaces(project: AircraftProject):
    wing = project.surface_named("Main wing") or project.surface("wing")
    tail = project.surface_named("Horizontal tail") or project.surface("tail")
    if wing is None or len(wing.stations) != 3:
        raise ValueError("optimization requires a three-station main wing")
    if tail is None or len(tail.stations) < 2:
        raise ValueError("optimization requires a horizontal tail with at least two stations")
    return wing, tail


def set_actual_cg(
    project: AircraftProject,
    target_x_cg: float,
    adjustable_mass_name: str,
) -> float:
    """Move one point mass so the complete aircraft reaches target_x_cg."""
    item = project.mass_named(adjustable_mass_name)
    if item is None:
        raise ValueError(f"adjustable mass {adjustable_mass_name!r} does not exist")
    if item.mass is None or item.distributed not in {"", "point"}:
        raise ValueError("the adjustable mass must be a point mass with a fixed total mass")

    original_x = item.x
    components = project.components()
    total_mass = sum(component.mass for component in components)
    total_moment = sum(component.mass * component.x for component in components)
    fixed_moment = total_moment - item.mass * original_x
    item.x = (target_x_cg * total_mass - fixed_moment) / item.mass
    return item.x


def apply_design(
    source: AircraftProject,
    vector: Sequence[float],
    wing_airfoil: str,
    tail_airfoil: str,
    adjustable_mass_name: str,
) -> tuple[AircraftProject, float]:
    project = deepcopy(source)
    wing, tail = find_surfaces(project)
    wing_chord, taper, tip_z, tip_twist, tail_chord, tail_twist, cg_x = vector

    # Only the horizontal tail is the pitch control. The source currently
    # gives the wing, horizontal tail, and vertical tail one shared command.
    for surface in project.surfaces:
        surface.trim_control = "fixed"
    tail.trim_control = "elevator"

    wing.stations[0].chord = wing_chord
    wing.stations[1].chord = wing_chord
    tip_chord = wing_chord * taper
    wing.stations[2].chord = tip_chord
    wing.stations[2].z = tip_z
    wing.stations[2].twist_deg = tip_twist

    # Preserve the baseline straight trailing edge on the tapered outer panel.
    root_le = wing.stations[0].x_le
    wing.stations[0].x_le = root_le
    wing.stations[1].x_le = root_le
    wing.stations[2].x_le = root_le + wing_chord - tip_chord
    for station in wing.stations:
        station.airfoil = wing_airfoil

    for station in tail.stations:
        station.chord = tail_chord
        station.twist_deg = tail_twist
        station.airfoil = tail_airfoil

    adjustable_mass_x = set_actual_cg(project, cg_x, adjustable_mass_name)
    return project, adjustable_mass_x


def estimate_stall_speed(
    project: AircraftProject,
    case,
    point,
    iterations: int = 4,
) -> float:
    """Estimate 1g stall from local wing cl/cl_max with Reynolds correction."""
    wing, _ = find_surfaces(project)
    view = point.trim.solution.surface(wing.name)
    positive_cl = np.maximum(view.cl, 0.0)
    speed = max(1.0, 0.25 * case.speed)
    for _ in range(iterations):
        re_scale = speed / case.speed
        scaled_solution = replace(
            point.trim.solution,
            Re=point.trim.solution.Re * re_scale,
        )
        cl_max = surface_section_cl_max(
            project,
            wing,
            scaled_solution,
            replace(case, speed=speed),
        )
        loading_ratio = float(np.max(positive_cl / np.maximum(cl_max, 1.0e-9)))
        speed = case.speed * math.sqrt(max(loading_ratio, 1.0e-9))
    return float(speed)


def refine_stall_speed(
    project: AircraftProject,
    case_name: str,
    initial_speed: float,
    ns: int,
    nc: int,
    iterations: int = 3,
) -> float:
    """Refine the stall estimate with trimmed analyses near the predicted stall."""
    case = project.case(case_name)
    wing, _ = find_surfaces(project)
    speed = initial_speed
    for _ in range(iterations):
        low_speed_case = replace(case, speed=float(speed))
        point = run_design_point(project, low_speed_case, ns=ns, nc=nc)
        view = point.trim.solution.surface(wing.name)
        cl_max = surface_section_cl_max(
            project, wing, point.trim.solution, low_speed_case
        )
        ratio = float(np.max(np.maximum(view.cl, 0.0) / np.maximum(cl_max, 1.0e-9)))
        speed *= math.sqrt(max(ratio, 1.0e-9))
    return float(speed)


class DesignEvaluator:
    """Pickleable objective used by serial or multiprocessing DE workers."""

    def __init__(
        self,
        project: AircraftProject,
        case_name: str,
        wing_airfoil: str,
        tail_airfoil: str,
        adjustable_mass_name: str,
        mass_x_bounds: tuple[float, float],
        targets: Targets,
        objective: str,
        stall_speed_weight: float,
        ns: int,
        nc: int,
    ) -> None:
        self.project = project
        self.case_name = case_name
        self.wing_airfoil = wing_airfoil
        self.tail_airfoil = tail_airfoil
        self.adjustable_mass_name = adjustable_mass_name
        self.mass_x_bounds = mass_x_bounds
        self.targets = targets
        self.objective = objective
        self.stall_speed_weight = stall_speed_weight
        self.ns = ns
        self.nc = nc
        self._cache: dict[tuple[float, ...], tuple[float, Metrics | None, str]] = {}

    def evaluate(self, vector: Sequence[float]) -> tuple[float, Metrics | None, str]:
        key = tuple(float(value) for value in vector)
        if key in self._cache:
            return self._cache[key]
        try:
            candidate, mass_x = apply_design(
                self.project,
                key,
                self.wing_airfoil,
                self.tail_airfoil,
                self.adjustable_mass_name,
            )
            if not self.mass_x_bounds[0] <= mass_x <= self.mass_x_bounds[1]:
                distance = max(self.mass_x_bounds[0] - mass_x, mass_x - self.mass_x_bounds[1])
                result = (1.0e8 + 1.0e8 * distance**2, None, "adjustable mass outside bounds")
                self._cache[key] = result
                return result

            point = run_design_point(candidate, candidate.case(self.case_name), ns=self.ns, nc=self.nc)
            case = candidate.case(self.case_name)
            _, _, mac = candidate.reference_quantities()
            metrics = Metrics(
                static_margin=float(point.trim.static_margin),
                span_efficiency=float(point.trim.solution.e_inv),
                trim_deflection_deg=float(point.trim.trim_deflection),
                alpha_deg=float(point.trim.alpha),
                cd_total=float(point.CD_total),
                lift_to_drag=float(point.lift_to_drag),
                stall_speed_estimate_mps=estimate_stall_speed(candidate, case, point),
                x_cg_m=float(point.trim.x_cg),
                x_np_m=float(point.trim.x_np),
                mac_m=float(mac),
                mass_kg=float(point.trim.mass),
                adjustable_mass_x_m=float(mass_x),
            )
            performance = (
                -metrics.lift_to_drag
                if self.objective == "lift-to-drag"
                else metrics.cd_total
            )
            performance += self.stall_speed_weight * metrics.stall_speed_estimate_mps
            score = performance + self.targets.penalty_weight * constraint_penalty(
                metrics, self.targets
            )
            result = (float(score), metrics, "")
        except (
            TrimNotPossibleError,
            ValueError,
            KeyError,
            RuntimeError,
            OverflowError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            result = (1.0e12, None, str(exc))
        self._cache[key] = result
        return result

    def __call__(self, vector: Sequence[float]) -> float:
        return self.evaluate(vector)[0]


def constraint_penalty(metrics: Metrics, targets: Targets) -> float:
    margin_violation = max(
        targets.static_margin_min - metrics.static_margin,
        metrics.static_margin - targets.static_margin_max,
        0.0,
    ) / 0.01
    efficiency_violation = max(
        targets.span_efficiency_min - metrics.span_efficiency, 0.0
    ) / 0.01
    # Zero deflection is an optimization target, not merely a bounded
    # constraint. Keeping this residual active inside the feasibility
    # tolerance prevents solutions from settling on the +/- tolerance edge.
    control_residual = metrics.trim_deflection_deg / 1.0
    return margin_violation**2 + efficiency_violation**2 + control_residual**2


def is_feasible(metrics: Metrics | None, targets: Targets) -> bool:
    return metrics is not None and (
        targets.static_margin_min <= metrics.static_margin <= targets.static_margin_max
        and metrics.span_efficiency >= targets.span_efficiency_min
        and abs(metrics.trim_deflection_deg) <= targets.control_tolerance_deg
    )


def case_metrics(project: AircraftProject, ns: int, nc: int) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for case in project.cases:
        try:
            point = run_design_point(project, case, ns=ns, nc=nc)
            results[case.name] = {
                "trim_converged": True,
                "alpha_deg": float(point.trim.alpha),
                "trim_deflection_deg": float(point.trim.trim_deflection),
                "static_margin": float(point.trim.static_margin),
                "span_efficiency": float(point.trim.solution.e_inv),
                "cd_total": float(point.CD_total),
                "lift_to_drag": float(point.lift_to_drag),
            }
        except (
            TrimNotPossibleError,
            ValueError,
            RuntimeError,
            OverflowError,
            np.linalg.LinAlgError,
        ) as exc:
            results[case.name] = {"trim_converged": False, "error": str(exc)}
    return results


def checkpoint_path(output_dir: Path, wing_airfoil: str, tail_airfoil: str) -> Path:
    return output_dir / "checkpoints" / f"{safe_name(wing_airfoil)}__{safe_name(tail_airfoil)}.json"


def save_current_candidate(
    intermediate_result,
    evaluator: DesignEvaluator,
    source: AircraftProject,
    wing_airfoil: str,
    tail_airfoil: str,
    args: argparse.Namespace,
    targets: Targets,
    stage: str,
) -> None:
    """Save the latest generation so Ctrl+C still leaves an inspectable design."""
    vector = np.asarray(intermediate_result.x, dtype=float)
    score, metrics, error = evaluator.evaluate(vector)
    candidate, mass_x = apply_design(
        source,
        vector,
        wing_airfoil,
        tail_airfoil,
        args.adjustable_mass,
    )
    candidate.save(args.output_dir / "current_best_design.flightlab.json")
    progress = {
        "status": "in_progress",
        "stage": stage,
        "note": (
            "Metrics use the optimization panel resolution and have not yet "
            "received final high-resolution verification."
        ),
        "wing_airfoil": wing_airfoil,
        "tail_airfoil": tail_airfoil,
        "variables": dict(zip(VARIABLE_NAMES, map(float, vector))),
        "score": float(score),
        "search_feasible": is_feasible(metrics, targets),
        "search_metrics": asdict(metrics) if metrics else None,
        "adjustable_mass_x_m": float(mass_x),
        "iteration": int(getattr(intermediate_result, "nit", 0)),
        "function_evaluations": int(getattr(intermediate_result, "nfev", 0)),
        "error": error,
    }
    (args.output_dir / "current_result.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
    )


def optimize_pair(
    source: AircraftProject,
    wing_airfoil: str,
    tail_airfoil: str,
    bounds: Sequence[tuple[float, float]],
    args: argparse.Namespace,
    targets: Targets,
) -> dict:
    evaluator = DesignEvaluator(
        source,
        args.case,
        wing_airfoil,
        tail_airfoil,
        args.adjustable_mass,
        args.mass_x_bounds,
        targets,
        args.objective,
        args.stall_speed_weight,
        args.ns,
        args.nc,
    )

    def differential_callback(intermediate_result):
        save_current_candidate(
            intermediate_result,
            evaluator,
            source,
            wing_airfoil,
            tail_airfoil,
            args,
            targets,
            "differential_evolution",
        )

    result = differential_evolution(
        evaluator,
        bounds=bounds,
        strategy="best1bin",
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=args.tol,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=args.seed,
        workers=args.workers,
        updating="immediate" if args.workers == 1 else "deferred",
        polish=False,
        disp=args.verbose,
        callback=differential_callback,
    )

    local_result = None
    if args.local_maxiter > 0:
        def local_callback(intermediate_result):
            save_current_candidate(
                intermediate_result,
                evaluator,
                source,
                wing_airfoil,
                tail_airfoil,
                args,
                targets,
                "local_refinement",
            )

        local_result = minimize(
            evaluator,
            result.x,
            method="COBYQA",
            bounds=bounds,
            callback=local_callback,
            options={"maxiter": args.local_maxiter, "disp": args.verbose},
        )
        if local_result.fun < result.fun:
            best_x = local_result.x
            best_score = float(local_result.fun)
        else:
            best_x = result.x
            best_score = float(result.fun)
    else:
        best_x = result.x
        best_score = float(result.fun)

    _, metrics, error = evaluator.evaluate(best_x)
    candidate, mass_x = apply_design(
        source,
        best_x,
        wing_airfoil,
        tail_airfoil,
        args.adjustable_mass,
    )
    verification = case_metrics(candidate, args.verify_ns, args.verify_nc)
    verified_stall_speed = None
    if metrics is not None:
        try:
            verified_stall_speed = refine_stall_speed(
                candidate,
                args.case,
                metrics.stall_speed_estimate_mps,
                args.verify_ns,
                args.verify_nc,
            )
        except (TrimNotPossibleError, ValueError, RuntimeError, np.linalg.LinAlgError):
            pass
    cruise_verified = verification.get(args.case, {})
    verified_feasible = (
        cruise_verified.get("trim_converged", False)
        and targets.static_margin_min
        <= cruise_verified.get("static_margin", -math.inf)
        <= targets.static_margin_max
        and cruise_verified.get("span_efficiency", -math.inf) >= targets.span_efficiency_min
        and abs(cruise_verified.get("trim_deflection_deg", math.inf))
        <= targets.control_tolerance_deg
    )
    off_design_trimmed = all(row.get("trim_converged", False) for row in verification.values())

    return {
        "wing_airfoil": wing_airfoil,
        "tail_airfoil": tail_airfoil,
        "variables": dict(zip(VARIABLE_NAMES, map(float, best_x))),
        "score": best_score,
        "objective": args.objective,
        "stall_speed_weight": args.stall_speed_weight,
        "search_feasible": is_feasible(metrics, targets),
        "verified_feasible": bool(verified_feasible),
        "all_cases_trimmed": off_design_trimmed,
        "search_metrics": asdict(metrics) if metrics else None,
        "verification": verification,
        "verified_stall_speed_mps": verified_stall_speed,
        "adjustable_mass_x_m": float(mass_x),
        "error": error,
        "differential_evolution": {
            "success": bool(result.success),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
        },
        "local_refinement": None
        if local_result is None
        else {
            "success": bool(local_result.success),
            "message": str(local_result.message),
            "nit": int(local_result.nit),
            "nfev": int(local_result.nfev),
        },
    }


def save_summary_csv(results: Sequence[dict], path: Path, case_name: str) -> None:
    fields = [
        "wing_airfoil",
        "tail_airfoil",
        *VARIABLE_NAMES,
        "score",
        "verified_feasible",
        "all_cases_trimmed",
        "static_margin",
        "span_efficiency",
        "trim_deflection_deg",
        "cd_total",
        "lift_to_drag",
        "stall_speed_mps",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            verified = result.get("verification", {}).get(case_name, {})
            writer.writerow(
                {
                    "wing_airfoil": result["wing_airfoil"],
                    "tail_airfoil": result["tail_airfoil"],
                    **result["variables"],
                    "score": result["score"],
                    "verified_feasible": result["verified_feasible"],
                    "all_cases_trimmed": result["all_cases_trimmed"],
                    "static_margin": verified.get("static_margin"),
                    "span_efficiency": verified.get("span_efficiency"),
                    "trim_deflection_deg": verified.get("trim_deflection_deg"),
                    "cd_total": verified.get("cd_total"),
                    "lift_to_drag": verified.get("lift_to_drag"),
                    "stall_speed_mps": result.get("verified_stall_speed_mps"),
                }
            )


def save_best_completed(
    results: Sequence[dict],
    source: AircraftProject,
    args: argparse.Namespace,
) -> dict | None:
    """Write the best fully verified result among completed airfoil pairs."""
    feasible = [
        result
        for result in results
        if result["verified_feasible"] and result["all_cases_trimmed"]
    ]
    ranked = feasible or [
        result for result in results if result.get("search_metrics") is not None
    ]
    if not ranked:
        return None
    best = min(ranked, key=lambda result: result["score"])
    best_vector = [best["variables"][name] for name in VARIABLE_NAMES]
    best_project, _ = apply_design(
        source,
        best_vector,
        best["wing_airfoil"],
        best["tail_airfoil"],
        args.adjustable_mass,
    )
    best_project.notes = (
        best_project.notes
        + f"\nOptimized by optimize_dimstar.py for {args.case}; "
        + f"wing={best['wing_airfoil']}, tail={best['tail_airfoil']}."
    ).strip()
    best_project.save(args.output_dir / "best_design.flightlab.json")
    (args.output_dir / "best_result.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", type=Path, default=Path("ProjectDimstar0.5.flightlab.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("dimstar_optimization"))
    parser.add_argument("--case", default="Cruise")
    parser.add_argument("--adjustable-mass", default="The entire plane")
    parser.add_argument(
        "--wing-airfoils",
        type=comma_list,
        default=comma_list("e168,esa40,sd7037,e212,fx62k131,s1223,clarky,naca2412"),
    )
    parser.add_argument(
        "--stall-speed-weight",
        type=float,
        default=0.02,
        help="objective cost per m/s of estimated 1g stall speed; use 0 to disable",
    )
    parser.add_argument(
        "--tail-airfoils",
        type=comma_list,
        default=comma_list("naca0012,naca64a210"),
    )
    parser.add_argument("--wing-chord-bounds", type=numeric_pair, default=(0.14, 0.30))
    parser.add_argument("--taper-bounds", type=numeric_pair, default=(0.25, 1.00))
    parser.add_argument("--tip-elevation-bounds", type=numeric_pair, default=(0.00, 0.20))
    parser.add_argument("--tip-twist-bounds", type=numeric_pair, default=(-6.0, 4.0))
    parser.add_argument("--tail-chord-bounds", type=numeric_pair, default=(0.06, 0.20))
    parser.add_argument("--tail-twist-bounds", type=numeric_pair, default=(-8.0, 8.0))
    parser.add_argument("--cg-bounds", type=numeric_pair, default=(0.02, 0.14))
    parser.add_argument("--mass-x-bounds", type=numeric_pair, default=(-0.10, 0.35))
    parser.add_argument("--static-margin-bounds", type=numeric_pair, default=(0.10, 0.20))
    parser.add_argument("--span-efficiency-min", type=float, default=0.97)
    parser.add_argument("--control-tolerance", type=float, default=0.05)
    parser.add_argument("--penalty-weight", type=float, default=1_000.0)
    parser.add_argument(
        "--objective",
        choices=("lift-to-drag", "drag"),
        default="lift-to-drag",
        help="maximize Cruise L/D (default) or minimize total drag coefficient",
    )
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument("--tol", type=float, default=5.0e-4)
    parser.add_argument("--local-maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--workers", type=int, default=-1, help="1 for serial; -1 for all CPU cores")
    parser.add_argument("--ns", type=int, default=16, help="spanwise panels during optimization")
    parser.add_argument("--nc", type=int, default=3, help="chordwise panels during optimization")
    parser.add_argument("--verify-ns", type=int, default=40)
    parser.add_argument("--verify-nc", type=int, default=6)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stall_speed_weight < 0.0:
        raise ValueError("stall-speed weight must be zero or positive")
    source = AircraftProject.load(args.project)
    source.case(args.case)
    find_surfaces(source)
    if source.mass_named(args.adjustable_mass) is None:
        raise ValueError(f"adjustable mass {args.adjustable_mass!r} was not found")

    # Resolve every candidate now so typos fail before a long optimization.
    for airfoil_name in set(args.wing_airfoils + args.tail_airfoils):
        source.section(airfoil_name)

    bounds = [
        args.wing_chord_bounds,
        args.taper_bounds,
        args.tip_elevation_bounds,
        args.tip_twist_bounds,
        args.tail_chord_bounds,
        args.tail_twist_bounds,
        args.cg_bounds,
    ]
    targets = Targets(
        static_margin_min=args.static_margin_bounds[0],
        static_margin_max=args.static_margin_bounds[1],
        span_efficiency_min=args.span_efficiency_min,
        control_tolerance_deg=args.control_tolerance,
        penalty_weight=args.penalty_weight,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    config = {
        "project": str(args.project.resolve()),
        "variables": VARIABLE_NAMES,
        "bounds": dict(zip(VARIABLE_NAMES, bounds)),
        "targets": asdict(targets),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    signature_data = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "project_sha256": hashlib.sha256(args.project.read_bytes()).hexdigest(),
        "bounds": config["bounds"],
        "targets": config["targets"],
        "case": args.case,
        "objective": args.objective,
        "stall_speed_weight": args.stall_speed_weight,
        "adjustable_mass": args.adjustable_mass,
        "mass_x_bounds": args.mass_x_bounds,
        "optimizer": {
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "tol": args.tol,
            "local_maxiter": args.local_maxiter,
            "seed": args.seed,
            "ns": args.ns,
            "nc": args.nc,
            "verify_ns": args.verify_ns,
            "verify_nc": args.verify_nc,
        },
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config["run_signature"] = run_signature
    (args.output_dir / "optimization_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    results = []
    pairs = [(wing, tail) for wing in args.wing_airfoils for tail in args.tail_airfoils]
    for pair_number, (wing_airfoil, tail_airfoil) in enumerate(pairs, start=1):
        checkpoint = checkpoint_path(args.output_dir, wing_airfoil, tail_airfoil)
        saved = None
        if checkpoint.exists() and not args.no_resume:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved is not None and saved.get("run_signature") == run_signature:
            print(f"[{pair_number}/{len(pairs)}] loading {wing_airfoil} / {tail_airfoil}")
            result = saved
        else:
            print(f"[{pair_number}/{len(pairs)}] optimizing {wing_airfoil} / {tail_airfoil}")
            try:
                result = optimize_pair(
                    source, wing_airfoil, tail_airfoil, bounds, args, targets
                )
            except KeyboardInterrupt:
                print("\nOptimization stopped by user.")
                print(
                    f"Latest generation: {args.output_dir / 'current_result.json'}"
                )
                print(
                    "Latest generation design: "
                    f"{args.output_dir / 'current_best_design.flightlab.json'}"
                )
                if results:
                    save_summary_csv(
                        results, args.output_dir / "airfoil_results.csv", args.case
                    )
                    save_best_completed(results, source, args)
                    print(
                        "Best completed airfoil pair: "
                        f"{args.output_dir / 'best_result.json'}"
                    )
                raise SystemExit(130)
            result["run_signature"] = run_signature
            checkpoint.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            completed_vector = [result["variables"][name] for name in VARIABLE_NAMES]
            completed_project, _ = apply_design(
                source,
                completed_vector,
                result["wing_airfoil"],
                result["tail_airfoil"],
                args.adjustable_mass,
            )
            completed_project.save(
                args.output_dir / "current_best_design.flightlab.json"
            )
            (args.output_dir / "current_result.json").write_text(
                json.dumps({"status": "pair_completed", **result}, indent=2) + "\n",
                encoding="utf-8",
            )
        results.append(result)
        save_summary_csv(results, args.output_dir / "airfoil_results.csv", args.case)
        save_best_completed(results, source, args)

    best = save_best_completed(results, source, args)
    if best is None:
        raise RuntimeError("all candidate optimizations failed; inspect checkpoint errors")

    status = "feasible" if best["verified_feasible"] else "best available, not feasible"
    cruise = best["verification"][args.case]
    print(f"Best result ({status}): {best['wing_airfoil']} / {best['tail_airfoil']}")
    print(
        f"  SM={cruise.get('static_margin', math.nan):.4f}, "
        f"e={cruise.get('span_efficiency', math.nan):.4f}, "
        f"control={cruise.get('trim_deflection_deg', math.nan):+.4f} deg, "
        f"L/D={cruise.get('lift_to_drag', math.nan):.4f}, "
        f"CD={cruise.get('cd_total', math.nan):.5f}, "
        f"stall={best.get('verified_stall_speed_mps') or math.nan:.3f} m/s"
    )
    print(f"Saved {args.output_dir / 'best_design.flightlab.json'}")


if __name__ == "__main__":
    main()
