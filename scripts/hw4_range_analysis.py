"""Compare maximum-efficiency and maximum-range trim speeds for HW4."""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from flightlab.project import AircraftProject
from flightlab.project_analysis import TrimNotPossibleError, run_design_point


ROOT = Path(__file__).resolve().parent
PROJECT_PATH = ROOT / "hw4.flightlab (2).json"
OUTPUT_DIR = ROOT / "plot"


def main() -> None:
    project = AircraftProject.load(PROJECT_PATH)
    case = project.case("hw4")

    trial_speeds = np.arange(40.0, 110.1, 2.0)
    valid_speeds = []
    valid_lift_to_drag = []
    for speed in trial_speeds:
        try:
            point = run_design_point(project, replace(case, speed=float(speed)))
        except TrimNotPossibleError:
            print(f"Skipping {speed:.0f} m/s: no accepted trim solution")
            continue
        valid_speeds.append(speed)
        valid_lift_to_drag.append(point.lift_to_drag)
    speeds = np.asarray(valid_speeds)
    lift_to_drag = np.asarray(valid_lift_to_drag)
    range_metric = speeds * lift_to_drag

    def metrics(speed: float) -> tuple[float, float]:
        point = run_design_point(project, replace(case, speed=float(speed)))
        return point.lift_to_drag, speed * point.lift_to_drag

    efficiency_index = int(np.argmax(lift_to_drag))
    range_index = int(np.argmax(range_metric))
    max_efficiency = minimize_scalar(
        lambda speed: -metrics(speed)[0],
        bounds=(speeds[efficiency_index - 1], speeds[efficiency_index + 1]),
        method="bounded",
        options={"xatol": 1e-4},
    )
    max_range = minimize_scalar(
        lambda speed: -metrics(speed)[1],
        bounds=(speeds[range_index - 1], speeds[range_index + 1]),
        method="bounded",
        options={"xatol": 1e-4},
    )
    efficiency_speed = float(max_efficiency.x)
    range_speed = float(max_range.x)
    efficiency_value = metrics(efficiency_speed)[0]
    range_value = metrics(range_speed)[1]

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savetxt(
        OUTPUT_DIR / "hw4_trim_metrics.csv",
        np.column_stack((speeds, lift_to_drag, range_metric)),
        delimiter=",",
        header="V_m_per_s,L_over_D,V_L_over_D_m_per_s",
        comments="",
        fmt="%.8f",
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(speeds, lift_to_drag, "o-", label="Trim points")
    axes[0].plot(efficiency_speed, efficiency_value, "*", ms=13, label="Maximum L/D")
    axes[0].set_ylabel("L/D")
    axes[0].legend()

    axes[1].plot(speeds, range_metric, "o-", label="Trim points")
    axes[1].plot(range_speed, range_value, "*", ms=13, label="Maximum range metric")
    axes[1].axvline(84.0, color="0.35", linestyle="--", label="DC-3 cruise: 84 m/s")
    axes[1].set_xlabel("True airspeed, V [m/s]")
    axes[1].set_ylabel("V L/D [m/s]")
    axes[1].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle("HW4 trimmed aerodynamic efficiency and range metric")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hw4_range_metrics.png", dpi=200)

    print(f"Maximum L/D: {efficiency_value:.3f} at {efficiency_speed:.2f} m/s")
    print(f"Maximum V L/D: {range_value:.3f} m/s at {range_speed:.2f} m/s")
    print(f"Difference from DC-3 cruise: {range_speed - 84.0:+.2f} m/s")


if __name__ == "__main__":
    main()
