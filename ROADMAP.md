# Pylinkage Roadmap - Suggested Features

This document outlines potential features that would significantly enhance pylinkage's capabilities for mechanical engineering applications.

---

## Completed Features

### Classical Synthesis Methods

**Status:** Implemented
**Location:** `src/pylinkage/synthesis/`

Full implementation of classical mechanism synthesis theory:

- **Burmester theory** - `burmester.py`
- **Function generation** - `function_generation.py` (Freudenstein equation)
- **Path generation** - `path_generation.py` (coupler curves through specified points)
- **Motion generation** - `motion_generation.py` (rigid body guidance)
- **Utilities** - `utils.py` (Grashof criterion: `is_grashof()`, `is_crank_rocker()`)
- **Conversion** - `conversion.py` (`fourbar_from_lengths()`, `solution_to_linkage()`)

```python
from pylinkage.synthesis import (
    function_generation,
    path_generation,
    motion_generation,
    is_grashof,
    solution_to_linkage,
)

# Path generation example
precision_points = [(0, 1), (1, 2), (2, 1.5), (3, 0)]
result = path_generation(precision_points)
for solution in result.solutions:
    linkage = solution_to_linkage(solution)
```

---

### Interactive Visualization

**Status:** Implemented
**Location:** `src/pylinkage/visualizer/`

Multiple visualization backends available:

- **Matplotlib** - `static.py`, `animated.py` (GIF output via `show_linkage()`)
- **Plotly** - `plotly_viz.py` (interactive HTML via `plot_linkage_plotly()`)
- **drawsvg** - `drawsvg_viz.py` (publication-quality SVG via `save_linkage_svg()`)
- **PSO dashboards** - `pso_plots.py`

```python
from pylinkage.visualizer import plot_linkage_plotly, save_linkage_svg

# Plotly interactive
plot_linkage_plotly(linkage, loci)

# Publication SVG
save_linkage_svg(linkage, "output.svg")
```

---

### Standard Linkage Description Formats (JSON)

**Status:** Partially implemented
**Location:** `src/pylinkage/mechanism/serialization.py`

JSON serialization for the Mechanism model is complete:

```python
from pylinkage.mechanism import (
    mechanism_to_json,
    mechanism_from_json,
    Mechanism,
)

# Save to JSON
mechanism_to_json(mechanism, "linkage.json")

# Load from JSON
mechanism = mechanism_from_json("linkage.json")
```

**Not yet implemented:**
- URDF export (ROS ecosystem)
- SDF export (Gazebo)
- NetworkX graph export

---

### Velocity & Acceleration Analysis

**Status:** Implemented
**Location:** `src/pylinkage/solver/velocity.py`, `src/pylinkage/solver/acceleration.py`, `src/pylinkage/visualizer/kinematics.py`

Full velocity and acceleration analysis with both low-level numba solvers and high-level API:

**Low-level (numba-compiled):**
- `solve_crank_velocity()`, `solve_rrr_velocity()`, etc. in `velocity.py`
- `solve_crank_acceleration()`, `solve_rrr_acceleration()`, etc. in `acceleration.py`
- `step_single_velocity()`, `step_single_acceleration()` in `simulation.py`

**High-level API:**

```python
from pylinkage.simulation import Linkage

# Set input angular velocity on crank
linkage.set_input_velocity(crank, omega=10.0, alpha=0.0)  # rad/s, rad/s²

# Query velocities/accelerations on components
joint.velocity          # (vx, vy) linear velocity
joint.acceleration      # (ax, ay) linear acceleration

# Batch query
linkage.get_velocities()      # All joint velocities
linkage.get_accelerations()   # All joint accelerations

# Generator-based stepping with derivatives
for positions, velocities, accelerations in linkage.step_with_derivatives():
    ...

# Fast simulation returning all derivatives
positions, velocities, accelerations = linkage.step_fast_with_kinematics()
```

**Visualization:**

```python
from pylinkage.visualizer import show_kinematics, animate_kinematics

# Static frame with velocity vectors
fig = show_kinematics(linkage, frame_index=25, show_velocity=True)

# Animated with velocity vectors
fig = animate_kinematics(linkage, show_velocity=True, save_path="vel.gif")
```

---

## In Progress / Partially Implemented

### CAD Export

**Status:** Implemented (SVG, DXF, STEP)
**Impact:** Medium-High - Bridges simulation to fabrication
**Location:** `src/pylinkage/visualizer/`

Multi-format CAD export for fabrication workflows:

- **SVG** - Publication-quality diagrams via `drawsvg_viz.py`
- **DXF** - AutoCAD/CNC via `dxf_export.py` (uses `ezdxf`)
- **STEP** - 3D CAD interchange via `step_export.py` (uses `build123d`)

```python
from pylinkage.visualizer import (
    save_linkage_svg,
    save_linkage_dxf,
    save_linkage_step,
    LinkProfile,
)

# SVG (publication-quality)
save_linkage_svg(linkage, "output.svg", scale=80, show_loci=True)

# DXF (2D CAD/CNC)
save_linkage_dxf(linkage, "output.dxf")

# STEP (3D CAD)
save_linkage_step(linkage, "output.step")

# STEP with custom dimensions
profile = LinkProfile(width=10, thickness=3, fillet_radius=0.5)
save_linkage_step(linkage, "output.step", link_profile=profile)
```

**Installation:** DXF and STEP export require optional dependencies:
```bash
pip install pylinkage[cad]  # Installs ezdxf and build123d
```

---

### Transmission Angle Analysis

**Status:** Implemented
**Location:** `src/pylinkage/linkage/transmission.py`

Analyze mechanism quality via transmission angle:

- Compute transmission angle throughout motion cycle
- Flag configurations with poor transmission (< 40° or > 140°)
- Auto-detection of joints for standard four-bar linkages

```python
from pylinkage.linkage import analyze_transmission, TransmissionAngleAnalysis

# Current position
angle = linkage.transmission_angle()

# Full cycle analysis
analysis = linkage.analyze_transmission()
analysis.min_angle      # Worst case
analysis.max_angle
analysis.mean_angle
analysis.is_acceptable  # True if always in [40, 140]°
analysis.worst_angle()  # Angle with max deviation from 90°

# Custom acceptable range
analysis = linkage.analyze_transmission(acceptable_range=(30.0, 150.0))

# Use in optimization
@kinematic_minimization
def fitness(loci, linkage):
    analysis = linkage.analyze_transmission()
    if analysis.min_angle < 40:
        raise UnbuildableError("Poor transmission angle")
    return some_metric(loci)
```

---

### Stroke Analysis (Prismatic Joints)

**Status:** Implemented
**Location:** `src/pylinkage/linkage/transmission.py`

Analyze slider/piston travel for mechanisms with Prismatic joints:

- Compute slide position along the prismatic axis
- Track min/max/range of travel over a motion cycle
- Auto-detection of Prismatic joints

```python
from pylinkage.linkage import analyze_stroke, StrokeAnalysis

# Current slide position
pos = linkage.stroke_position()

# Full cycle analysis
analysis = linkage.analyze_stroke()
analysis.min_position    # Minimum slide position
analysis.max_position    # Maximum slide position
analysis.stroke_range    # Total travel (max - min)
analysis.amplitude       # Half the stroke range
analysis.center_position # Center of travel

# Use in optimization for slider-crank mechanisms
@kinematic_minimization
def fitness(loci, linkage):
    analysis = linkage.analyze_stroke()
    if analysis.stroke_range < required_stroke:
        raise UnbuildableError("Insufficient stroke")
    return some_metric(loci)
```

---

### Sensitivity & Tolerance Analysis

**Status:** Implemented
**Location:** `src/pylinkage/linkage/sensitivity.py`

Analyze how dimensional variations affect output:

- Sensitivity analysis: measures impact of each constraint on output path
- Monte Carlo tolerance analysis for manufacturing variation
- Human-readable constraint names (e.g., `crank_radius`, `coupler_dist1`)
- Optional transmission angle sensitivity tracking

```python
from pylinkage.linkage import (
    analyze_sensitivity,
    analyze_tolerance,
    SensitivityAnalysis,
    ToleranceAnalysis,
)

# Sensitivity analysis
analysis = linkage.sensitivity_analysis(delta=0.01)  # 1% perturbation
print(analysis.most_sensitive)         # Most sensitive constraint
print(analysis.sensitivity_ranking)    # Ranked by impact

# Export to pandas DataFrame (requires: pip install pylinkage[analysis])
df = analysis.to_dataframe()

# Tolerance analysis (Monte Carlo)
tolerances = {
    'crank_radius': 0.1,      # +/- 0.1 mm
    'coupler_dist1': 0.2,     # +/- 0.2 mm
}
result = linkage.tolerance_analysis(tolerances, n_samples=1000)
result.mean_deviation     # Mean path deviation from nominal
result.max_deviation      # Worst-case deviation
result.plot_cloud()       # Scatter plot of output positions
```

**Installation:** DataFrame export requires optional pandas:
```bash
pip install pylinkage[analysis]
```

---

## Not Yet Implemented

### Multi-Objective Optimization

**Status:** Not implemented (single objective only)
**Impact:** Low-Medium - Advanced optimization

Pareto-optimal solutions for competing objectives:

- Minimize path error AND maximize transmission angle
- Trade-off visualization
- NSGA-II or similar algorithm

```python
from pylinkage.optimization import multi_objective_optimization

objectives = [
    path_error_function,      # Minimize
    transmission_angle_min,   # Maximize
]

pareto_front = multi_objective_optimization(
    linkage,
    objectives,
    bounds=bounds,
    algorithm='nsga2'
)

pareto_front.plot()  # Pareto frontier visualization
pareto_front.solutions  # List of non-dominated solutions
```

---

### URDF/SDF Export

**Status:** Not implemented
**Impact:** Medium - ROS/Gazebo ecosystem interoperability

```python
# Proposed API
linkage.to_urdf("linkage.urdf")
mechanism.to_sdf("linkage.sdf")

# Import
from pylinkage.io import from_urdf
linkage = from_urdf("robot.urdf", base_link="base", end_link="tool")
```

---

### Higher-Order Assur Groups

**Status:** Not implemented
**Impact:** Medium - Enables more complex mechanisms
**Location:** `src/pylinkage/solver/solve.py`, `src/pylinkage/assur/signature.py`

Currently only Dyads (Class I, 2 links) are implemented. Higher-order groups raise `NotImplementedError`:

- **Triads** (Class II, 4 links)
- **Tetrads** (Class III, 6 links)

---

## Implementation Priority Recommendation

| Priority | Feature | Status | Effort | Next Action |
|----------|---------|--------|--------|-------------|
| ~~1~~ | ~~High-level Velocity API~~ | ✅ Done | — | — |
| ~~1~~ | ~~DXF/STEP Export~~ | ✅ Done | — | — |
| ~~2~~ | ~~Sensitivity Analysis~~ | ✅ Done | — | — |
| 1 | Triad/Tetrad Support | Not started | High | Extend solver/assur modules |
| 2 | URDF/SDF Export | Not started | Medium | XML serialization |
| 3 | Multi-Objective Opt | Not started | Medium | NSGA-II integration |

---

## Deprecated APIs (v1.0 Migration)

The `pylinkage.joints` module is deprecated and will be removed in v1.0.0. Migration path:

| Old (joints) | New Location |
|--------------|--------------|
| `Static` | `Ground` from `pylinkage.components` |
| `Crank` | `Crank` from `pylinkage.actuators` |
| `Revolute` | `RRRDyad` from `pylinkage.dyads` |
| `Prismatic`/`Linear` | `RRPDyad` from `pylinkage.dyads` |
| `Fixed` | `FixedDyad` from `pylinkage.dyads` |

All classes are also re-exported from `pylinkage.dyads` for backwards compatibility during the transition period.
