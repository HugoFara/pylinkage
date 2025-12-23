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

**Web Application** (Streamlit):
- Location: `app/`
- Run with: `uv run streamlit run app/main.py`
- Features: Interactive joint manipulation, real-time simulation

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

## In Progress / Partially Implemented

### Velocity & Acceleration Analysis

**Status:** Partially implemented (low-level only)
**Impact:** High - Fundamental kinematics capability
**Location:** `src/pylinkage/solver/velocity.py`, `src/pylinkage/solver/acceleration.py`

Low-level numba-compiled solvers exist for per-joint velocity and acceleration computation. These are optimized for use in the fast simulation pipeline.

**What exists:**
- `solve_crank_velocity()`, `solve_rrr_velocity()`, etc. in `velocity.py`
- `solve_crank_acceleration()`, `solve_rrr_acceleration()`, etc. in `acceleration.py`
- Analytic differentiation of position equations

**What's missing (high-level API):**

```python
# Proposed high-level API (not yet implemented)
linkage.set_input_velocity(crank, omega=10.0)  # rad/s

# After stepping, query velocities
joint.velocity          # (vx, vy) linear velocity
joint.angular_velocity  # omega for the link

# Batch query
linkage.get_velocities()      # All joint velocities
linkage.get_accelerations()   # All joint accelerations
```

**Next steps:**
1. Integrate velocity/acceleration into `Linkage.step()` or add `Linkage.step_with_derivatives()`
2. Store results in `Simulation` dataclass
3. Add velocity/acceleration properties to joint classes

---

### CAD Export

**Status:** Partially implemented (SVG only)
**Impact:** Medium-High - Bridges simulation to fabrication

**What exists:**
- SVG export via `drawsvg_viz.py` with ISO 3952 kinematic symbols

```python
from pylinkage.visualizer import save_linkage_svg

save_linkage_svg(linkage, "output.svg", scale=80, show_loci=True)
```

**What's missing:**
- **DXF** - AutoCAD/CNC (would use `ezdxf`)
- **STEP** - 3D CAD interchange (would use `cadquery`)

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

## Not Yet Implemented

### Sensitivity & Tolerance Analysis

**Status:** Not implemented
**Impact:** Medium - Manufacturing considerations

Analyze how dimensional variations affect output:

- Sensitivity of output path to each dimension
- Monte Carlo tolerance analysis
- Identify critical dimensions

```python
# Proposed API
sens = linkage.sensitivity_analysis(output_joint)
sens.to_dataframe()
# Returns: which dimensions most affect output position

tolerances = {
    'crank_radius': 0.1,  # +/- 0.1 mm
    'link_length': 0.2,
}
result = linkage.tolerance_analysis(tolerances, samples=1000)
result.output_variation  # Expected variation in output path
result.plot_cloud()      # Scatter plot of output positions
```

---

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
| 1 | High-level Velocity API | Low-level done | Medium | Integrate into Linkage class |
| 2 | DXF Export | Not started | Low | Add `ezdxf` dependency |
| 3 | Triad/Tetrad Support | Not started | High | Extend solver/assur modules |
| 4 | Sensitivity Analysis | Not started | Medium | Monte Carlo wrapper |
| 5 | URDF/SDF Export | Not started | Medium | XML serialization |
| 6 | Multi-Objective Opt | Not started | Medium | NSGA-II integration |

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
