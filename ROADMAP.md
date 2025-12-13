# Pylinkage Roadmap - Suggested Features

This document outlines potential features that would significantly enhance pylinkage's capabilities for mechanical engineering applications.

## Priority 1: Velocity & Acceleration Analysis

**Status:** Not implemented
**Impact:** High - Fundamental kinematics capability

Currently pylinkage only computes positions. Adding velocity and acceleration analysis would enable:

- Angular velocity of all links given input crank speed
- Linear velocity of any point on any link
- Angular and linear accelerations
- Velocity and acceleration diagrams/vectors

### Proposed API

```python
# Set input crank angular velocity
linkage.set_input_velocity(crank, omega=10.0)  # rad/s

# After stepping, query velocities
joint.velocity        # (vx, vy) linear velocity
joint.angular_velocity  # omega for the link

# Accelerations
joint.acceleration         # (ax, ay) linear acceleration
joint.angular_acceleration  # alpha for the link

# Or batch query
linkage.get_velocities()      # All joint velocities
linkage.get_accelerations()   # All joint accelerations
```

### Implementation Approach

1. Use analytic differentiation of position equations
2. For Revolute joints: apply relative velocity/acceleration formulas
3. Chain rule through the solving order
4. Store results alongside position data in `Simulation`

---

## Priority 2: CAD Export

**Status:** Not implemented
**Impact:** Medium-High - Bridges simulation to fabrication

Export linkage geometry to standard CAD formats:

- **SVG** - Web/laser cutting
- **DXF** - AutoCAD/CNC
- **STEP** - 3D CAD interchange

### Proposed API

```python
# Export current configuration
linkage.export_svg("linkage.svg", scale=10, link_width=5)
linkage.export_dxf("linkage.dxf")

# Export with annotations
linkage.export_svg("linkage.svg",
    show_dimensions=True,
    show_joint_names=True,
    show_loci=True)

# Export animation frames
linkage.export_svg_sequence("frames/", iterations=24)
```

### Dependencies

- `svgwrite` for SVG export
- `ezdxf` for DXF export
- `cadquery` or similar for STEP (optional)

---

## Priority 3: Interactive Visualization

**Status:** Not implemented (only Matplotlib GIF)
**Impact:** Medium - Improved user experience

Modern interactive visualization:

- Real-time parameter adjustment with sliders
- Browser-based visualization
- Click-and-drag joint repositioning
- Zoom/pan controls

### Proposed API

```python
# Launch interactive viewer
linkage.show_interactive()

# Plotly-based (runs in browser)
from pylinkage.visualizer import PlotlyVisualizer
viz = PlotlyVisualizer(linkage)
viz.show()

# Jupyter widget
linkage.widget()  # Returns ipywidgets interactive
```

### Dependencies

- `plotly` for browser-based visualization
- `ipywidgets` for Jupyter integration (optional)

---

## Priority 4: Classical Synthesis Methods

**Status:** Not implemented
**Impact:** Medium - Advanced design capability

Implement classical mechanism synthesis theory:

- **Burmester theory** - Find 4-bar linkages passing through precision points
- **Function generation** - Match input/output angle relationships
- **Path generation** - Coupler curves through specified points
- **Motion generation** - Rigid body guidance

### Proposed API

```python
from pylinkage.synthesis import four_bar_synthesis

# Path generation: find 4-bar where coupler passes through points
precision_points = [(0, 1), (1, 2), (2, 1.5), (3, 0)]
solutions = four_bar_synthesis.path_generation(precision_points)

for linkage in solutions:
    linkage.show()

# Function generation: input angle -> output angle
angle_pairs = [(0, 0), (30, 45), (60, 80), (90, 90)]
solutions = four_bar_synthesis.function_generation(angle_pairs)
```

---

## Priority 5: Transmission Angle Analysis

**Status:** Not implemented
**Impact:** Medium - Design quality metric

Analyze mechanism quality via transmission angle:

- Compute transmission angle throughout motion cycle
- Flag configurations with poor transmission (< 40 deg or > 140 deg)
- Optimize for transmission angle quality

### Proposed API

```python
# Get transmission angle at current position
angle = linkage.transmission_angle(input_joint, output_joint)

# Get min/max over full cycle
analysis = linkage.analyze_transmission()
analysis.min_angle      # Worst case
analysis.max_angle
analysis.mean_angle
analysis.is_acceptable  # True if always in [40, 140] deg range

# Use in optimization
@kinematic_minimization
def fitness(loci, linkage):
    analysis = linkage.analyze_transmission()
    if analysis.min_angle < 40:
        raise UnbuildableError("Poor transmission angle")
    return some_metric(loci)
```

---

## Priority 6: Sensitivity & Tolerance Analysis

**Status:** Not implemented
**Impact:** Medium - Manufacturing considerations

Analyze how dimensional variations affect output:

- Sensitivity of output path to each dimension
- Monte Carlo tolerance analysis
- Identify critical dimensions

### Proposed API

```python
# Sensitivity analysis
sens = linkage.sensitivity_analysis(output_joint)
sens.to_dataframe()
# Returns: which dimensions most affect output position

# Tolerance stackup
tolerances = {
    'crank_radius': 0.1,  # +/- 0.1 mm
    'link_length': 0.2,
}
result = linkage.tolerance_analysis(tolerances, samples=1000)
result.output_variation  # Expected variation in output path
result.plot_cloud()      # Scatter plot of output positions
```

---

## Priority 7: Multi-Objective Optimization

**Status:** Not implemented (single objective only)
**Impact:** Low-Medium - Advanced optimization

Pareto-optimal solutions for competing objectives:

- Minimize path error AND maximize transmission angle
- Trade-off visualization
- NSGA-II or similar algorithm

### Proposed API

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

## Priority 8: Standard Linkage Description Formats

**Status:** Not implemented
**Impact:** Medium - Interoperability with other tools

Support industry-standard and academic formats for linkage definition import/export:

### Target Formats

- **Graph-based (JSON/YAML)** - Nodes for joints, edges for links. Simple and portable.
- **URDF** - ROS ecosystem standard, XML-based robot description
- **SDF** - Gazebo simulation description format
- **Kinematic graph adjacency matrix** - Pure numerical representation for analysis

### Proposed API

```python
# Export to various formats
linkage.to_urdf("linkage.urdf")
linkage.to_json("linkage.json")
linkage.to_graph()  # Returns networkx graph

# Import from formats
from pylinkage.io import from_urdf, from_json, from_graph

linkage = from_urdf("robot.urdf", base_link="base", end_link="tool")
linkage = from_json("mechanism.json")
linkage = from_graph(nx_graph, joint_types={...})

# Graph-based JSON schema
{
    "joints": [
        {"id": "A", "type": "static", "position": [0, 0]},
        {"id": "B", "type": "crank", "parent": "A", "radius": 30},
        {"id": "C", "type": "revolute", "parents": ["B", "D"], "distances": [80, 60]},
        {"id": "D", "type": "static", "position": [100, 0]}
    ],
    "driver": "B",
    "metadata": {"name": "four-bar", "units": "mm"}
}

# Adjacency matrix representation
adjacency = linkage.to_adjacency_matrix()
#        J0  J1  J2  J3
# Link0 [ 1,  1,  0,  0]  # ground-crank
# Link1 [ 0,  1,  1,  0]  # crank-coupler
# Link2 [ 0,  0,  1,  1]  # coupler-rocker
# Link3 [ 1,  0,  0,  1]  # ground-rocker
```

### Format Comparison

| Format | Closed Loops | Readability | Tool Support | Use Case |
|--------|--------------|-------------|--------------|----------|
| JSON/YAML | Yes | High | Universal | General interchange |
| URDF | Partial | High | ROS/Gazebo | Robotics integration |
| SDF | Partial | High | Gazebo | Simulation |
| Graph/Matrix | Yes | Low | NetworkX | Academic analysis |
| D-H Parameters | No | Low | Robotics libs | Serial chains only |

### Implementation Approach

1. Define a canonical JSON schema for planar linkages
2. Implement `to_json()` / `from_json()` as the primary interchange format
3. Add URDF export for ROS ecosystem compatibility
4. Provide `to_graph()` for integration with NetworkX for graph analysis
5. Document schema for third-party tool integration

### Dependencies

- `networkx` for graph representation (optional)
- Standard library `json` / `xml.etree` for serialization

---

## Implementation Order Recommendation

1. **Velocity & Acceleration** - Core kinematics, no new dependencies
2. **Transmission Angle** - Simple to implement, high value for mechanism quality
3. **CAD Export (SVG first)** - Practical value, minimal dependency
4. **Standard Description Formats (JSON first)** - Enables interoperability, minimal dependencies
5. **Interactive Visualization** - UX improvement
6. **Sensitivity Analysis** - Engineering rigor for manufacturing
7. **Synthesis Methods** - Advanced design capability
8. **Multi-Objective Optimization** - Niche use case
