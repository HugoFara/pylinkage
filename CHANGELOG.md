# Changelog

All notable changes to pylinkage are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`multi_objective_optimization` parallel evaluation.** New
  ``n_workers`` and ``linkage_factory`` keyword arguments route
  candidate evaluation through a
  ``concurrent.futures.ProcessPoolExecutor`` when ``n_workers > 1``.
  ``linkage_factory`` is an escape hatch for linkages that are not
  picklable (e.g. carry cached numba ``SolverData``): each worker
  builds its own linkage via the factory instead of receiving a
  pickled copy. This unblocks downstream packages (notably
  ``leggedsnake``) dropping their custom parallel NSGA problem
  wrappers.

- **`simulation.Linkage.to_hypergraph()`** (+ the module-level
  :func:`pylinkage.hypergraph.from_sim_linkage`). Converts a modern
  ``simulation.Linkage`` (the object produced by synthesis and
  ``co_optimize``) into a ``(HypergraphLinkage, Dimensions)`` pair so
  downstream hypergraph-native consumers (notably
  ``leggedsnake.Walker.from_synthesis``) can ingest synthesis output
  directly without a local shim. Covers ``Ground``, ``Crank``,
  ``ArcCrank``, ``LinearActuator``, ``RRRDyad``, ``FixedDyad``,
  ``RRPDyad`` and ``PPDyad``; unknown component types raise
  ``NotImplementedError`` instead of silently dropping topology.

- **`Dimensions.to_dict` / `Dimensions.from_dict`** (and matching
  ``DriverAngle.to_dict`` / ``DriverAngle.from_dict``). Returns a
  JSON-safe representation so downstream consumers (notably
  ``leggedsnake.serialization``) can drop their manual hyperedge-key
  stringification helpers. Hyperedge pairwise constraints are emitted
  as ``[node_a, node_b, distance]`` triples; ``from_dict`` also
  accepts the legacy ``"('a', 'b')"`` stringified-tuple form for
  back-compat with previously saved files.

### Fixed

- **`multi_objective_optimization` single-objective runs.** pymoo
  returns the single best solution with ``F`` shape ``(n_obj,)`` and
  ``X`` shape ``(n_var,)`` when ``n_obj == 1``, not ``(n_pop, n_obj)``.
  The previous code indexed ``result.F[:, k]`` unconditionally and
  crashed. Results are now normalised to 2-D before building the
  Ensemble so ``n_obj == 1`` works uniformly.

### Removed

- **`pylinkage.hypergraph._types`** re-export module (deprecated since
  0.8.0). Import the canonical types from ``pylinkage._types``
  (``JointType``, ``NodeRole``, ``NodeId``, …) directly. All internal
  callers have been migrated.

- **`pylinkage.solver.JOINT_LINEAR`** constant. This was always an
  alias for ``JOINT_PRISMATIC = 4``; the duplicate name has been
  removed. Use ``JOINT_PRISMATIC``.

- **``"Linear"`` entry in ``pylinkage.visualizer.SYMBOL_SPECS``** and
  the matching ``"Linear"`` branch in the auto-detect path inside
  ``pylinkage.linkage.transmission``. These matched the legacy
  ``joints.Linear`` class name, which is gone. Modern prismatic
  components are resolved through ``"Prismatic"`` / ``"RRPDyad"`` /
  ``"LinearActuator"``.

- **`get_num_constraints` / `set_num_constraints`** on both
  `simulation.Linkage` and `Mechanism`. These deprecated wrappers were
  added to ease the rename to ``get_constraints`` / ``set_constraints``
  and are now gone. Calling them raises ``AttributeError``.

### Changed

- **`pylinkage._compat`** now targets only the modern surface. The
  joint-legacy branches (``Static`` / ``_StaticBase`` / ``Revolute`` /
  ``Pivot`` / ``Fixed`` / ``Prismatic`` / ``Linear`` name matches) were
  dead after phase 2c; ``is_ground`` / ``is_dyad`` now only recognise
  the modern component classes and the Mechanism joint types.

### Added

- **Modern-container parity with the legacy `Linkage`.** Both
  `pylinkage.simulation.Linkage` and `pylinkage.mechanism.Mechanism`
  now carry the full kinematic and analysis surface the legacy class
  used to expose:
  - `compile()` + `step_fast()` — pre-compile the numba SolverData once
    and reuse it across calls (via `pylinkage.bridge`).
  - `step_fast_with_kinematics()` — batched numba simulation returning
    `(positions, velocities, accelerations)` trajectory arrays.
  - `set_input_velocity(driver_or_crank, omega, alpha=0.0)` +
    `get_velocities()` / `get_accelerations()` — per-joint
    velocity/acceleration accessors.
  - `step_with_derivatives()` — per-step Python path that computes
    velocities and accelerations via `solver.velocity` /
    `solver.acceleration`.
  - `analyze_transmission()` / `analyze_stroke()` /
    `analyze_sensitivity()` / `analyze_tolerance()` — bound-method
    shims over the free functions in `pylinkage.linkage`.
  - `transmission_angle()` / `stroke_position()` — single-shot
    accessors at the current pose.
  - `set_completely(constraints, positions)` — apply a flat constraint
    vector and joint positions in one call.
  - `simulation(iterations=None, dt=1.0)` — context manager that
    restores the initial joint positions on exit (new helper in
    `pylinkage._simulation_context.Simulation`).
  - `indeterminacy()` — planar Gruebler-Kutzbach mobility (a standard
    Grashof four-bar returns `1`).

- **`Mechanism` cross-API aliases.** `get_coords` / `set_coords` now
  work on a `Mechanism` as well, delegating to the native
  `get_joint_positions` / `set_joint_positions`.

- **`Mechanism.rebuild(initial_positions=None)`** — matches
  `simulation.Linkage.rebuild`. Optionally writes new joint positions
  and always clears the cached SolverData so the next `step_fast()`
  recompiles.

- **`Mechanism.step(iterations=None, dt=1.0)`** — accepts an
  `iterations` keyword (matching both legacy and modern Linkage).

- **`Joint.velocity` / `Joint.acceleration`** (runtime state,
  `compare=False`) on all `Mechanism` joints — populated by
  `step_with_derivatives` and `step_fast_with_kinematics`.

### Changed

- **`pylinkage._compat`**: `is_ground` now recognises a Mechanism
  `GroundJoint`; `is_driver` recognises a `RevoluteJoint` that sits as
  the output of a `DriverLink`/`ArcDriverLink`; `is_dyad` recognises a
  non-ground, non-driver `RevoluteJoint`/`PrismaticJoint`. The
  container-agnostic analysis helpers in `pylinkage.linkage.*` now
  auto-detect four-bar joints on a `Mechanism` without additional
  hints.
- **`pylinkage.bridge.solver_conversion.linkage_to_solver_data`** gains
  a `_mechanism_to_solver_data` dispatch path. `Mechanism`'s
  Links-on-constraints data model is now translated to `SolverData`
  for numba simulation (driver outputs become `JOINT_CRANK` with
  radius/angular-velocity pulled from the owning `DriverLink`; driven
  revolute joints become `JOINT_REVOLUTE` with anchor distances
  walked from the joint's `_links`).

### Fixed

- **Solver-cache invalidation on constraint mutation.**
  `simulation.Linkage.set_num_constraints` and
  `Mechanism.set_constraints` now clear `_solver_data` before applying
  the new constraints, so a subsequent `step_fast()` rebuilds the
  numba arrays. Without this, optimizers that round-tripped candidate
  constraints would silently keep simulating the previous parameters.

### Removed

- **`pylinkage.linkage.Linkage` and `pylinkage.linkage.Simulation`**:
  the legacy ``Linkage`` class is gone. ``pl.Linkage`` now points at
  :class:`pylinkage.simulation.Linkage` (component/actuator/dyad API);
  ``pl.Simulation`` points at the shared
  :class:`pylinkage._simulation_context.Simulation` context manager.
  Internal TYPE_CHECKING imports that referenced
  ``pylinkage.linkage.Linkage`` have been repointed to
  ``pylinkage.simulation.Linkage``. User code that built linkages via
  ``pl.Linkage(joints=[...])`` must migrate to the component API —
  see the migration notes in the 0.10.0 notebooks and tutorials.
- **`pylinkage.joints` module** (legacy joint API — `Static`, `Crank`,
  `Revolute`, `Pivot`, `Fixed`, `Prismatic`, `Joint`). Deprecated since
  0.7.0 (Pivot since 0.6.0). Use the component/actuator/dyad API:
  `pylinkage.components.Ground`, `pylinkage.actuators.Crank`,
  `pylinkage.dyads.RRRDyad` / `FixedDyad` / `RRPDyad`. Top-level
  re-exports (`pl.Static`, `pl.Crank`, …) are gone.
- **`pylinkage.linkage.Linkage.to_dict/from_dict/to_json/from_json`**:
  serialization of legacy joint-based linkages is no longer supported.
  Use `pylinkage.mechanism.mechanism_to_dict/from_dict` on a `Mechanism`
  instead.
- **`pylinkage.linkage.serialization`**: module removed — served legacy
  joints only.
- **`pylinkage.mechanism.mechanism_from_linkage` /
  `mechanism_to_linkage` / `convert_legacy_dict`**: bridged the legacy
  `Linkage` ↔ `Mechanism` models, neither of which needs the bridge now
  that the legacy joint API is gone. Use `pylinkage.mechanism.fourbar`
  and friends to build a `Mechanism` directly.
- **`pylinkage.hypergraph.from_linkage`** and
  **`pylinkage.assur.linkage_to_graph`**: same rationale as the legacy
  `to_linkage()` / `graph_to_linkage()` removed in 0.9.0. Use
  `from_mechanism` / `mechanism_to_graph` respectively.
- **`pylinkage.symbolic.linkage_to_symbolic` /
  `symbolic_to_linkage`**: removed. Build `SymbolicLinkage` directly
  with :class:`SymCrank` / :class:`SymRevolute` / :class:`SymStatic`, or
  use :func:`fourbar_symbolic`.
- **`pylinkage.optimization.grid_search.tqdm_verbosity()`**: overdue since
  0.7.0. Use `tqdm.tqdm(iterable, disable=not verbose)` directly.
- **`SynthesisResult.__len__` / `__iter__` / `__getitem__` / `__bool__`**:
  deprecated in 0.9.0. Access `result.solutions` (or `result.ensemble` for
  batch operations) instead — e.g. `len(result.solutions)`,
  `for linkage in result.solutions`, `result.solutions[i]`.
- **`pylinkage.hypergraph.to_linkage()`**: deprecated in 0.8.0. Use
  `pylinkage.hypergraph.to_mechanism()` for conversion to the current
  `Mechanism` model.
- **`pylinkage.assur.graph_to_linkage()`**: deprecated in 0.8.0. Use
  `pylinkage.assur.graph_to_mechanism()` for conversion to the current
  `Mechanism` model.

### Changed

- **`pylinkage.bridge.solver_conversion`**: collapsed to a single
  compatibility-agnostic implementation that dispatches through
  `pylinkage._compat` — no more legacy joint-type dispatch.
- **`pylinkage.synthesis.nbar_solution_to_linkage` /
  `_generic_nbar_to_linkage`**: now build a modern
  `pylinkage.simulation.Linkage` from the component/actuator/dyad API
  instead of a legacy joint-based `Linkage`.
- **`pylinkage.synthesis.linkage_to_synthesis_params`**: accepts the
  component API only; raises `ValueError` for legacy linkages.

## [0.9.0] - 2026-04-14

### Added

- **`extract_trajectory(loci, joint=-1)`** in `pylinkage.linkage.analysis`
  (re-exported from `pylinkage`): returns `(xs, ys)` numpy arrays for one
  joint's path, skipping unbuildable frames. Replaces the recurring
  `[(p[i][0], p[i][1]) for p in loci if p[i][0] is not None]` boilerplate.
  Accepts integer index, joint name, or joint instance (when `linkage` is
  given).

- **`extract_trajectories(loci, linkage=None)`**: all-joints variant of
  the above. Returns `{joint_name: (xs, ys)}` when `linkage` is given,
  or `{index: (xs, ys)}` otherwise. Skips `None` frames per joint.

- **`pylinkage.mechanism.fourbar(crank, coupler, rocker, ground, ...)`** and
  **`slider_crank(crank, rod, ...)`** factory functions: collapse the
  eight-line `MechanismBuilder` chain for the canonical four-bar and
  slider-crank topologies into a single call, returning an assembled
  `Mechanism`.

- **`TransmissionAngleAnalysis.plot(ax=None)`**: one-line replacement for
  the matplotlib boilerplate (`axhline` at the acceptable-range bounds,
  the 90° optimum, fixed `[0, 180]` y-axis, crank-angle x-axis). Accepts
  an existing axes for use inside subplot grids.

- **Population abstractions for batch mechanism work** (`pylinkage.population`):
  - `Member`: universal single-mechanism record (dimensions, scores, trajectory).
    `to_loci()` converts trajectories to the tuple format the visualizer expects.
  - `Ensemble`: topology-bound population — one linkage structure with N parameter
    variants. Batch simulation via the numba solver, numpy-style indexing
    (`ens[i]` → Member, `ens[1:3]` → Ensemble), columnar scores for vectorized
    `rank()`, `top()`, `filter()`, `filter_by_score()`. Visualization shortcuts:
    `show()`, `plot_plotly()`, `save_svg()`.
  - `Population`: heterogeneous collection of Ensembles, keyed by topology label.
    `simulate_all()`, `rank()`, `top()` across topologies.
    `from_members()` auto-groups by topology key.
    `from_topology_solutions()` wraps multi-topology synthesis results with
    `QualityMetrics` as score columns.
  - `SynthesisResult.ensemble` property: lazily builds an Ensemble from synthesis
    solutions with link lengths as score columns.

- **`skip_unbuildable` mode for `Linkage.step()`:** new boolean parameter that
  catches `UnbuildableError` per iteration and yields `None`-coordinate tuples
  instead of aborting the entire simulation. Non-Grashof and double-rocker
  linkages now recover the valid trajectory on both sides of dead zones.

- **Dual Annealing optimizer:** `dual_annealing_optimization()` wraps scipy's
  generalized simulated annealing — a single-trajectory global optimizer effective
  for problems with many local minima and expensive evaluations.
- **Optimizer chaining:** `chain_optimizers()` runs multiple optimizers in sequence,
  automatically feeding each result as the starting point for the next stage.
  Common pattern: global search (DE/PSO) → local refinement (Nelder-Mead).

- **Co-optimization of topology + dimensions:**
  - Mixed-variable evolutionary optimizer (`co_optimize()`) jointly searching
    discrete topology space and continuous dimensional space using NSGA-II/III
    via pymoo with custom genetic operators.
  - Topology neighborhood graph (`build_neighborhood_graph()`,
    `topology_neighbors()`, `topology_distance()`) defining adjacency between
    all 19 catalog topologies via add_dyad, remove_dyad, swap_variant, and
    restructure operations.
  - Custom pymoo operators: `MixedCrossover` (BLX-alpha blend + topology swap),
    `MixedMutation` (Gaussian perturbation + topology neighbor mutation),
    `warm_start_sampling()` (seed population from Phase 3 synthesis results).
  - Virtual edge encoding: expands hyperedges (ternary links) into pairwise
    distances and adds implicit ground-link virtual edges for chromosome
    representation.
  - Simultaneous triad placement via `scipy.optimize.least_squares` for
    topologies with circular dependencies (e.g., Stephenson six-bar).
  - Warm-start pipeline: `warm_start_co_optimization()` converts Phase 3
    `TopologySolution` results to `MixedChromosome` seeds for NSGA-II.
  - New types: `MixedChromosome`, `CoOptimizationConfig`, `CoOptSolution`,
    `CoOptimizationResult`.
  - `TopologyCatalog.topology_index()` and `topology_by_index()` for
    integer-indexed topology lookup.

- **Triad solving in mechanism simulation:** `Mechanism.step()` now uses Assur
  group decomposition internally, solving dyads and triads via `solve_group()`
  dispatch. Six-bar linkages (Watt and Stephenson types) can be simulated
  end-to-end. `graph_to_mechanism()` handles triad groups (2 internal nodes,
  4+ edges), creating the appropriate joints and links.

- **Convenience builders for six-bar linkages:**
  - `watt_from_lengths()`: build a Watt six-bar from seven link lengths +
    ground length. Returns a `SimLinkage` ready for simulation.
  - `stephenson_from_lengths()`: build a Stephenson six-bar from the same
    parameter pattern. Both include ASCII kinematic chain diagrams in docstrings.
  - Exported from `pylinkage.synthesis` alongside `fourbar_from_lengths()`.

- **Topology enumeration:**
  - Graph isomorphism detection via WL-1 color refinement + backtracking
    verification: `canonical_form()`, `canonical_hash()`, `are_isomorphic()`.
  - Systematic enumeration of all non-isomorphic 1-DOF planar linkage
    topologies up to 8 links: `enumerate_topologies()`, `enumerate_all()`.
    Validated against Mruthyunjaya 1984: 1 four-bar + 2 six-bars + 16 eight-bars = 19.
  - Built-in topology catalog (`TopologyCatalog`, `CatalogEntry`, `load_catalog()`)
    with JSON-serialized `HypergraphLinkage` graphs and metadata (link assortment,
    family, joint count).
  - All new symbols exported from `pylinkage.topology`.

### Fixed

- **`compute_dof` hyperedge counting:** `compute_mobility()` counted each hyperedge
  with k nodes as (k−1) links instead of 1 rigid body, giving wrong DOF for any
  mechanism with ternary or higher links (all six-bars and eight-bars).

### Deprecated

- **`SynthesisResult` collection protocol:** `len(result)`, `result[i]`,
  `for linkage in result`, and `bool(result)` now emit `DeprecationWarning`.
  Use `result.ensemble` instead. Will be removed in 1.0.0.
### Changed

- **`fourbar_from_lengths()` now returns `SimLinkage`** (from `pylinkage.simulation`)
  instead of the legacy `Linkage` (from `pylinkage.linkage`). The new object uses
  the component/actuator/dyad API: access joints via `.components` instead of
  `.joints`. The `.step()` method is unchanged. `linkage_to_synthesis_params()`
  accepts both old and new linkage types.

- **All optimization functions now return `Ensemble`** instead of `list[Agent]`,
  `list[MutableAgent]`, or `ParetoFront`. Affected functions:
  `particle_swarm_optimization()`, `trials_and_errors_optimization()`,
  `differential_evolution_optimization()`, `dual_annealing_optimization()`,
  `minimize_linkage()`, `chain_optimizers()`, `multi_objective_optimization()`,
  and all async variants. Migration: replace `score, dims, pos = result[0]`
  (Agent tuple unpacking) with `member = result[0]; member.score`.

- **Default simulation resolution increased from ~63 to 360 steps per rotation:**
  The default angular velocity for `Crank`, `ArcCrank`, `DriverLink`, and
  `ArcDriverLink` changed from `0.1` rad/step to `tau / 360` (~0.01745 rad/step),
  giving one sample per degree. The `Mechanism.get_rotation_period()` fallback
  and all synthesis/visualizer iteration defaults changed from `100` to `360`
  accordingly. A `DEFAULT_ANGULAR_VELOCITY` constant is now exported from
  `pylinkage.actuators`.

- **PSO is now pure NumPy:** `particle_swarm_optimization()` no longer depends on
  pyswarms (unmaintained since 2021). Replaced with a built-in local-best
  ring-topology PSO. The `pso` optional extra is kept but empty for backwards
  compatibility. The API is unchanged.

- **Renamed abbreviated parameters for clarity
  ([#17](https://github.com/HugoFara/pylinkage/issues/17)):**
  All old names are still accepted as keyword arguments for backwards
  compatibility.
  - `iters` → `iterations` in `particle_swarm_optimization()` and its async
    variant.
  - `pos` → `initial_positions` in `Linkage.rebuild()`.
  - `init_positions` → `initial_positions` in `Agent`, `MutableAgent`, and
    `ParetoSolution`.

### Removed

- **`HypostaticError`** alias removed. Use `UnderconstrainedError` directly
  (alias was deprecated since 0.7.0).
- **`Linear`** joint alias removed. Use `Prismatic` directly
  (alias was deprecated since 0.7.0).
- **pyswarms dependency** removed from all extras (`pso`, `full`, dev group).

## [0.8.0] - 2026-03-28

### Added

- **Multi-objective optimization:**
  - New `multi_objective_optimization()` function using NSGA-II/NSGA-III algorithms via pymoo.
  - `ParetoFront` class for storing and analyzing non-dominated solutions.
  - `ParetoSolution` dataclass for individual Pareto-optimal solutions.
  - Pareto front visualization with `pareto.plot()`.
  - Hypervolume indicator computation with `pareto.hypervolume()`.
  - Best compromise solution selection with `pareto.best_compromise()`.
  - Crowding distance-based filtering with `pareto.filter()`.
  - New optional dependency group: `pip install pylinkage[moo]`.
- **Cam-follower mechanisms:**
  - New `pylinkage.cam` module with motion laws and profile definitions.
  - Motion laws: `HarmonicMotionLaw`, `CycloidalMotionLaw`, `ModifiedTrapezoidalMotionLaw`,
    `PolynomialMotionLaw` (with `polynomial_345()` and `polynomial_4567()` factory functions).
  - Profile types: `FunctionProfile` (motion law-based) and `PointArrayProfile` (spline interpolation).
  - `TranslatingCamFollower` dyad for linear follower motion driven by cam rotation.
  - `OscillatingCamFollower` dyad for rocker arm motion driven by cam rotation.
  - Both knife-edge (roller_radius=0) and roller followers supported.
  - Numba-compiled profile evaluation for high-performance simulation.
- **Triad (Class II) Assur groups:**
  - New `Dyad` and `Triad` classes parameterized by signature string, replacing
    the per-type classes (`DyadRRR`, `DyadRRP`, etc.) which are kept as aliases.
  - `signature_to_hypergraph()` now generates triad topologies (6-joint signatures).
  - `decompose_assur_groups()` detects triads when no dyad can be formed,
    enabling decomposition of six-bar mechanisms (Watt and Stephenson types).
  - Solver dispatches by `solver_category` (circle-circle, circle-line, line-line)
    instead of `isinstance()`, automatically supporting new group signatures.
- **Topology analysis:**
  - New `pylinkage.topology` module with `compute_dof()` implementing Grübler's
    formula (`DOF = 3(n−1) − 2j₁ − j₂`) on `HypergraphLinkage`.
  - `compute_mobility()` returns full `MobilityInfo` (DOF, link count, joint counts).
- SymPy for analytical optimization.
- Native computation of velocity and acceleration with visualizations.
- Linkage synthesis with Burgmester's theory, function, path and motion generation.
- Adds scipy.
  - Exact optimization solving (better than numpy) + support constraints.
  - Adds a new optimization: differential evolution.
- **High-level velocity/acceleration API:**
  - `Component.velocity` and `Component.acceleration` properties on all components.
  - `simulation.Linkage.set_input_velocity(actuator, omega, alpha)` to set crank angular velocity.
  - `simulation.Linkage.step_with_derivatives()` generator yielding (positions, velocities, accelerations).
  - `simulation.Linkage.get_velocities()` and `get_accelerations()` batch query methods.
  - `solver.step_single_acceleration()` numba-compiled acceleration solver.
  - Exported acceleration solvers: `solve_crank_acceleration`, `solve_revolute_acceleration`,
    `solve_fixed_acceleration`, `solve_prismatic_acceleration`.

### Fixed

- **PSO score sign:** `particle_swarm_optimization()` returned the negated pyswarms
  cost when `order_relation=max`, producing incorrect (often negative) scores.
- **Mechanism builder branch selection:** `MechanismBuilder.set_branch()` produced
  inconsistent assembly configurations because circle-circle constraints arrived in
  non-deterministic order depending on which connected port was solved first.
  Constraints are now sorted by center position before intersection, making branch
  0/1 deterministic.

### Changed

- **Breaking:** Dropped Python 3.9 support. Minimum version is now Python 3.10.
- Added Python 3.14 to CI test matrix.
- **Breaking:** `Linkage.step_fast_with_kinematics()` now returns a 3-tuple
  `(positions, velocities, accelerations)` instead of 2-tuple.
- **Breaking:** `LinearActuator.velocity` attribute renamed to `LinearActuator.speed`
  to avoid conflict with the new `Component.velocity` property.
- `simulate_with_kinematics()` now computes accelerations in addition to velocities.

## [0.7.0] - 2025-12-13

### Added in 0.7.0

- Serialization: adds linkage serialization features.
- Typing: adds typing.
- Test: adds complete testing coverage.
- Adds support for Python 3.14.
- Hypergraph as the base theory for linkages.

### Changed in 0.7.0

- Switches to `uv`.
- Renames `HypostaticError` to `UnderconstrainedError` and `hyperstaticity()` to `indeterminacy()`.
  Old names kept as deprecated aliases.
- Separate linkage definition from actual solving:
  - The internal solver is now numba + NumPy, almost 100x faster!
  - The user-facing code is now based on [Assur groups](https://en.wikipedia.org/wiki/Assur_group), that is more formal.

### Fixed in 0.7.0

- `__find_solving_order__()` is now properly tested and implemented ([#16](https://github.com/HugoFara/pylinkage/issues/16)).

### Deprecated in 0.7.0

- `Linear` joint term is now deprecated in favor of `Prismatic`.

### Removed in 0.7.0

- Removed support for Python 3.9.

## [0.6.0] - 2024-10-02

### Added in 0.6.0

- New joint: the ``Linear`` joint!
- New sub-package: optimization.collections.
  ``optimization.collections.Agent`` and
  ``optimization.collections.MutableAgent`` are two new classes that should
  standardize the format of optimization, related to
  ([#5](https://github.com/HugoFara/pylinkage/issues/5)).
  - ``Agent`` is immutable and inherits from a namedtuple. It is recommended
    to use it, as it is a bit faster.
  - ``MutableAgent`` is mutable. It may be deprecated/removed if ``Agent`` is satisfactory.
- New sub-package: geometry.
  - It introduces two new functions ``line_from_points`` and ``circle_line_intersection``.
- New examples:
  - ``examples/strider.py`` from [leggedsnake](https://github.com/HugoFara/leggedsnake),
  based on the [Strider Linkage](https://www.diywalkers.com/strider-linkage-plans.html).
  - ``examples/inverted_stroke_engine.py`` is a demo of a
    [four-stroke engine](https://en.wikipedia.org/wiki/Four-stroke_engine)
    featuring a Linear joint.
- ``Linkage.set_completely`` is a new method combining both
  ``Linkage.set_num_constraints`` and ``Linkage.set_coords``.
- New exception ``NotCompletelyDefinedError``, when a joint is reloading but
  its anchor coordinates are set to None.
- Some run configuration files added *for users of PyCharm*:
  - Run all tests with "All Tests".
  - Regenerate documentation with "Sphinx Documentation".

### Changed in 0.6.0

- Optimization return type changed ([#5](https://github.com/HugoFara/pylinkage/issues/5)):
  - ``trials_and_error_optimization`` return an array of ``MutableAgent``.
  - ``particle_swarm_optimization`` return an array of one ``Agent``.
  - It should not be a breaking change for most users.
- Changes to the "history" style.
  - It is no longer a global variable in example scripts.
  - It was in format iterations[dimensions, score], now it is a standard
    iterations[score, dimensions, initial pos].
  - ``repr_polar_swarm`` (in example scripts) changed to follow the new format.
  - ``swarm_tiled_repr`` takes (index, swarm) as input argument. swarm is
    (score, dim, pos) for each agent for this iteration.
- ``repr_polar_swarm`` reload frame only when a new buildable linkage is generated.
  - This makes the display much faster.
  - For each iteration, you may see linkages that do not exist anymore.
- Folders reorganization:
  - The ``geometry`` module is now a package (``pylinkage/geometry``)
  - New package ``pylinkage/linkage``:
    - ``pylinkage/linkage.py`` separated and inserted in this package.
  - New package: ``pylinkage/joints``
    - Joints definition are in respective files.
  - New package ``pylinkage/optimization/``
    - ``pylinkage/optimizer.py`` split and inserted in.
    - Trials-and-errors related functions goes to ``grid_search.py``.
    - Particle swarm optimization is at ``particle_swarm.py``.
    - New file ``utils.py`` for ``generate_bounds``.
  - Tests follow the same renaming.
  - From the user perspective, no change (execution *may* be a bit faster)
  - ``source/`` renamed to ``sphinx/`` because it was confusing and only for
    Sphinx configuration.
- Transition from Numpydoc to reST for docstrings ([#12](https://github.com/HugoFara/pylinkage/issues/12)).
- ``__secant_circles_intersections__`` renamed to
``secant_circles_intersections`` (in ``pylinkage/geometry/secants.py``).

### Fixed in 0.6.0

- ``swarm_tiled_repr`` in ``visualizer.py`` was wrongly assigning dimensions.
- Setting ``locus_highlight`` in ``plot_static_linkage`` would result in an error.
- ``Pivot.reload`` was returning arbitrary point when we had an infinity of solutions.
- The highlighted locus was sometimes buggy in ``plot_static_linkage`` in
``visualizer.py``.

### Deprecated in 0.6.0

- Using ``tqdm_verbosity`` is deprecated in favor of using ``disable=True`` in
  a tqdm object.
- The ``Pivot`` class is deprecated in favor of the ``Revolute`` class.
The name "Pivot joint" is not standard.
Related to [#13](https://github.com/HugoFara/pylinkage/issues/13).
- The ``hyperstaticity`` method is renamed ``indeterminacy`` in ``Linkage``
(linkage.py)

### Removed in 0.6.0

- Drops support for Python 3.7 and 3.8 as both versions reached end-of-life.
- ``movement_bounding_bow`` is replaced by ``movement_bounding_box`` (typo in
  function name).

## [0.5.3] - 2023-06-23

### Added in 0.5.3

- We now checked compatibility with Python 3.10 and 3.11.
- ``pyproject.toml`` is now the official definition of the package.
- ``Linkage.hyperstaticity`` now clearly outputs a warning when used.

### Changed in 0.5.3

- ``master`` branch is now ``main``.
- ``docs/example/fourbar_linkage.py`` can now be used as a module (not the
  target but anyway).
- ``docs/examples`` moved to ``examples/`` (main folder).
  - Now ``docs/`` only contains sphinx documentation.
- ``docs/examples/images`` moved to ``images/``.

### Fixed in 0.5.3

- Setting a motor with a negative rotation angle do no longer break ``get_rotation_period``
([#7](https://github.com/HugoFara/pylinkage/issues/7)).
- ``Pivot.reload`` and ``Linkage.__find_solving_order__`` were raising
  Warnings (stopping the code), when they should only print a message
  (intended behavior).
- Fixed many typos in documentation as well as in code.
- The ``TestPSO.test_convergence`` is now faster on average, and when it
  fails in the first time, it launches a bigger test.
- Minor linting in the demo file ``docs/example/fourbar_linkage.py``.

### Deprecated in 0.5.3

- Using Python 3.7 is officially deprecated ([end of life by 2023-06-27](https://devguide.python.org/versions/)).
It will no longer be tested, use it at your own risks!

## [0.5.2] - 2021-07-21

### Added in 0.5.2

- You can see the best score and best dimensions updating in
   ``trials_and_errors_optimization``.

### Changed in 0.5.2

- The optimizer tests are 5 times quicker (~1 second now) and raise less
  false positive.
- The sidebar in the documentation makes navigation easier.
- A bit of reorganization in optimizers, it should not affect users.

## [0.5.1] - 2021-07-14

### Added in 0.5.1

- The trial and errors optimization now have a progress bar (same kind of the
  one in particle swarm optimization), using
  [tqdm](https://pypi.org/project/tqdm/).

### Changed in 0.5.1

- [matplotlib](https://matplotlib.org/) and tqdm now required.

## [0.5.0] - 2021-07-12

End of alpha development!
The package is now robust enough to be used by a mere human.
This version introduces a lot of changes and simplifications,
so everything is not perfect yet,
but it is complete enough to be considered a beta version.

Git tags will no longer receive an "-alpha" mention.

### Added in 0.5.0

- It is now possible and advised to import useful functions from
  pylinkage.{object}, without full path. For instance, use
  ``from pylinkage import Linkage`` instead of
  ``from pylinkage.linkage import Linkage``.
- Each module had his header improved.
- The ``generate_bounds`` functions is a simple way to generate bounds before
  optimization.
- The ``order_relation`` arguments of ``particle_swarm_optimization`` and
  ``trials_and_errors_optimization`` let you choose between maximization and
  minimization problem.
- You can specify a custom order relation with ``trials_and_errors_optimization``.
- The ``verbose`` argument in optimizers can disable verbosity.
- ``Static`` joints can now be defined implicitly.
- The ``utility`` module provides two useful decorators
  ``kinematic_minimization`` and ``kinematic_optimizatino``. They greatly
  simplify the workflow of defining fitness functions.
- Versioning is now done thanks to bump2version.

### Changed in 0.5.0

- The ``particle_swarm_optimization`` ``eval_func`` signature is now similar to
the one ot ``trials_and_errors`` optimization. Wrappers are no longer needed!
- The ``trials_and_errors_optimization`` function now asks for bounds instead
  of dilatation and compression factors.
- In ``trials_and_errors_optimization`` absolute step ``delta_dim`` is now
  replaced by number of subdivisions ``divisions``.

### Fixed in 0.5.0

- After many hours of computations, default parameters in
  ``particle_swarm_optimization`` are much more efficient. With the demo
  ``fourbar_linkage``, the output wasn't even convergent sometimes. Now we
  have a high convergence rate (~100%), and results equivalent to the
  ``trials_and_errors_optimization`` (in the example).
- ``variator`` function of ``optimizer`` module was poorly working.
- The docstrings were not displayed properly in documentation, this is fixed.

## [0.4.1] - 2021-07-11

### Added in 0.4.1

- The legend in ``visualizer.py`` is back!
- Documentation published to GitHub pages! It is contained in the ``docs/``
  folder.
- ``setup.cfg`` now include links to the website.

### Changed in 0.4.1

- Examples moved from ``pylinkage/examples/`` to ``docs/examples/``.
- Tests moved from ``pylinkage/tests/`` to ``tests/``.

## [0.4.0] - 2021-07-06

### Added in 0.4.0

- The ``bounding_box`` method of geometry allows computing the bounding box of
  a 2D points finite set.
- You can now customize colors of linkage's bars with the ``COLOR_SWITCHER``
  variable of ``visualizer.py``.
- ``movement_bounding_box`` in ``visualizer.py`` to get the bounding box of
  multiple loci.
- ``parameters`` is optional in ``trials_and_errors_optimization`` (former
  ``exhaustive_optimization``)
- ``pylinkage/tests/test_optimizer.py`` for testing the optimizers, but it is a
  bit ugly as for now.
- Flake8 validation in ``tox.ini``

### Fixed in 0.4.0

- ``set_num_constraints`` in ``Linkage`` was misbehaving due to update 0.3.0.
- Cost history is no longer plotted automatically after a PSO.

### Changed in 0.4.0

- ``exhaustive_optimization`` is now known as
  ``trials_and_errors_optimizattion``.
- Axes on linkage visualization are now named "x" and "y". It was "Points
  abcsices" and "Ordinates".
- A default view of the linkage is displayed in ``plot_static_linkage``.
- Default padding in linkage representation was changed from an absolute value
  of 0.5 to a relative 20%.
- Static view of linkage is now aligned with its kinematic one.
- ``get_pos`` method of ``Linkage`` is now known as ``get_coords`` for
  consistency.
- Parameters renamed, reorganized and removed in ``particle_swarm_optimization``
  to align to PySwarms.
- ``README.md`` updated consequently to the changes.

### Removed in 0.4.0

- Legacy built-in Particle Swarm Optimization, to avoid confusion.
- We do no longer show a default legend on static representation.

## [0.3.0] - 2021-07-05

### Added in 0.3.0

- ``Joint`` objects now have a ``get_constraints`` method, consistent with
  their ``set_constraints`` one.
- ``Linkage`` now has a ``get_num_constraints`` method as syntactic sugar.
- Code vulnerabilities checker
- Walkthrough's example has been expanded and now seems to be complete.

### Changed in 0.3.0

- ``Linkage``'s method ``set_num_constraints`` behaviour changed! You should
  now add ``flat=False`` to come back to the previous behavior.
- ``pylinkage/examples/fourbar_linkage.py`` expanded and finished.
- The ``begin`` parameter of ``article_swarm_optimization`` is no longer
  mandatory. ``linkage.get_num_constraints()`` will be used if ``begin`` is not
  provided.
- More flexible package version in ``environment.yml``
- Output file name now is formatted as "Kinematic {linkage.name}" in
  ``plot_kinematic_linkage`` function of ``pylinkage/visualizer.py``
- Python 3.6 is no longer tested in ``tox.ini``. Python 3.9 is now tested.

### Fixed in 0.3.0

- When linkage animation was saved, last frames were often missing in
  ``pylinkage/visualizer.py``, function ``plot_kinematic_linkage``.

## [0.2.2] - 2021-06-22

### Added in 0.2.2

- More continuous integration workflows for multiple Python versions.

### Fixed in 0.2.2

- ``README.md`` could not be seen in PyPi.
- Various types

## [0.2.1] - 2021-06-16

### Added in 0.2.1

- ``swarm_tiled_repr`` function for  ``pylinkage/visualizer.py``, for
visualization of PySwarms.
- EXPERIMENTAL! ``hyperstaticity`` method ``Linkage``'s hyperstaticity (over constrained)
calculation.

### Changed in 0.2.1

- ``pylinkage/exception.py`` now handles exceptions in another file.
- Documentation improvements.
- Python style improvements.
- ``.gitignore`` now modified from the standard GitHub gitignore example for
  Python.

### Fixed in 0.2.1

- ``circle`` method of ``Pivot`` in ``pylinkage/linkage.py``. It was causing errors
- ``tox.ini`` now fixed.

## [0.2.0] - 2021-06-14

### Added in 0.2.0

- ``pylinkage/vizualizer.py`` view your linkages using matplotlib!
- Issue templates in ``.github/ISSUE_TEMPLATE/``
- ``.github/workflows/python-package-conda.yml``: conda tests with unittest
  workflow.
- ``CODE_OF_CONDUCT.md``
- ``MANIFEST.in``
- ``README.md``
- ``environment.yml``
- ``setup.cfg`` now replaces ``setup.py``
- ``tox.ini``
- ``CHANGELOG.md``

### Changed in 0.2.0

- ``.gitignore`` Python Package specific extensions added
- ``MIT License`` → ``LICENSE``
- ``lib/`` → ``pylinkage/``
- ``tests/`` → ``pylinkage/tests/``
- Revamped package organization.
- Cleared ``setup.py``

## [0.0.1] - 2021-06-12

### Added in 0.0.1

- ``lib/geometry.py`` as a mathematical basis for kinematic optimization
- ``lib/linkage.py``, linkage builder
- ``lib/optimizer.py``, with Particle Swarm Optimization (built-in and PySwarms),
  and exhaustive optimization.
- ``MIT License``.
- ``requirements.txt``.
- ``setup.py``.
- ``tests/__init__.py``.
- ``tests/test_geometry.py``.
- ``tests/test_linkage.py``.
- ``.gitignore``.
