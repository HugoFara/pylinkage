# Changelog
All notable changes to pylinkage are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- The ``Linear`` joint is here! His geometric model was of course implemented.
- ``line_from_points`` and ``circle_line_intersection``.

### Changed
- ``__secant_circles_intersections__`` renamed to
``secant_circles_intersections`` in ``geometry.py``.

### Fixed
- The highlighted locus was sometimes buggy in ``plot_static_linkage`` in 
``visualizer.py``.

### Deprecated
- The ``hyperstaticity`` method is renamed ``indeterminacy`` in ``Linkage`` 
(linkage.py)

## [0.5.2] - 2021-07-21
### Added
 - You can see the best score and best dimensions updating in 
   ``trials_and_errors_optimization``.

### Changed
 - The optimizers tests are 5 times quicker (~1 second now)  and raise less 
   false positive.
 - The sidebar in the documentation make navigation easier.
 - A bit of reorganization in optimizers, it should not affect users.

## [0.5.1] - 2021-07-14
### Added
 - The trials and errors optimization now have a progress bar (same kind of the
   one in particle swarm optimization), using 
   [tqdm](https://pypi.org/project/tqdm/).

### Changed
 - [matplotlib](https://matplotlib.org/) and tqdm now required. 

## [0.5.0] - 2021-07-12
End alpha development! The package is now robust enough to be used by a mere 
human. This version introduces a lot of changes and simplifications, so 
everything is not perfect yet, but it is complete enough to be considered a beta
version.

Git tags will no longer receive a "-alpha" mention.
### Added
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

### Changed
 - The ``particle_swarm_optimization`` ``eval_func`` signature is now similar to
the one ot ``trials_and_errors`` optimization. Wrappers are no longer needed!
 - The ``trials_and_errors_optimization`` function now asks for bounds instead 
   of dilatation and compression factors.
 - In ``trials_and_errors_optimization`` absolute step ``delta_dim`` is now 
   replaced by number of subdivisions ``divisions``.

### Fixed
 - After many hours of computations, default parameters in 
   ``particle_swarm_optimization`` are much more efficient. With the demo 
   ``fourbar_linkage``, the output wasn't even convergent sometimes. Now we 
   have a high convergence rate (~100%), and results equivalent to the 
   ``trials_and_errors_optimization`` (in the example). 
 - ``variator`` function of ``optimizer`` module was poorly working.
 - The docstrings were not displayed properly in documentation, this is fixed.

## [0.4.1] - 2021-07-11
### Added
 - The legend in ``visualizer.py`` is back!
 - Documentation published to GitHub pages! It is contained in the ``docs/`` 
   folder.
 - ``setup.cfg`` now include links to the website.

### Changed
 - Examples moved from ``pylinkage/examples/`` to ``docs/examples/``.
 - Tests moved from ``pylinkage/tests/`` to ``tests/``.

## [0.4.0] - 2021-07-06
### Added
 - The ``bounding_box`` method of geometry allow to compute the bounding box of 
   a finite set of 2D points. 
 - You can now customize colors of linkage's bars with the ``COLOR_SWITCHER`` 
   variable of ``visualizer.py``.
 - ``movement_bounding_box`` in ``visualizer.py`` to get the bounding box of 
   multiple loci.
 - ``parameters`` is optional in ``trials_and_errors_optimization`` (former 
   ``exhaustive_optimization``)
 - ``pylinkage/tests/test_optimizer.py`` for testing the optimizers, but it is a
   bit ugly as for now.
 - Flake8 validation in ``tox.ini``

### Fixed
 - ``set_num_constraints`` in ``Linkage`` was misbehaving due to update 0.3.0.
 - Cost history is no longer plotted automatically after a PSO.

### Changed
 - ``exhaustive_optimization`` is now known as ``trials_and_errors_optimizattion``.
 - Axis on linkage visualization are now named "x" and "y", and no longer 
   "Points abcsices" and "Ordinates".
 - A default view of the linkage is displayed in ``plot_static_linkage``.
 - Default padding in linkage representation was changed from an absolute value
   of 0.5 to a relative 20%.
 - Static view of linkage is now aligned with its kinematic one.
 - ``get_pos`` method of ``Linkage`` is now known as ``get_coords`` for 
   consistency.
 - Parameters renamed, reorganized and removed in ``particle_swarm_optimization``
   to align to PySwarms.
 - ``README.md`` updated consequently to the changes.

### Removed
 - Legacy built-in Particle Swarm Optimization, to avoid confusions.
 - We do no longer show a default legend on static representation.

## [0.3.0] - 2021-07-05
### Added
 - ``Joint`` objects now have a ``get_constraints`` method, consistent with 
   their ``set_constraints`` one.
 - ``Linkage`` now has a ``get_num_constraints`` method as syntactic sugar.
 - Code vulnerabilities checker
 - Walkthrough example has been expanded and now seems to be complete.

### Changed
 - ``Linkage``'s method ``set_num_constraints`` behaviour changed! You should
   now add ``flat=False`` to come back to the previous behaviour.
 - ``pylinkage/examples/fourbar_linkage.py`` expanded and finished.
 - The ``begin`` parameter of ``article_swarm_optimization`` is no longer
   mandatory. ``linkage.get_num_constraints()`` will be used if ``begin`` is not
   provided.
 - More flexible package version in ``environment.yml``
 - Output file name is now formatted as "Kinematic {linkage.name}" in
   ``plot_kinematic_linkage`` function of ``pylinkage/visualizer.py``
 - Python 3.6 is no longer tested in ``tox.ini``. Python 3.9 is now tested.

### Fixed
 - When linkage animation was saved, last frames were often missing in
   ``pylinkage/visualizer.py``, function ``plot_kinematic_linkage``.

## [0.2.2] - 2021-06-22
### Added
 - More continuous integrations workflows for multiple Python versions.

### Fixed
 - ``README.md`` could not be seen in PyPi.
 - Various types

## [0.2.1] - 2021-06-16
### Added
- ``swarm_tiled_repr`` function for  ``pylinkage/visualizer.py``, for 
  visualization of PySwarms.
- EXPERIMENTAL! ``hyperstaticity`` method ``Linkage``'s hyperstaticity 
  calculation.


### Changed
- ``pylinkage/exception.py`` now handles exceptions in another file.
- Documentation improvements.
- Python style improvements.
- ``.gitignore`` now modified from the standard GitHub gitignore example for 
  Python.

### Fixed
- ``circle`` method of ``Pivot`` in ``pylinkage/linkage.py``. was causing errors
- ``tox.ini`` now fixed.

## [0.2.0] - 2021-06-14
### Added
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

### Changed
 - ``.gitignore`` Python Package specific extensions added
 - ``MIT License`` → ``LICENSE``
 - ``lib/`` → ``pylinkage/``
 - ``tests/`` → ``pylinkage/tests/``
 - Revamped package organization.
 - Cleared ``setup.py``

## [0.0.1] - 2021-06-12
### Added
- ``lib/geometry.py`` as a mathematical for kinematic optimization
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

