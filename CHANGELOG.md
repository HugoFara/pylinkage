# Changelog

All notable changes to pylinkage are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3] - 2023-06-23

### Added in 0.5.3

- We now checked compatibility with Python 3.10 and 3.11.
- ``pyproject.toml`` is now the official definition of the package.
- ``Linkage.hyperstaticity`` now clearly outputs a warning when used.

### Changed in 0.5.3

- ``master`` branch is now ``main``.
- ``docs/example/fourbar_linkage.py`` can now be used as a module (not the target but anyway).

### Fixed in 0.5.3

- Setting a motor with a negative rotation angle do no longer break ``get_rotation_period`` 
([#7](https://github.com/HugoFara/pylinkage/issues/7)).
- ``Pivot.reload`` and ``Linkage.__find_solving_order__`` were raising Warnings (stopping the code), 
when they should only print a message (intended behavior).
- Fixed many typos in documentation as well as in code.
- The ``TestPSO.test_convergence`` is now faster on average, and when it fails in the first time, it launches a bigger 
test.
- Minor lintings in the demo file ``docs/example/fourbar_linkage.py``.

### Deprecated in 0.5.3

- Using Python 3.7 is officially deprecated ([end of life by 2023-06-27](https://devguide.python.org/versions/)). 
It will no longer be tested, use it at your own risks!

## [0.5.2] - 2021-07-21

### Added in 0.5.2

- You can see the best score and the best dimensions updating in 
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

- ``exhaustive_optimization`` is now known as ``trials_and_errors_optimizattion``.
- Axes on linkage visualization are now named "x" and "y". It was "Points abcsices" and "Ordinates".
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

