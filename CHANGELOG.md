# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2021-07-05
### Added
 - ``Joint`` objects now have a ``get_constraints`` method, consistent with their ``set_constraints`` one.
 - ``Linkage`` now has a ``get_num_constraints`` method.
 - Code vulnerabilities checker
 - Walktrough example has been expanded and now seems to be complete.

### Changed
 - ``Linkage``'s method ``set_num_constraints`` behaviour changed! You should now add ``flat=False`` to come back to the previous behaviour.
 - ``pylinkage/examples/fourbar_linkage.py`` expanded and finished.
 - The ``begin`` parameter of ``article_swarm_optimization`` is now longer mandatory. ``linkage.get_num_constraints()`` will be used if ``begin`` is not provided.
 - More flexible package version in ``environment.yml``
 - Output file name now is now formatted as "Kinematic {linkage.name}" in ``plot_kinematic_linkage`` function of ``pylinkage/visualizer.py``
 - Python 3.6 is no longer tested in ``tox.ini``. Python 3.9 is now tested.

### Fixed
 - When linkage animation was saved, last frames were often missing in``pylinkage/visualizer.py``, function ``plot_kinematic_linkage``

## [0.2.1] - 2021-06-16
### Added
- ``swarm_tiled_repr`` function for  ``pylinkage/visualizer.py``, for vizualisation of PySwarms.
- EXPERIMENTAL! ``hyperstaticity`` method ``Linkage``'s hyperstaticity calculation.


### Changed
- ``pylinkage/exception.py`` now handles exceptions in another file.
- Documentation improvements.
- Python style improvements.
- ``.gitignore`` now modifed from the standard GitHub gitignore example for Python.

### Fixed
- ``circle`` method of ``Pivot`` in ``pylinkage/linkage.py``. was causing errors
- ``tox.ini`` now fixed.

## [0.2.0] - 2021-06-14
### Added
- ``pylinkage/vizualizer.py`` view your linkages using matplotlib!
- Issue templates in ``.github/ISSUE_TEMPLATE/``
- ``.github/workflows/python-package-conda.yml``: conda tests with unittest workflow
- ``CODE_OF_CONDUCT.md``
- ``MANIFEST.in``
- ``README.md``
- ``environment.yml``
- ``setup.cfg`` now replaces ``setup.py``
- ``tox.ini``
- ``CHANGELOG.md``

### Changed
 -``.gitignore`` Python Package specific extensions added
 - ``MIT License`` → ``LICENSE``
 - ``lib/`` → ``pylinkage/``
 - ``tests/`` → ``pylinkage/tests/``
 - Revamped package organization.
 - Cleared ``setup.py``

## [0.0.1] - 2021-06-12
### Added
- ``lib/geometry.py`` as a mathematical for kinematic optimization
- ``lib/linkage.py``, linkage builder
- ``lib/optimizer.py``, with Particle Swarm Optimization (built-in and PySwarms), and exhaustive
- ``MIT License``
- ``requirements.txt``
- ``setup.py``
- ``tests/__init__.py``
- ``tests/test_geometry.py``
- ``tests/test_linkage.py``
- ``.gitignore``

