# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2] - 2021-06-14
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
- ``tests/\_\_init\_\_.py``
- ``tests/test_geometry.py``
- ``tests/test_linkage.py``
- ``.gitignore``

