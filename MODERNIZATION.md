# Pylinkage Modernization Guide

This document tracks remaining improvements for pylinkage.

**Completed milestones:**
- ✅ Packaging consolidation (uv, pyproject.toml, hatchling)
- ✅ CI/CD improvements (pytest, coverage, Codecov, docs workflow)
- ✅ Type hints migration (PEP 484/604 annotations, `_types.py`)
- ✅ Python 3.9+ support (removed 3.7/3.8 compatibility code)
- ✅ Test infrastructure (pytest, 82% coverage, hypothesis)
- ✅ API enhancements (serialization, context manager, `__all__` exports)
- ✅ Code quality (f-strings, module exports, TODO resolution)
- ✅ Async optimization (progress callbacks, cancellation support)

---

## Remaining Improvements

### 1. Documentation Enhancements

**Priority: Medium**

The Sphinx setup is functional but could be improved:

1. **Add doctest examples** to docstrings for self-documenting code:

```python
def circle_intersect(c1, r1, c2, r2):
    """Find intersection points of two circles.

    Examples:
        >>> circle_intersect((0, 0), 1, (1, 0), 1)
        ((0.5, 0.866...), (0.5, -0.866...))
    """
```

2. **Add tutorials** in `sphinx/tutorials/`:
   - Getting started guide
   - Custom joint creation
   - Advanced optimization techniques

3. **API changelog** - document breaking changes per version

---

## Resources

- [PEP 621](https://peps.python.org/pep-0621/) - pyproject.toml metadata
- [PEP 484](https://peps.python.org/pep-0484/) - Type hints
- [pytest-cov](https://pytest-cov.readthedocs.io/) - Coverage plugin
- [mypy](https://mypy.readthedocs.io/) - Static type checker
