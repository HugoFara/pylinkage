from __future__ import annotations

import pytest

import pylinkage.visualizer as viz


class TestLazyAttrs:
    def test_all_lazy_attrs_loadable(self):
        from pylinkage.visualizer import _LAZY_ATTRS

        for name in _LAZY_ATTRS:
            val = getattr(viz, name)
            assert val is not None

    def test_cached_after_first_access(self):
        name = "LINK_COLORS"
        first = getattr(viz, name)
        second = getattr(viz, name)
        assert first is second

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = viz.this_does_not_exist

    def test_all_exports_declared(self):
        for name in viz.__all__:
            assert hasattr(viz, name)
