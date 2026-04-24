"""Tests for the exceptions module."""

from __future__ import annotations

import pytest

from pylinkage.exceptions import (
    NotCompletelyDefinedError,
    OptimizationError,
    UnbuildableError,
    UnderconstrainedError,
)


class TestUnbuildableError:
    def test_default_message(self):
        err = UnbuildableError(joint="J1")
        assert err.joint == "J1"
        assert err.message == "Unable to solve constraints"

    def test_custom_message(self):
        err = UnbuildableError(joint="J2", message="cannot build")
        assert err.joint == "J2"
        assert err.message == "cannot build"

    def test_str(self):
        err = UnbuildableError(joint="foo", message="bad")
        assert str(err) == "bad on foo"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(UnbuildableError) as info:
            raise UnbuildableError(joint="J")
        assert info.value.joint == "J"

    def test_subclass_of_exception(self):
        assert issubclass(UnbuildableError, Exception)


class TestUnderconstrainedError:
    def test_default_message(self):
        err = UnderconstrainedError(linkage="linkage1")
        assert err.linkage == "linkage1"
        assert str(err) == "The linkage is under-constrained!"

    def test_custom_message(self):
        err = UnderconstrainedError(linkage="linkage2", message="needs constraints")
        assert err.linkage == "linkage2"
        assert str(err) == "needs constraints"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(UnderconstrainedError):
            raise UnderconstrainedError(linkage="x")


class TestNotCompletelyDefinedError:
    def test_default_message(self):
        err = NotCompletelyDefinedError(joint="incomplete")
        assert err.joint == "incomplete"
        assert str(err) == "The joint is not completely defined!"

    def test_custom_message(self):
        err = NotCompletelyDefinedError(joint="J", message="missing params")
        assert err.joint == "J"
        assert str(err) == "missing params"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(NotCompletelyDefinedError):
            raise NotCompletelyDefinedError(joint="X")


class TestOptimizationError:
    def test_default_message(self):
        err = OptimizationError()
        assert err.message == "Optimization failed"
        assert str(err) == "Optimization failed"

    def test_custom_message(self):
        err = OptimizationError(message="convergence failed")
        assert err.message == "convergence failed"
        assert str(err) == "convergence failed"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(OptimizationError):
            raise OptimizationError("bad")


class TestExceptionsExportedFromPackage:
    def test_imports_work(self):
        from pylinkage import (
            NotCompletelyDefinedError as NCDE,
        )
        from pylinkage import (
            OptimizationError as OE,
        )
        from pylinkage import (
            UnbuildableError as UE,
        )
        from pylinkage import (
            UnderconstrainedError as UCE,
        )

        assert NCDE is NotCompletelyDefinedError
        assert OE is OptimizationError
        assert UE is UnbuildableError
        assert UCE is UnderconstrainedError
