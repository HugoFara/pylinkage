"""Machinery for deprecating a public name without breaking imports.

A deprecated name is removed from its module's namespace and served by a
module-level ``__getattr__`` (PEP 562) instead. Reading it still works —
``from pylinkage.dyads import Dyad`` resolves as it always did — but now
emits a :class:`DeprecationWarning` naming the replacement and the release
that will drop it.

Deprecations follow the policy in :doc:`/deprecations`: a name is announced
in one release and removed no earlier than the next major one.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["DeprecatedAlias", "deprecated_getattr"]


@dataclass(frozen=True)
class DeprecatedAlias:
    """A public name kept alive only for backwards compatibility.

    Attributes:
        value: Object the deprecated name resolves to.
        replacement: Fully qualified name callers should use instead.
        removed_in: pylinkage version that will stop providing the name.
        reason: Why the name is going away. Appended to the warning.
    """

    value: Any
    replacement: str
    removed_in: str
    reason: str = ""

    def message(self, module: str, name: str) -> str:
        """Build the warning text shown for ``module.name``.

        Args:
            module: Module the deprecated name is read from.
            name: The deprecated name itself.

        Returns:
            A sentence naming the replacement and the removal release,
            followed by the reason when one is given.
        """
        text = (
            f"{module}.{name} is deprecated and will be removed in "
            f"pylinkage {self.removed_in}; use {self.replacement} instead."
        )
        return f"{text} {self.reason}" if self.reason else text


def deprecated_getattr(
    module: str,
    aliases: Mapping[str, DeprecatedAlias],
) -> Callable[[str], Any]:
    """Build a module-level ``__getattr__`` that serves deprecated aliases.

    Bind the result as ``__getattr__`` at module scope. Names absent from
    ``aliases`` raise :class:`AttributeError` as usual, so the hook stays
    invisible to everything else.

    Args:
        module: ``__name__`` of the module installing the hook.
        aliases: Deprecated name mapped to the alias describing it.

    Returns:
        A function suitable for binding as the module's ``__getattr__``.
    """

    def __getattr__(name: str) -> Any:
        alias = aliases.get(name)
        if alias is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        warnings.warn(alias.message(module, name), DeprecationWarning, stacklevel=2)
        return alias.value

    return __getattr__
