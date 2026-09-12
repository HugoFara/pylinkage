pylinkage
=========

The packages below are listed alphabetically. To define and simulate a
mechanism you need four of them, in this order: :mod:`pylinkage.components`
(frame points), :mod:`pylinkage.actuators` (motor inputs),
:mod:`pylinkage.dyads` (the constrained pairs that close the loops) and
:mod:`pylinkage.simulation` (the ``Linkage`` container). The
:doc:`getting started tutorial </tutorials/getting_started>` walks through
them; the rest are what you reach for afterwards.

A name is public when it appears in a package's ``__all__``; modules inside a
package are implementation. See :doc:`/deprecations` for the rule and for
every name currently on its way out.

.. toctree::
   :maxdepth: 4

   pylinkage
