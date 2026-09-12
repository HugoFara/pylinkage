Custom Components
=================

This tutorial shows how to create custom kinematic elements by extending the
base ``Component`` classes. Custom components let you model constraints not
covered by the built-in actuators and dyads, and they take part in
simulation and optimization exactly like the built-in ones.

Understanding the Component Interface
-------------------------------------

Everything a ``Linkage`` simulates is a ``Component`` from
:mod:`pylinkage.components`. The abstract base class provides:

- ``x``, ``y``: the current position (``None`` until first solved)
- ``name``: a human-readable identifier
- ``position`` / ``coord()``: the ``(x, y)`` tuple
- ``set_coord(x, y)``: set the position directly
- ``velocity``, ``acceleration``: filled in by kinematic analysis

and requires three methods:

1. ``get_constraints()``: the geometric parameters (distances, angles, …)
   as a tuple. Optimizers concatenate these across the linkage.
2. ``set_constraints(*values)``: accept new values, in the same order.
3. ``reload(dt=1)``: recompute the position from the parents' positions.

Components that depend on other components subclass ``ConnectedComponent``
and add an ``anchors`` property returning the parents. The ``Linkage`` uses
it to find a solve order: grounds first, then actuators, then every
component whose anchors are already solved. ``BinaryDyad`` (from
:mod:`pylinkage.dyads`) is the ready-made two-anchor version, with
``anchor1``/``anchor2`` slots and an ``anchors`` property that resolves
``crank.output``-style proxies.

Example: Slider Joint
---------------------

Let's create a joint that sits a fixed fraction of the way between two
parent points — a pivot whose position along a rail is itself a design
parameter.

Step 1: Define the Class
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pylinkage.dyads import BinaryDyad


   class Slider(BinaryDyad):
       """A joint that sits a fraction ``t`` of the way from anchor1 to anchor2.

       - t=0 puts the joint on anchor1
       - t=1 puts the joint on anchor2
       - values in between interpolate linearly
       """

       __slots__ = ("t",)

       def __init__(self, anchor1, anchor2, t=0.5, name=None):
           """Create a Slider joint.

           :param anchor1: Parent component at the start of the slide.
           :param anchor2: Parent component at the end of the slide.
           :param t: Position parameter (0 to 1).
           :param name: Joint name.
           """
           super().__init__(None, None, name)  # position is computed below
           self.anchor1 = anchor1
           self.anchor2 = anchor2
           self.t = t
           self.reload()

Step 2: Implement get_constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Return the geometric parameters that define this joint's motion:

.. code-block:: python

       def get_constraints(self):
           """Return the slide parameter as constraint."""
           return (self.t,)

Step 3: Implement set_constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Accept new constraint values. ``Linkage.set_constraints`` passes exactly as
many positional values as ``get_constraints`` returned; the trailing
``*args`` keeps the signature compatible with the built-in dyads:

.. code-block:: python

       def set_constraints(self, t=None, *args):
           """Set the slide parameter.

           :param t: New position parameter (0 to 1).
           """
           if t is not None:
               self.t = t

Step 4: Implement reload
^^^^^^^^^^^^^^^^^^^^^^^^

Compute the joint's position from the parents' positions and the
constraints. ``_get_anchor_position`` reads either a component or an
actuator's ``output`` proxy:

.. code-block:: python

       def reload(self, dt=1):
           """Recompute position by interpolating between the anchors.

           :param dt: Time step (unused: the slider does not move on its own).
           """
           x0, y0 = self._get_anchor_position(self.anchor1)
           x1, y1 = self._get_anchor_position(self.anchor2)
           if None in (x0, y0, x1, y1):
               raise ValueError(f"{self.name}: anchors have no position yet")

           # Linear interpolation
           self.x = x0 + self.t * (x1 - x0)
           self.y = y0 + self.t * (y1 - y0)

Complete Slider Implementation
------------------------------

Here's the complete custom component:

.. code-block:: python

   from pylinkage.dyads import BinaryDyad


   class Slider(BinaryDyad):
       """A joint that sits a fraction ``t`` of the way from anchor1 to anchor2."""

       __slots__ = ("t",)

       def __init__(self, anchor1, anchor2, t=0.5, name=None):
           super().__init__(None, None, name)
           self.anchor1 = anchor1
           self.anchor2 = anchor2
           self.t = t
           self.reload()

       def get_constraints(self):
           return (self.t,)

       def set_constraints(self, t=None, *args):
           if t is not None:
               self.t = t

       def reload(self, dt=1):
           x0, y0 = self._get_anchor_position(self.anchor1)
           x1, y1 = self._get_anchor_position(self.anchor2)
           if None in (x0, y0, x1, y1):
               raise ValueError(f"{self.name}: anchors have no position yet")
           self.x = x0 + self.t * (x1 - x0)
           self.y = y0 + self.t * (y1 - y0)

Using the Custom Component
--------------------------

The slider is a pivot on a rail: a rocker hangs from it, driven by a crank.
Nothing special is needed to put it in a ``Linkage``; the solve order
follows from ``anchors``.

.. code-block:: python

   import pylinkage as pl
   from pylinkage.actuators import Crank
   from pylinkage.components import Ground
   from pylinkage.dyads import RRRDyad
   from pylinkage.simulation import Linkage

   # The rail
   p1 = Ground(0.0, 0.0, name="P1")
   p2 = Ground(4.0, 0.0, name="P2")

   # A pivot halfway along it (custom component, see above)
   pivot = Slider(anchor1=p1, anchor2=p2, t=0.5, name="Pivot")

   # A crank, and a rocker between the crank tip and the pivot
   motor = Ground(1.0, 3.0, name="Motor")
   crank = Crank(anchor=motor, radius=1.0, angular_velocity=0.1, name="Crank")
   rocker = RRRDyad(
       anchor1=crank.output,
       anchor2=pivot,
       distance1=3.0,
       distance2=2.5,
       name="Rocker",
   )

   linkage = Linkage([p1, p2, motor, pivot, crank, rocker], name="Sliding pivot")

   loci = list(linkage.step())
   print(f"Pivot at {pivot.position}, {len(loci)} steps simulated")
   print(f"Constraints: {linkage.get_constraints()}")

   pl.show_linkage(linkage)

**Expected output:**

.. code-block:: text

   Pivot at (2.0, 0.0), 63 steps simulated
   Constraints: [0.5, 1.0, 3.0, 2.5]

The slider's ``t`` is the first constraint, followed by the crank radius
and the rocker's two distances. Because it is a constraint, the optimizers
can move the pivot along the rail like any other dimension — here to
maximize the rocker's horizontal travel:

.. code-block:: python

   @pl.kinematic_maximization
   def stride(loci, **kwargs):
       """Horizontal travel of the rocker joint."""
       xs = [step[-1][0] for step in loci]
       return max(xs) - min(xs)


   lower, upper = pl.generate_bounds(
       linkage.get_constraints(), min_ratio=1.5, max_factor=1.5,
   )
   lower[0], upper[0] = 0.0, 1.0  # the slider fraction stays on the rail

   results = pl.particle_swarm_optimization(
       stride, linkage, bounds=(lower, upper),
       n_particles=20, iterations=20, verbose=False,
   )
   best = results[0]
   print(f"Stride {best.score:.3f} with the pivot at t={best.dimensions[0]:.2f}")

Example: Oscillating Joint
--------------------------

Here's another example: a joint that oscillates sinusoidally over time
around a single parent. With one anchor it subclasses
``ConnectedComponent`` directly and defines ``anchors`` itself.

.. code-block:: python

   import math
   from pylinkage.components import ConnectedComponent, Ground
   from pylinkage.simulation import Linkage


   class Oscillator(ConnectedComponent):
       """A joint that moves sinusoidally around a center point."""

       __slots__ = ("anchor", "amplitude", "period", "_time")

       def __init__(self, anchor, amplitude=1.0, period=60, name=None):
           """Create an oscillating joint.

           :param anchor: Center point of oscillation.
           :param amplitude: Maximum displacement from center.
           :param period: Steps per oscillation.
           :param name: Joint name.
           """
           super().__init__(None, None, name)
           self.anchor = anchor
           self.amplitude = amplitude
           self.period = period
           self._time = 0
           self.reload()

       @property
       def anchors(self):
           return (self.anchor,)

       def get_constraints(self):
           return (self.amplitude,)

       def set_constraints(self, amplitude=None, *args):
           if amplitude is not None:
               self.amplitude = amplitude

       def reload(self, dt=1):
           cx, cy = self.anchor.position
           phase = math.tau * self._time / self.period
           self.x = cx + self.amplitude * math.sin(phase)
           self.y = cy
           self._time += dt


   centre = Ground(0.0, 2.0, name="Centre")
   oscillator = Oscillator(centre, amplitude=1.0, period=60, name="Oscillator")
   linkage = Linkage([centre, oscillator])

   xs = [step[-1][0] for step in linkage.step(iterations=60)]
   print(f"x swings between {min(xs):.2f} and {max(xs):.2f}")

**Expected output:**

.. code-block:: text

   x swings between -1.00 and 1.00

Two things to know about a component that moves on its own:

- ``Linkage.step`` passes ``dt`` only to the built-in actuators (``Crank``,
  ``ArcCrank``, ``LinearActuator``); every other component gets
  ``reload()`` with the default ``dt``. Keep time yourself, as above.
- ``Linkage.get_rotation_period()`` knows only the built-in actuators, so
  pass ``iterations`` to ``step()`` explicitly.

Best Practices
--------------

When creating custom components:

1. **Use __slots__**: define ``__slots__`` with your additional attributes.
   The base classes use slots, so a subclass without them would silently
   grow a ``__dict__``.

2. **Compute the position in __init__**: call ``reload()`` at the end of
   the constructor when the anchors already have a position, so that
   ``get_coords()`` and the initial drawing are meaningful.

3. **Read anchors through ``_get_anchor_position``** (or ``.position``):
   an anchor may be an actuator's ``output`` proxy rather than a component.

4. **Return tuples from get_constraints**: always a tuple, even if empty,
   and accept the same number of positional values in ``set_constraints``.

5. **Document constraints**: clearly document what each constraint value
   means and its valid range; optimizers explore between bounds and know
   nothing else about it.

6. **Raise ``UnbuildableError`` for impossible geometry**: the
   ``@kinematic_minimization`` / ``@kinematic_maximization`` decorators
   turn it into an infinite penalty, which is how the built-in dyads
   reject configurations they cannot assemble.

7. **Keep the numba solver in mind**: ``step_fast()`` and the ``Ensemble``
   batch simulation only know the built-in components and raise
   ``NotImplementedError`` for a custom one. ``step()``, the optimizers
   and the visualizers work with any component.

Next Steps
----------

- :doc:`advanced_optimization` - Optimize linkages with custom components
- See :py:mod:`pylinkage.components` and :py:mod:`pylinkage.dyads` for the
  base classes and the built-in implementations to model on
