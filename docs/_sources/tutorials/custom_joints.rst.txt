Custom Joint Creation
=====================

This tutorial shows how to create custom joint types by extending the base
``Joint`` class. Custom joints let you model specialized mechanical constraints
not covered by the built-in joint types.

Understanding the Joint Interface
---------------------------------

All joints in pylinkage inherit from the abstract ``Joint`` class and must
implement three methods:

1. ``get_constraints()``: Returns the geometric constraints (distances, angles)
2. ``set_constraints()``: Sets the geometric constraints
3. ``reload(dt)``: Computes the joint's position based on its parents and constraints

The base ``Joint`` class provides:

- ``x``, ``y``: Current position coordinates
- ``joint0``, ``joint1``: Parent joints (can be ``None``)
- ``name``: Human-readable identifier
- ``coord()``: Returns ``(x, y)`` tuple
- ``set_coord(x, y)``: Sets position

Example: Slider Joint
---------------------

Let's create a slider joint that moves along a line defined by two parent points.
Unlike ``Linear`` which constrains to an infinite line, our slider will be
constrained to move only between its parent points.

Step 1: Define the Class
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pylinkage.joints.joint import Joint


   class Slider(Joint):
       """A joint that slides between two parent points.

       The joint's position is determined by a parameter ``t`` where:
       - t=0 means the joint is at joint0's position
       - t=1 means the joint is at joint1's position
       - Values between 0 and 1 interpolate linearly
       """

       __slots__ = ("t",)

       def __init__(
           self,
           x=0,
           y=0,
           joint0=None,
           joint1=None,
           t=0.5,
           name=None,
       ):
           """Create a Slider joint.

           :param x: Initial x position.
           :param y: Initial y position.
           :param joint0: First parent joint (start of slide).
           :param joint1: Second parent joint (end of slide).
           :param t: Position parameter (0 to 1).
           :param name: Joint name.
           """
           super().__init__(x, y, joint0, joint1, name)
           self.t = t

Step 2: Implement get_constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Return the geometric parameters that define this joint's motion:

.. code-block:: python

       def get_constraints(self):
           """Return the slide parameter as constraint."""
           return (self.t,)

Step 3: Implement set_constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Accept new constraint values:

.. code-block:: python

       def set_constraints(self, t=None):
           """Set the slide parameter.

           :param t: New position parameter (0 to 1).
           """
           if t is not None:
               self.t = t

Step 4: Implement reload
^^^^^^^^^^^^^^^^^^^^^^^^

Compute the joint's position based on parent positions and constraints:

.. code-block:: python

       def reload(self, dt=1):
           """Recompute position by interpolating between parents.

           :param dt: Time step (unused for this joint type).
           """
           if self.joint0 is None or self.joint1 is None:
               return

           # Get parent positions
           x0, y0 = self.joint0.coord()
           x1, y1 = self.joint1.coord()

           # Linear interpolation
           self.x = x0 + self.t * (x1 - x0)
           self.y = y0 + self.t * (y1 - y0)

Complete Slider Implementation
------------------------------

Here's the complete custom joint:

.. code-block:: python

   from pylinkage.joints.joint import Joint


   class Slider(Joint):
       """A joint that slides between two parent points."""

       __slots__ = ("t",)

       def __init__(
           self,
           x=0,
           y=0,
           joint0=None,
           joint1=None,
           t=0.5,
           name=None,
       ):
           super().__init__(x, y, joint0, joint1, name)
           self.t = t

       def get_constraints(self):
           return (self.t,)

       def set_constraints(self, t=None):
           if t is not None:
               self.t = t

       def reload(self, dt=1):
           if self.joint0 is None or self.joint1 is None:
               return
           x0, y0 = self.joint0.coord()
           x1, y1 = self.joint1.coord()
           self.x = x0 + self.t * (x1 - x0)
           self.y = y0 + self.t * (y1 - y0)

Using the Custom Joint
----------------------

Here's how to use the slider in a linkage:

.. code-block:: python

   import pylinkage as pl

   # Define anchor points
   p1 = pl.Static(0, 0, name="P1")
   p2 = pl.Static(4, 0, name="P2")

   # Create slider between points
   slider = Slider(
       joint0=p1,
       joint1=p2,
       t=0.5,        # Start in the middle
       name="Slider"
   )

   # Create a crank to drive motion
   crank = pl.Crank(
       joint0=(2, 2),
       angle=0,
       distance=1,
       name="Crank"
   )

   # Connect crank to slider with a revolute joint
   connector = pl.Revolute(
       joint0=crank,
       joint1=slider,
       distance0=2,
       distance1=0.5,
       name="Connector"
   )

   # Note: For this to work, slider.t would need to be updated
   # based on the mechanism's geometry, which requires more
   # sophisticated constraint solving.

Example: Oscillating Joint
--------------------------

Here's another example: a joint that oscillates sinusoidally over time.

.. code-block:: python

   import math
   from pylinkage.joints.joint import Joint


   class Oscillator(Joint):
       """A joint that moves sinusoidally around a center point."""

       __slots__ = ("amplitude", "frequency", "phase", "_time")

       def __init__(
           self,
           x=0,
           y=0,
           joint0=None,
           amplitude=1.0,
           frequency=1.0,
           phase=0.0,
           name=None,
       ):
           """Create an oscillating joint.

           :param joint0: Center point of oscillation.
           :param amplitude: Maximum displacement from center.
           :param frequency: Oscillation frequency.
           :param phase: Phase offset in radians.
           """
           super().__init__(x, y, joint0, None, name)
           self.amplitude = amplitude
           self.frequency = frequency
           self.phase = phase
           self._time = 0.0

       def get_constraints(self):
           return (self.amplitude, self.frequency, self.phase)

       def set_constraints(self, amplitude=None, frequency=None, phase=None):
           if amplitude is not None:
               self.amplitude = amplitude
           if frequency is not None:
               self.frequency = frequency
           if phase is not None:
               self.phase = phase

       def reload(self, dt=1):
           self._time += dt
           if self.joint0 is None:
               center_x, center_y = 0, 0
           else:
               center_x, center_y = self.joint0.coord()

           offset = self.amplitude * math.sin(
               self.frequency * self._time + self.phase
           )
           self.x = center_x + offset
           self.y = center_y

Best Practices
--------------

When creating custom joints:

1. **Use __slots__**: Define ``__slots__`` with your additional attributes to
   save memory and prevent accidental attribute creation.

2. **Handle None parents**: Check if parent joints are ``None`` in ``reload()``.

3. **Return tuples from get_constraints**: Always return a tuple, even if empty.

4. **Document constraints**: Clearly document what each constraint value means.

5. **Consider optimization**: If your joint will be used in optimization,
   ensure constraints are continuous values that can be interpolated.

6. **Validate inputs**: Consider raising ``NotCompletelyDefinedError`` if
   required parameters are missing.

Next Steps
----------

- :doc:`advanced_optimization` - Optimize linkages with custom joints
- See :py:mod:`pylinkage.joints` for built-in joint implementations
