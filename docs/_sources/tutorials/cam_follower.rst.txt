Cam-Follower Mechanisms
=======================

This tutorial covers the cam-follower module in pylinkage, which allows you to
create mechanisms with disk/plate cams driving translating or oscillating followers.

Introduction
------------

A cam-follower mechanism converts rotary motion into reciprocating or oscillating
motion. The cam is a rotating disk with a specially shaped profile, and the follower
is a component that traces this profile, producing the desired output motion.

Pylinkage supports:

- **Translating followers**: Move along a fixed linear guide
- **Oscillating followers**: Pivot about a fixed point (rocker arm)
- **Knife-edge followers**: Direct contact with cam surface (roller_radius=0)
- **Roller followers**: Rolling contact via a roller (roller_radius>0)

Motion Laws
-----------

Motion laws define the dimensionless displacement function that shapes the cam profile.
All motion laws produce normalized displacement s(u) where u is in [0,1] and s is in [0,1].

Available Motion Laws
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pylinkage.cam import (
       HarmonicMotionLaw,
       CycloidalMotionLaw,
       ModifiedTrapezoidalMotionLaw,
       PolynomialMotionLaw,
       polynomial_345,
       polynomial_4567,
   )

**Harmonic Motion Law**

Simple cosine-based motion. Has non-zero acceleration at boundaries.

.. code-block:: python

   law = HarmonicMotionLaw()
   print(law.displacement(0.0))  # 0.0 (start)
   print(law.displacement(0.5))  # 0.5 (midpoint)
   print(law.displacement(1.0))  # 1.0 (end)

**Cycloidal Motion Law**

Zero velocity and acceleration at boundaries (smooth start/stop).
Higher peak acceleration than harmonic.

.. code-block:: python

   law = CycloidalMotionLaw()
   print(law.velocity(0.0))  # 0.0 (starts smoothly)
   print(law.velocity(0.5))  # 2.0 (maximum velocity at midpoint)

**Modified Trapezoidal Motion Law**

Lower peak acceleration, good for high-load applications.

.. code-block:: python

   law = ModifiedTrapezoidalMotionLaw()

**Polynomial Motion Laws**

Polynomial motion with custom coefficients. The 3-4-5 and 4-5-6-7 polynomials
are common choices with zero velocity/acceleration at boundaries.

.. code-block:: python

   # 3-4-5 polynomial (zero velocity and acceleration at ends)
   law = polynomial_345()

   # 4-5-6-7 polynomial (also zero jerk at ends)
   law = polynomial_4567()

   # Custom polynomial coefficients
   law = PolynomialMotionLaw([0, 0, 0, 10, -15, 6])

Cam Profiles
------------

Cam profiles define how the cam radius varies with rotation angle.

FunctionProfile
^^^^^^^^^^^^^^^

Creates a profile from a motion law with rise-dwell-fall-dwell timing.

.. code-block:: python

   import math
   from pylinkage.cam import FunctionProfile, CycloidalMotionLaw

   profile = FunctionProfile(
       motion_law=CycloidalMotionLaw(),
       base_radius=1.0,      # Minimum radius
       total_lift=0.5,       # Maximum displacement
       rise_start=0.0,       # Angle where rise begins
       rise_end=math.pi/2,   # Angle where rise ends (dwell-high starts)
       dwell_high_end=math.pi,  # Angle where dwell-high ends (fall starts)
       fall_end=3*math.pi/2, # Angle where fall ends (dwell-low starts)
   )

   # Evaluate the profile
   print(profile.evaluate(0.0))         # 1.0 (base radius)
   print(profile.evaluate(math.pi/4))   # Rising...
   print(profile.evaluate(math.pi))     # 1.5 (base + lift, at peak)

PointArrayProfile
^^^^^^^^^^^^^^^^^

Creates a profile from discrete (angle, radius) points using spline interpolation.

.. code-block:: python

   import math
   from pylinkage.cam import PointArrayProfile

   angles = [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
   radii = [1.0, 1.2, 1.5, 1.5, 1.2, 1.0]

   profile = PointArrayProfile(angles=angles, radii=radii, periodic=True)

   # Evaluate at any angle (spline interpolation)
   print(profile.evaluate(math.pi/4))  # 1.2

Translating Cam Follower
------------------------

A translating follower moves linearly along a fixed guide axis based on the cam rotation.

Basic Example
^^^^^^^^^^^^^

.. code-block:: python

   import math
   from pylinkage.components import Ground
   from pylinkage.actuators import Crank
   from pylinkage.dyads import TranslatingCamFollower
   from pylinkage.simulation import Linkage
   from pylinkage.cam import FunctionProfile, HarmonicMotionLaw

   # Fixed cam center
   cam_center = Ground(0.0, 0.0, name="cam_center")

   # Cam driver (rotating crank)
   cam_crank = Crank(
       anchor=cam_center,
       radius=0.1,           # Small radius (just for visualization)
       angular_velocity=0.1, # Rotation speed
       name="cam"
   )

   # Cam profile
   profile = FunctionProfile(
       motion_law=HarmonicMotionLaw(),
       base_radius=1.0,
       total_lift=0.5,
       rise_start=0.0,
       rise_end=math.pi,
       dwell_high_end=math.pi,
       fall_end=2*math.pi,
   )

   # Guide point (defines direction of follower motion)
   guide = Ground(0.0, 1.5, name="guide")

   # Translating follower
   follower = TranslatingCamFollower(
       cam_driver=cam_crank,
       profile=profile,
       guide=guide,
       guide_angle=math.pi/2,  # Vertical motion
       roller_radius=0.0,      # Knife-edge follower
       name="follower"
   )

   # Create linkage
   linkage = Linkage(
       components=[cam_center, guide, cam_crank, follower],
       name="Cam mechanism"
   )

   # Simulate
   for positions in linkage.step():
       follower_pos = follower.output.position
       print(f"Follower at: {follower_pos}")

Roller Follower
^^^^^^^^^^^^^^^

For a roller follower, set ``roller_radius`` to a positive value:

.. code-block:: python

   follower = TranslatingCamFollower(
       cam_driver=cam_crank,
       profile=profile,
       guide=guide,
       guide_angle=math.pi/2,
       roller_radius=0.1,  # Roller with 0.1 unit radius
   )

Oscillating Cam Follower
------------------------

An oscillating follower pivots about a fixed point, with an arm that traces the cam profile.

Basic Example
^^^^^^^^^^^^^

.. code-block:: python

   import math
   from pylinkage.components import Ground
   from pylinkage.actuators import Crank
   from pylinkage.dyads import OscillatingCamFollower
   from pylinkage.simulation import Linkage
   from pylinkage.cam import FunctionProfile, CycloidalMotionLaw

   # Fixed points
   cam_center = Ground(0.0, 0.0, name="cam_center")
   pivot = Ground(2.0, 0.0, name="pivot")

   # Cam driver
   cam_crank = Crank(
       anchor=cam_center,
       radius=0.1,
       angular_velocity=0.1,
   )

   # Profile that maps cam angle to angular displacement of the arm
   profile = FunctionProfile(
       motion_law=CycloidalMotionLaw(),
       base_radius=0.0,          # Base angle offset (radians)
       total_lift=math.pi/4,     # Maximum angular swing (45 degrees)
       rise_start=0.0,
       rise_end=math.pi,
       dwell_high_end=math.pi,
       fall_end=2*math.pi,
   )

   # Oscillating follower
   follower = OscillatingCamFollower(
       cam_driver=cam_crank,
       profile=profile,
       pivot_anchor=pivot,
       arm_length=1.5,
       initial_angle=math.pi/2,  # Arm points upward at rest
       roller_radius=0.0,
   )

   linkage = Linkage(
       components=[cam_center, pivot, cam_crank, follower],
   )

   for positions in linkage.step():
       output_pos = follower.output.position
       print(f"Arm end at: {output_pos}")

Connecting to Downstream Linkages
---------------------------------

Cam followers can drive downstream linkages. The ``output`` property provides
an anchor point that other dyads can connect to.

Example: Cam-Driven Four-Bar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import math
   from pylinkage.components import Ground
   from pylinkage.actuators import Crank
   from pylinkage.dyads import TranslatingCamFollower, RRRDyad
   from pylinkage.simulation import Linkage
   from pylinkage.cam import FunctionProfile, HarmonicMotionLaw

   # Ground points
   cam_center = Ground(0.0, 0.0, name="cam_center")
   guide = Ground(0.0, 2.0, name="guide")
   rocker_pivot = Ground(3.0, 0.0, name="rocker_pivot")

   # Cam mechanism
   cam_crank = Crank(anchor=cam_center, radius=0.1, angular_velocity=0.1)

   profile = FunctionProfile(
       motion_law=HarmonicMotionLaw(),
       base_radius=1.0,
       total_lift=1.0,
       rise_start=0.0,
       rise_end=math.pi,
       dwell_high_end=math.pi,
       fall_end=2*math.pi,
   )

   follower = TranslatingCamFollower(
       cam_driver=cam_crank,
       profile=profile,
       guide=guide,
       guide_angle=math.pi/2,
       roller_radius=0.0,
   )

   # Connect downstream rocker via RRRDyad
   rocker = RRRDyad(
       anchor1=follower.output,  # Connects to follower output
       anchor2=rocker_pivot,
       distance1=2.5,
       distance2=2.0,
   )

   linkage = Linkage(
       components=[cam_center, guide, rocker_pivot, cam_crank, follower, rocker],
   )

   # Simulate the complete mechanism
   for positions in linkage.step():
       rocker_pos = rocker.output.position
       print(f"Rocker output at: {rocker_pos}")

Profile Analysis
----------------

Cam profiles provide methods for mechanical analysis.

Pressure Angle
^^^^^^^^^^^^^^

The pressure angle indicates the efficiency of force transmission.
High pressure angles (>30 degrees) can cause binding.

.. code-block:: python

   import math

   # Check pressure angle at various cam positions
   for angle in [0, math.pi/4, math.pi/2, math.pi]:
       pressure = profile.pressure_angle(angle)
       print(f"At {math.degrees(angle):.0f} deg: pressure angle = {math.degrees(pressure):.1f} deg")

Pitch Radius
^^^^^^^^^^^^

For roller followers, the pitch curve is the path traced by the roller center.

.. code-block:: python

   roller_radius = 0.1
   for angle in [0, math.pi/4, math.pi/2]:
       pitch = profile.pitch_radius(angle, roller_radius)
       print(f"Pitch radius at {math.degrees(angle):.0f} deg: {pitch:.3f}")

Optimization
------------

Profile parameters can be optimized using pylinkage's constraint system.

.. code-block:: python

   # Get optimizable constraints
   constraints = profile.get_constraints()
   print(constraints)  # (base_radius, total_lift)

   # Modify constraints
   profile.set_constraints(base_radius=1.2, total_lift=0.6)

For cam followers in a linkage:

.. code-block:: python

   # Get all constraints from the follower
   constraints = list(follower.get_constraints())
   # Returns: roller_radius + profile constraints

   # These can be used with PSO optimization
   from pylinkage.optimization import particle_swarm_optimization

Next Steps
----------

- :doc:`advanced_optimization` - Optimize cam profile parameters with PSO
- :doc:`kinematics_optimization` - Analyze velocities and accelerations
- :doc:`custom_joints` - Create custom dyad types
