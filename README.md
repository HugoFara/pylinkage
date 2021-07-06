[![PyPI version fury.io](https://badge.fury.io/py/pylinkage.svg)](https://pypi.python.org/pypi/pylinkage/)
# pylinkage

A linkage builder written in Python. This package is made to create planar linkages and optimize them kinematically thanks to [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization). It is still an early work, so it should receive great changes in the future.

## Installation
### Using pip
This package is in the PyPi as [pylinkage](https://pypi.org/project/pylinkage/), and can be installed using:
``pip install pylinkage``

It is the recommended way downloading it, since the release versions are synced.

### Setting up Virtual Environment
We provide an [environment.yml](https://github.com/HugoFara/leggedsnake/environment.yml) file for conda. Use ``conda env update --file environment.yml --name pylinkage-env`` to install the requirements in a separate environment. 

## Usage

As of today, the code is segmented in three parts:
* [geometry.py](https://github.com/HugoFara/pylinkage/blob/main/pylinkage/geometry.py) that module handles geometric primitives, such as circle intersections, distance claculation. It works in Euclidian space only. Aside from ``dist`` and ``sqr_dist`` functions, you might not use it directly.
* [linkage.py](https://github.com/HugoFara/pylinkage/blob/main/pylinkage/linkage.py) this module describes joints and linkages 
  * Due to the geometric approach, joints (instances of ``Joint`` object) are defined without links. 
  * The ``Linkage`` class that will make your code shorter.
* [optimizer.py](https://github.com/HugoFara/pylinkage/blob/main/pylinkage/optimizer.py) proposes three optimizations based on three techniques:
  * The "exhaustive" optimization (``exhaustive_optimization`` function) is a dumb optimization method, consisting or trying sequencially all positions. It is here for demonstration purposes only, and you should not use it if you are looking for an efficient technique.
  * The built-in Particle Swarm Optimizer (PSO). I started with it, so it offers a large set of useful options for linkage optimization. However, it is here for legacy purposes, and is much short than the PySwarms module.
  * PSO using [PySwarms](https://github.com/ljvmiranda921/pyswarms). We provide a wrapper function to PySwarm from ljvmiranda921, that will progressively be extended.
* [visualizer.py](https://github.com/HugoFara/pylinkage/blob/main/pylinkage/visualizer.py) can make graphic illustrations of your linkage using matplotlib.
  * It is also used to visualise your n-dimensional swarm, which is not supported by PySwarms.

## Requirements

Python 3, numpy for calculation, matplotlib for drawing, and standard libraries. You will also need PySwarms for the optimization since the built-in PSO is deprecated and will be removed soon.

## Example

Let's start with a crank-rocker [four-bar linkage](https://en.wikipedia.org/wiki/Four-bar_linkage), as classic of mechanics. 

### Joints definition
The first thing you need if to define points belonging to the frame:

```python
import pylinkage.linkage as pl

frame_first = pl.Static(0, 0)
frame_second = pl.Static(3, 0)
```
* ``O, O``: "x, y" position of the first point.
* ``3, 0``: same for the second point. 

Then we have to define at least one crank because we do a kinematic simulation.
```python
crank = pl.Crank(0, 1, joint0=frame_first, angle=0.31, distance=1)
```

Here you need some explanations: 
* ``0, 1``: x and y initial coordinates of the **tail** of the crank link.
* ``joint0``: the parent Joint to link with. The pin will be created on the position of the parent, which is the head of the crank link.
* ``angle``: the crank will rotate with this angle, in radians, at each iteration.
* ``distance``: distance to keep constant between crank link tail and head.

Now we add a pin joint to close the kinematic loop.
```python
pin = pl.Pivot(3, 2, joint0=crank, joint1=frame_second, distance0=3, distance1=1)
```
Here comes some help:
* ``joint0``, ``joint1``: first and second ``Joint``s you want to link to, the order is not important.
* ``distance0``, ``distance1``: distance to keep constant between this joint and his two parents.

And here comes the pain:
Why do we specify initial coordinates ``3, 2``? Moreover, they seem incompatible with distance to parents/parents' positions! 
  * This explanation is simple: mathematically a pin joint the intersection of two circles. The intersection is often two points. To choose the strating point, we calculate both intersection (when possible), then we keep the intersection closer to the previous position as the solution. 


Wait! A linkage with a single motor and only one pin joint? That doesn't make sense!
:Behind the curtain, many joints are created on the fly. When you define a ``Crank`` joint, it creates a motor **and** a pin joint on the crank's link head. For a ``Pivot`` joint, it creates **3 pin joints**: one on the position of each of its parents, and one its position, forming a reshapable triangle. This is why pylinkage is so short to write.

### Linkage definition and simulation
Once your linkage is finished, you can either can the ``reload`` method of each ``Joint`` in a loop, or put everything in a ``Linkage`` that will handle this can of thinks for you. 

Linkage definition is simple:
```python
my_linkage = pl.Linkage(joints=(frame_first, frame_second, crank, pin))
```
That's all!

Now we want to simulate it and to get the locus of ``pin``. Just use the ``step`` method of ``Linkage`` to make a complete rotation.
```python
locus = my_linkage.step()
```
You can also specify the number of steps with the ``iteration`` argument, or subdivisions of each iteration with``dt``.

Let's recape.
```python
import pylinkage.linkage as pl

# Static points in space, belonging to the frame
frame_first = pl.Static(0, 0)
frame_second = pl.Static(3, 0)
# Main motor
crank = pl.Crank(0, 1, joint0=frame_first, angle=.31, distance=1)
# Close the loop
pin = pl.Pivot(3, 2, joint0=crank, joint1=frame_second, 
               distance0=3, distance1=1)

my_linkage = pl.Linkage(joints=(frame_first, frame_second, crank, pin))

locus = my_linkage.step()
```

### Visualization
Firsting first, you made a cool linkage, but only you know what it is. Let's add friendly names to joints, so the communication is simplified.
```python
frame_first.name = "A"
crank.name = "B"
pin.name = "C"
frame_second.name = "D"
# Linkage can also have names
my_linkage.name = "Four-bar linkage"
```

Then you can view your linkage!

```python
import pylinkage.visualizer as visu

visu.show_linkage(my_linkage)
```
![A four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/pylinkage/examples/images/Kinematic%20My%20four-bar%20linkage.gif)

Last recap, rearranging names:
```python
import pylinkage.linkage as pl
import pylinkage.visualizer as visu

# Static points in space, belonging to the frame
frame_first = pl.Static(0, 0, name="A")
frame_second = pl.Static(3, 0, name="D")
# Main motor
crank = pl.Crank(0, 1, joint0=frame_first, angle=.31, distance=1, name="B")
# Close the loop
pin = pl.Pivot(3, 2, joint0=crank, joint1=frame_second,
               distance0=3, distance1=1, name="C")

# Linkage definition
my_linkage = pl.Linkage(
    joints=(crank, pin),
    order=(crank, pin),
    name="My four-bar linkage"
)

# Visualization
visu.show_linkage(my_linkage)
```

### Optimization
Now, we want automatic optimization of our linkage, using a certain criterion. Let's find a four-bar linkage that make a quarter of a circle. It is a common problem if you want to build a [windscreen wiper](https://en.wikipedia.org/wiki/Windscreen_wiper) for instance.

Our objective function, often called the fitness function, is the following:
```python
from pylinkage.geometry import bounding_box

# We save initial position because we don't want a completely different movement
init_pos = my_linkage.get_coords()

def fitness_func(linkage, params, *args):
    """
    Return how fit the locus is to describe a quarter of circle.

    It is a minisation problem and the theorical best score is 0.
    """
    linkage.set_coords(init_pos)
    linkage.set_num_constraints(params)
    try:
        points = 12
        n = linkage.get_rotation_period()
        # Complete revolution with 12 points
        tuple(tuple(i) for i in linkage.step(iterations=points + 1,
                                             dt=n/points))
        # Again with n points, and at least 12 iterations
        n = 96
        factor = int(points / n) + 1
        L = tuple(tuple(i) for i in linkage.step(
            iterations=n * factor, dt=1 / factor))
    except UnbuildableError:
        return -float('inf')
    else:
        # Locus of the Joint 'pin", mast in linkage order
        tip_locus = tuple(x[-1] for x in L)
        # We get the bounding box
        curr_bb = bounding_box(tip_locus)
        # We set the reference bounding box with frame_second as down-left
        # corner and size 2
        ref_bb = (frame_second.y, frame_second.x + 2,
                  frame_second.y + 2, frame_second.x)
        # Our score is the square sum of the edges distances
        return -sum((pos - ref_pos) ** 2
                    for pos, ref_pos in zip(curr_bb, ref_bb))
```
Please not that it is a *minization* problem, with 0 as lower bound.

We need to get the geometric constraints as the optimization parameters.
``constraints = tuple(my_linkage.get_num_constraints())```

With this constraints, score should be around -3. 

Let's start with a candide optimization, the [trial-and-error](https://en.wikipedia.org/wiki/Trial_and_error) method. Here it is a serial test of switches.
```python
# Exhaustive optimization as an example ONLY
score, position, coord = opti.exhaustive_optimization(
    eval_func=fitness_func,
    linkage=my_linkage,
    delta_dim=.1,
    n_results=1,
)[0]

```
Here the problem is simple enough, so that method typical return the maximal theorical value of 0.0.

However, with more complex linkages you need something more robust, and more efficient. Then we will use [particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization). Here are the principles:
* The parameters are the geometric constrints (the dimensions) of the linkage.
* A dimension set (a n-uplet) is called a *particule* or an *agents*. Think of it like a bee.
* The particles move in a n-vectorial space. That is, if we have n geometric constraints, the particules move in a n-D space.
* Together, the particules form the *swarm*.
* Each time they move, their score is evaluated by our fitness function.
* They know their best score, and know the current score of they neighbours.
* Together they will try find the extremum in the space. Here it is a minimum.

it is particularly relevant when the fitness function is not resource-greedy.

Due to incompatibilities, we need a wrapper.
```python
# A simple wrapper
def PSO_fitness_wrapper(constraints, *args):
    """A simple wrapper to make the fitness function compatible."""
    return fitness_func(my_linkage, constraints, *args)


# We reinitialize the linkage (an optimal linkage is not interesting)
my_linkage.set_num_constraints(constraints)
# As we do for initial positions
my_linkage.set_coords(init_pos)


import numpy as np

# Optimization is more efficient with a start space
bounds = (np.zeros(len(constraints)), np.ones(len(constraints)) * 5)

score = opti.particle_swarm_optimization(
    eval_func=PSO_fitness_wrapper,
    linkage=my_linkage,
    bounds=bounds
).swarm.best_cost

```
Here again the result should be 0.0.

So we made something that say it works, let's verify it:

![An optimized four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/pylinkage/examples/images/Kinematic%20Windscreen%20wiper.gif)

With a bit of imagination you have a wonderful windscreen wiper!
