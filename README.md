# Pylinkage

[![PyPI version fury.io](https://badge.fury.io/py/pylinkage.svg)](https://pypi.python.org/pypi/pylinkage/)
[![Downloads](https://static.pepy.tech/personalized-badge/pylinkage?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/pylinkage)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg )](https://raw.githubusercontent.com/HugoFara/pylinkage/main/LICENSE.rst)

Pylinkage is a Python linkage builder and optimizer. 
You can create planar linkages and optimize them with a [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization). 
It is still in beta, so don't hesitate to post pull request or issues for features you would like to see!.

## Installation

### Using pip

This package is in the PyPi as [pylinkage](https://pypi.org/project/pylinkage/), and can be installed using:

```shell
pip install pylinkage
```

It is the recommended way of downloading it.

### Setting up Virtual Environment

We provide an [environment.yml](https://github.com/HugoFara/leggedsnake/blob/main/environment.yml) file for conda. 
Use ``conda env update --file environment.yml --name pylinkage-env`` to install the requirements in a separate environment.

## Short demo

Let's start with a short demo of the package's capabilities.

It is highly Pythonic and emphasizes efficient code. Here is the definition of a **four-bar linkage**.
```python
import pylinkage as pl

# Main motor
crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1)
# Close the loop
pin = pl.Revolute(
    3, 2, joint0=crank, joint1=(3, 0), 
    distance0=3, distance1=1
)

# Create the linkage
my_linkage = pl.Linkage(joints=(crank, pin))

# Show the results
pl.show_linkage(my_linkage)
```

![A four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/images/Kinematic%20My%20four-bar%20linkage.gif)

Cool, isn't it? But the package doesn't stop here as it provides a library **to achieve any movement**. 

Let's say that you want the ``pin`` joint to stay in top-right corner, with an amplitude of [0, 90] exactly. 
You can solve it by yourself, you can ask the code to do it for you.

Define evaluation function called ``fitness_func``, that returns a good score to the linkage 
when the ``pin`` joint stays in the first quadrant, and 0 otherwise. 

Then just run the following code:

```python
# Let's set some bounds to keep the dimensions reasonable
bounds = pl.generate_bounds(my_linkage.get_num_constraints())

# Now comes the "magic" function that solve of issue
score, position, coord = pl.particle_swarm_optimization(
    eval_func=fitness_func,
    linkage=my_linkage,
    bounds=bounds,
    order_relation=min,
)[0]

# Let's use the result in our linkage!
my_linkage.set_num_constraints(constraints) # Dimensions
my_linkage.set_coords(init_pos) # Initial position

pl.show_linkage(my_linkage)
```

![An optimized four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/images/Kinematic%20Windscreen%20wiper.gif)

Tadaaa!
We defined a mechanism, solved an issue and viewed the result in a few lines of code!
And remember, you can define any objective function, so let's give it a try!

## Complete example

Let's start with a crank-rocker [four-bar linkage](https://en.wikipedia.org/wiki/Four-bar_linkage), as a classic mechanism. 

### Joints definition

First, we define at least one crank because we want a kinematic simulation.

```python
crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1)
```

Here we are actually doing the following:

* ``0, 1``: x and y initial coordinates of the **tail** of the crank link.
* ``joint0``: the position of the parent Joint to link with, here it is a fixed point in space. 
The pin will be created on the position of the parent, which is the head of the crank link.
* ``angle``: the crank will rotate with this angle, in radians, at each iteration.
* ``distance``: distance to keep constant between crank link tail and head.

Now we add a pin joint to close the kinematic loop.

```python
pin = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1)
```

In human language, here is what is happening:

* ``joint0``, ``joint1``: first and second ``Joint``s you want to link to, the order is not important.
* ``distance0``, ``distance1``: distance to keep constant between this joint and his two parents.

And here comes the trick:
Why do we specify initial coordinates ``3, 2``? They even seem incompatible with distance to parents/parents' positions! 
  * This explanation is simple: mathematically a pin joint the intersection of two circles. 
The intersection is often two points. 
To choose the starting point, we calculate both intersection (when possible), 
then we keep the intersection closer to the previous position as the solution. 


Wait! A linkage with a single motor and only one pin joint? That doesn't make sense!
: Behind the curtain, many joints are created on the fly. 
When you define a ``Crank`` joint, it creates a motor **and** a pin joint on the crank's link head. 
For a ``Revolute`` joint, it creates **3 pin joints**: one on each of its parents' positions, and one its position, 
which forms a deformable triangle. 
This is why pylinkage is so short to write.

### Linkage definition and simulation

Once your linkage is finished, you can either use the ``reload`` method of each ``Joint`` in a loop, 
or put everything in a ``Linkage`` that will handle this can of thinks for you. 

Linkage definition is simple:

```python
my_linkage = pl.Linkage(joints=(crank, pin))
```

That's all!

Now we want to simulate it and to get the locus of ``pin``. Just use the ``step`` method of ``Linkage`` to make a complete rotation.

```python
locus = my_linkage.step()
```
You can also specify the number of steps with the ``iteration`` argument, or subdivisions of each iteration with``dt``.

Let's recap.

```python
import pylinkage as pl

# Main motor
crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1)
# Close the loop
pin = pl.Revolute(
    3, 2, joint0=crank, joint1=(3, 0), 
    distance0=3, distance1=1
)

my_linkage = pl.Linkage(joints=(crank, pin))

locus = my_linkage.step()
```

### Visualization

First thing first, you made a cool linkage, but only you know what it is.
Let's add friendly names to joints, so the communication is simplified.

```python
crank.name = "B"
pin.name = "C"
# Linkage can also have names
my_linkage.name = "Four-bar linkage"
```

Then you can view your linkage!

```python
pl.show_linkage(my_linkage)
```

![A four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/images/Kinematic%20My%20four-bar%20linkage.gif)

Last recap, rearranging names:

```python
# Main motor
crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1, name="B")
# Close the loop
pin = pl.Revolute(
    3, 2, joint0=crank, joint1=(3, 0), 
    distance0=3, distance1=1, name="C"
)

# Linkage definition
my_linkage = pl.Linkage(
    joints=(crank, pin),
    order=(crank, pin),
    name="My four-bar linkage"
)

# Visualization
pl.show_linkage(my_linkage)
```

### Optimization

Now, we want automatic optimization of our linkage, using a certain criterion. 
Let's find a four-bar linkage that makes a quarter of a circle. 
It is a common problem if you want to build a [windshield wiper](https://en.wikipedia.org/wiki/Windscreen_wiper) for instance.

Our objective function, often called the fitness function, is the following:

```python
# We save the initial positions because we don't want a completely different movement
init_pos = my_linkage.get_coords()

@pl.kinematic_minimizastion
def fitness_func(loci, **_kwargs):
    """
    Return how fit the locus is to describe a quarter of circle.

    It is a minimization problem and the theoretic best score is 0.
    """
    # Locus of the Joint 'pin", mast in linkage order
    tip_locus = tuple(x[-1] for x in loci)
    # We get the bounding box
    curr_bb = bounding_box(tip_locus)
    # We set the reference bounding box, in order (min_y, max_x, max_y, min_x)
    ref_bb = (0, 5, 3, 0)
    # Our score is the square sum of the edge distances
    return sum((pos - ref_pos) ** 2 for pos, ref_pos in zip(curr_bb, ref_bb))
```

Please note that it is a *minimization* problem, with 0 as lower bound. 
On the first line, you notice a decorator; which plays a crucial role:

* The decorator arguments are (linkage, constraints), it can also receive ``init_pos``
* It sets the linkage with the constraints.
* Then it verifies if the linkage can do a complete crank turn.
  * If it can, pass the arguments and the resulting loci (path of joints) to the decorated function.
  * If not, return the penalty. In a minimization problem the penalty will be ``float('inf')``.
* The decorated function should return the score of this linkage.  

With this constraint, the best theoretic score is 0.0. 

Let's start with a candide optimization, the [trial-and-error](https://en.wikipedia.org/wiki/Trial_and_error) method. 
Here it is a serial test of switches.

```python
# Exhaustive optimization as an example ONLY
score, position, coord = pl.trials_and_errors_optimization(
    eval_func=fitness_func,
    linkage=my_linkage,
    divisions=25,
    n_results=1,
    order_relation=min,
)[0]
```

Here the problem is simple enough, so that method takes only a few seconds and returns 0.05.

However, with more complex linkages, you need something more robust and more efficient. 
Then we will use [particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization). 
Here are the principles:

* The parameters are the geometric constraints (the dimensions) of the linkage.
* A dimension set (an n-uplet) is called a *particle* or an *agent*. Think of it like a bee.
* The particles move in an n-vectorial space. That is, if we have n geometric constraints, the particles move in an n-D space.
* Together, the particles form the *swarm*.
* Each time they move, their score is evaluated by our fitness function.
* They know their best score, and know the current score of their neighbor.
* Together they will try to find the extreme in the space. Here it is a minimum.

It is particularly relevant when the fitness function is not resource-greedy.

```python
# We reset the linkage (an optimal linkage is not interesting)
my_linkage.set_num_constraints(constraints)
# As we do for initial positions
my_linkage.set_coords(init_pos)

# Optimization is more efficient with a start space
bounds = pl.generate_bounds(my_linkage.get_num_constraints())

score, position, coord = pl.particle_swarm_optimization(
    eval_func=fitness_func,
    linkage=my_linkage,
    bounds=bounds,
    order_relation=min,
)[0]
```
Here the result can vary, but it is rarely above 0.2.

So we made something that says it works, let's verify it:

![An optimized four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/images/Kinematic%20Windscreen%20wiper.gif)

With a bit of imagination, you have a wonderful windshield wiper!


## Code with PyLinkage 

The idea of pylinkage is "one-linkage-one-file".
The idea is that you reuse the same structure for linkage declaration,
in order to make things easy to follow.
Of course, you can adapt the project to your needs, but I recommend the following approach:

1. Declare your joints (any ``Joint`` defined in the ``pylinkage/joints`` package).
2. Arrange joints together in a ``Linkage`` (see the ``pylinkage/linkage`` package).
3. (optional) Manually simulate your linkage.
As we use kinematic planar linkages, constraints are solved as 2D geometry (``pylinkage/geometry``).
4. Optimize your ``Linkage`` with ``pylinkage/optimization``.
5. View the result with ``pylinkage/visualizar``.

## Requirements

Python 3, numpy for calculation, matplotlib for drawing, and standard libraries. 
You will also need PySwarms for the Particle Swarm Optimization.

## Contributing

**Pylinkage is open to contribution**. 
I may consider any pull request, but I ask you to respect the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
and follow the guidelines as defined in [CONTRIBUTING.md](CONTRIBUTING.md). 
