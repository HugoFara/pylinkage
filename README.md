# pylinkage

A linkage builder written in Python. This package is made to create planar linkages and optimize them kinematically thanks to [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization). It is still an early work, so it should receive great changes in the future.

## Usage

As of today, the code is segmented in three parts:
* [geometry.py](https://github.com/HugoFara/pylinkage/blob/main/lib/geometry.py) that module handles geometric primitives, such as circle intersections, distance claculation. It works in Euclidian space only. Aside from ``dist`` and ``sqr_dist`` functions, you might not use it directly.
* [linkage.py](https://github.com/HugoFara/pylinkage/blob/main/lib/linkage.py) this module describes joints and linkages 
  * Due to the geometric approach, joints (instances of ``Joint`` object) are defined without links. 
  * The ``Linkage`` class that will make your code shorter.
* [optimizer.py](https://github.com/HugoFara/pylinkage/blob/main/lib/optimizer.py) proposes three optimizations based on three techniques:
  * The "exhaustive" optimization (``exhaustive_optimization`` function) is a dumb optimization method, consisting or trying sequencially all positions. It is here for demonstration purposes only, and you should not use it if you are looking for an efficient technique.
  * The built-in Particle Swarm Optimizer (PSO). I started with it, so it offers a large set of useful options for linkage optimization. However, it is here for legacy purposes, and is much short than the PySwarms module.
  * PSO using [PySwarms](https://github.com/ljvmiranda921/pyswarms). We provide a wrapper function to PySwarm from ljvmiranda921, that will progressively be extended.

## Requirements

Python 3, numpy for calculation, matplotlib for drawing, and standard libraries. If you do not want to use PySwarms feel free to use the built-in PSO.

## Usage

Documentation coming soon.
