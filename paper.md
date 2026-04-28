---
title: 'pylinkage: Co-optimization of Linkage Topology and Dimensions in Python'
tags:
  - Python
  - mechanical engineering
  - linkage
  - kinematic synthesis
  - particle swarm optimization
  - metaheuristics
  - kinematic analysis
authors:
  - name: Hugo Farajallah
    orcid: 0009-0001-0769-4434
    affiliation: 1
    corresponding: true
affiliations:
  - name: Independent Researcher, France
    index: 1
date: 27 April 2026
bibliography: paper.bib
---

# Summary

Linkage synthesis is an engineering field aiming at creating a linkage
that answers a set of constraints, for instance a target path. Its classical formulation often comes from Burmester's theory [@burmester1888lehrbuch], which
approximates four-bar linkages going through up to five points.
Freudenstein's equation [@freudenstein1954analytical] then gave an exact
analytical solution to the function-generation problem. Both methods use
a topology chosen in advance and solve the geometric problem on top of
it. Modern approaches gave a more computational solution to the same
problem, with metaheuristics such as particle swarms and differential
evolution used extensively to choose dimensions for any given topology.
The most prominent example is Jansen's walking linkage
[@jansen2007strandbeest]. While the dimensional problem can now be
solved computationally, topology synthesis itself remains the largest
open challenge in the field.

`pylinkage` is a Python library aimed at designing, simulating, and
optimizing planar linkages. It provides both classical kinematic
synthesis and modern solvers based on metaheuristics, to give
researchers an end-to-end pipeline. Its main distinguishing choice is
to treat dimensions and topology as a single co-optimization problem.
It bases linkages on hypergraphs with Assur group decomposition
[@assur1952investigation], which gives a formal topological object,
manipulated in structures called *ensembles* rather than as single
Python definitions.

# Statement of need

When this project started, Python tooling for linkages was limited.
`Pyslvs` [@yuan2022pyslvs] is GUI-first and not designed for use inside
a research pipeline. It was mostly meant to design and optimize a single
topology, in the same fashion as historical packages. `mechanism`
[@morris2024mechanism], another library, provides complete kinematic
analysis of a given mechanism — including cams and gears — yet does not
provide features for synthesis or optimization. Some commercial tools
exist (SAM, MechDesigner, LinkageDesigner) but they require a paid
licence and are not programmable from a script. The research landscape
itself is fragmented, with each lab bringing its own implementation.

`pylinkage` aims at serving three audiences:

- **Researchers** working on mechanical design and evolutionary
  robotics, who need a fast backend for optimization.
- **Engineering educators and students**, who can make use of the
  shipped implementations of Burmester's theory, Freudenstein's
  equation, and Grashof classification.
- **Mechanical engineers** who need a full pipeline for the mechanism
  and advanced features such as sensitivity analysis.

To the best of our knowledge, `pylinkage` is the only library that
treats linkage topology as an optimization target and provides the full
pipeline from design to export to 3D software, with co-optimization in
a single open-source package.

# State of the field

Linkage synthesis becomes harder with the number of links. Four-bar
linkages have been extensively studied and complete atlases exist
[@hrones1951analysis; @mccarthy2011geometric; @nobari2022links].

Five-bar and higher-order linkages remain an open challenge. Modern
approaches use machine learning to generate higher-order topologies in
a reasonable time, at the cost of higher training cost
[@nobari2022links]. Classical analytical approaches do not scale.
Design is mostly manual, where an engineer starts with an intuition of
a topology and optimizes its dimensions, as was done by Ghassaei in her variant of the Jansen linkage [@ghassaei2011crankbased], and by the TrotBot family [@diywalkers].

# Software design

`pylinkage` is built around two axes.

The first axis is a layered set of abstractions. A linkage is defined
as a hypergraph, from which we can derive an Assur group decomposition.
This is then used to define the primitives such as circle–circle and
circle–line intersections during simulation. These layers make a very
fast implementation, pure and `numba`-compatible [@lam2015numba], while
allowing extension to a variety of systems and even symbolic
optimization through SymPy [@meurer2017sympy]. The goal of this
abstraction is to decouple the topology problem from the dimensional
problem, so that we can elegantly find solutions to the optimization
problem because of the design.

The second axis is vectorized computation over ensembles of mechanisms.
We operate over arrays of mechanisms by default rather than a single
mechanism at a time. It means that it is fast enough to define and
optimize a mechanism on a single consumer laptop in a few minutes,
using either pymoo [@blank2020pymoo] or PySwarms
[@miranda2018pyswarms] under the hood.

Finally, an important aspect of the package is its adaptability and
flexibility across a number of fields. Most dependencies are optional —
the only ones required are NumPy and `tqdm`. All the others can be
installed on demand, depending on the user's needs. A full installation
is also supported using `pylinkage[full]`.

# Research impact statement

`pylinkage` has been under public development since 2021, with over
sixteen tagged releases and more than 400 commits. The test suite
contains over 900 tests with approximately 90% line coverage. In its
early stage (pre-0.1.0), it was developed for the undergraduate thesis
of the author, and has been developed since then as a public library.
Recent versions introduced a broader research scope and Jupyter
tutorial notebooks covering each major feature.

Several packages use `pylinkage`: `Acinonyx` [@bastidas2025acinonyx] is
an independent open-source project that uses `pylinkage` for trajectory
simulation inside a browser-based UI for multi-link path synthesis.
The author's companion projects `leggedsnake`
[@farajallah2023leggedsnake], which uses physics-based simulation and
genetic algorithms to optimize dynamic walking mechanisms, and
`pylinkage-editor`, a browser-based visual front-end, provide
additional validation of the package's extensibility.

# AI usage disclosure

Generative AI tools were used during development of `pylinkage` from
version 0.7.0 onward for all aspects of code writing, specifically
Claude Opus 4, 4.5, 4.6, and 4.7. Claude Opus 4.7 was also used for
copy-editing of this paper. All AI-generated content was reviewed,
edited, and validated by the human author. Core design decisions —
including the hypergraph-based architecture, the ensemble-first API,
the layering of the solver and abstract modules, and the choice of
synthesis and optimization methods — were made by the author. The
author is solely responsible for the correctness and licensing of all
submitted materials.

# Acknowledgements

The author thanks the maintainers of `numba`, `SymPy`, `pymoo`,
`PySwarms`, and `matplotlib`, on which `pylinkage` builds.
Special thanks to Theo Jansen, Sylvain Liepchitz and Olivier Moreau,
that inspired the idea for this library back in 2016.

# References