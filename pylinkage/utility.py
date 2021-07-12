"""
The utility module provides various useful functions.

Created on Mon Jul 12 00:00:01 2021.

@author: HugoFara
"""
from pylinkage.exceptions import UnbuildableError

def kinematic_default_test(func, error_penalty):
    """
    Standard run for any linkage before a complete fitness evaluation.

    This decorator make a kinematic simulation, before passing the loci to the
    decorated function.

    Parameters
    ----------
        func : callable
            Fitness function to be decorated.
        error_penalty : float
            Penalty value for unbuildable linkage. Common values include
            float('inf') and 0.
    """
    def wrapper(linkage, params, init_pos=None):
        """
        Decorated function.

        Parameters
        ----------
        linkage : pylinkage.linkage.Linkage
            The linkage to optimize.
        params : tuple[float]
            Geometric constraints to pass to linkage.set_num_constraints.
        init_pos : tuple[tuple[float]]
            List of initial positions for the joints. If None it will be
            redifined at each succefull iteration. The default is None.
        """
        if init_pos is not None:
            linkage.set_coords(init_pos)
        linkage.set_num_constraints(params)
        try:
            points = 12
            n = linkage.get_rotation_period()
            # Complete revolution with 12 points
            tuple(
                tuple(i) for i in linkage.step(
                    iterations=points + 1, dt=n / points
                )
            )
            # Again with n points, and at least 12 iterations
            n = 96
            factor = int(points / n) + 1
            loci = tuple(
                tuple(i) for i in linkage.step(
                    iterations=n * factor, dt=1 / factor
                )
            )
        except UnbuildableError:
            return error_penalty
        else:
            # We redefine intial position if requested
            if init_pos is None:
                init_pos = linkage.get_coords()
            return func(
                linkage=linkage, params=params, init_pos=init_pos, loci=loci
            )
    return wrapper

def kinematic_maximization(func):
    """
    Standard run for any linkage before a complete fitness evaluation.

    This decorator make a kinematic simulation, before passing the loci to the
    decorated function. In case of error, the penalty value is -float('inf')

    Parameters
    ----------
        func : callable
            Fitness function to be decorated.
    """
    return kinematic_default_test(func, -float('inf'))


def kinematic_minimization(func):
    """
    Standard run for any linkage before a complete fitness evaluation.

    This decorator make a kinematic simulation, before passing the loci to the
    decorated function. In case of error, the penalty value is float('inf')

    Parameters
    ----------
        func : callable
            Fitness function to be decorated.
    """
    return kinematic_default_test(func, float('inf'))