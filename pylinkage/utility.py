"""
The utility module provides various useful functions.

Created on Mon Jul 12 00:00:01 2021.

@author: HugoFara
"""
from pylinkage.interface.exceptions import UnbuildableError


def kinematic_default_test(func, error_penalty):
    """Standard run for any linkage before a complete fitness evaluation.
    
    This decorator makes a kinematic simulation, before passing the loci to the
    decorated function.

    :param func: Fitness function to be decorated.
    :type func: Callable
    :param error_penalty: Penalty value for unbuildable linkage. Common values include
            float('inf') and 0.
    :type error_penalty: float

    
    """
    def wrapper(linkage, params, init_pos=None):
        """Decorated function.

        :param linkage: The linkage to optimize.
        :type linkage: pylinkage.linkage.Linkage
        :param params: Geometric constraints to pass to linkage.set_num_constraints.
        :type params: tuple[float]
        :param init_pos: List of initial positions for the joints. If None it will be
            redefined at each successful iteration. (Default value = None)
        :type init_pos: tuple[tuple[float]]

        
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
            # We redefine the initial position if requested
            if init_pos is None:
                init_pos = linkage.get_coords()
            return func(
                linkage=linkage, params=params, init_pos=init_pos, loci=loci
            )
    return wrapper


def kinematic_maximization(func):
    """Standard run for any linkage before a complete fitness evaluation.
    
    This decorator makes a kinematic simulation, before passing the loci to the
    decorated function. In case of error, the penalty value is -float('inf')

    :param func: Fitness function to be decorated.
    :type func: Callable

    
    """
    return kinematic_default_test(func, -float('inf'))


def kinematic_minimization(func):
    """Standard run for any linkage before a complete fitness evaluation.
    
    This decorator makes a kinematic simulation, before passing the loci to the
    decorated function. In case of error, the penalty value is float('inf')

    :param func: Fitness function to be decorated.
    :type func: Callable

    
    """
    return kinematic_default_test(func, float('inf'))
