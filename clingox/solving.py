'''
This module provides functions to approximate the cautious consequences of a program
'''

from typing import Tuple, Sequence
from clingo.control import Control
from clingo.configuration import Configuration
from clingo.symbol import Symbol


def approximate_cautions_consequences(ctl: Control) -> Tuple[Sequence[Symbol], Sequence[Symbol]]:
    '''
    Computes an approximation of the caustions consequences of the program represented by ctl.
    Run in polynomial time.

    Parameters
    ----------
    ctl
        A control object with a program. Grounding should be perfomed on this control object
        before calling this function
    Returns
    -------
    A pair of sequences of symbols representing a lower and an upper bound to all the stable
    models of the program. All symbols in the lower bound are cautios consequences of the program.
    For every symbol 'a' not in the upper bound 'not a' is a cautios consequence.
    '''
    assert isinstance(ctl.configuration.solve, Configuration)
    solve_limits = ctl.configuration.solve.solve_limit
    ctl.configuration.solve.solve_limit = 0
    ctl.solve()
    ctl.cleanup()
    lower = []
    upper = []
    for sa in ctl.symbolic_atoms:
        upper.append(sa.symbol)
        if sa.is_fact:
            lower.append(sa.symbol)
    ctl.configuration.solve.solve_limit = solve_limits
    return lower, upper
