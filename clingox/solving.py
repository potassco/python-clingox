"""
This module provides functions to approximate the cautious consequences of a
program
"""

from typing import Optional, Sequence, Tuple

from clingo.configuration import Configuration
from clingo.control import Control
from clingo.symbol import Symbol


def approximate(ctl: Control) -> Optional[Tuple[Sequence[Symbol], Sequence[Symbol]]]:
    """
    Approximate the stable models of a program.

    Parameters
    ----------
    ctl
        A control object with a program. Grounding should be performed on this
        control object before calling this function.

    Returns
    -------
    Returns `None` if the problem is determined unsatisfiable. Otherwise,
    returns an approximation of the stable models of the program in form of a
    pair of sequences of symbols. Atoms contained in the first sequence are
    true and atoms not contained in the second sequence are false in all stable
    models.

    Notes
    -----
    Runs in polynomial time. An approximation might be returned even if the
    problem is unsatisfiable.
    """
    # solve with a limit of 0 conflicts to propagate direct consequences
    assert isinstance(ctl.configuration.solve, Configuration)
    solve_limit = ctl.configuration.solve.solve_limit
    ctl.configuration.solve.solve_limit = 0
    ctl.solve()
    ctl.configuration.solve.solve_limit = solve_limit
    ctl.cleanup()

    # check if the problem is conflicting
    if ctl.is_conflicting:
        return None

    # return approximation
    lower = []
    upper = []
    for sa in ctl.symbolic_atoms:
        upper.append(sa.symbol)
        if sa.is_fact:
            lower.append(sa.symbol)
    return lower, upper
