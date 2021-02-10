'''
This module provides a backend wrapper to work with symbols instead of integer
literals.
'''
from typing import Iterable, Sequence, Tuple
from itertools import chain
from clingo import HeuristicType, Symbol, Backend, TruthValue


def _add_sign(lit: int, sign: bool):
    '''
    Invert the literal if sign is negative and otherwise leave it untouched.
    '''
    return lit if sign else -lit

class SymbolicBackend():
    '''
    Backend wrapper providing a interface to extend a logic program. It mirrors
    the interface of clingo's Backend, but using Symbols rather than integers
    to represent literals.

    Implements: `ContextManager[SymbolicBackend]`.

    See Also
    --------
    clingo.Backend, clingo.Control.backend

    Notes
    --------
    The `SymbolicBackend` is a context manager and must be used with Python's
    `with` statement.

    Examples
    --------
    The following example shows how to add the rules

        a :- b, not c.
        b.

    to a program:

        >>> import clingo
        >>> from clingox.backends import SymbolicBackend
        >>> ctl = clingo.Control()
        >>> a = clingo.Function("a")
        >>> b = clingo.Function("b")
        >>> c = clingo.Function("c")
        >>> with SymbolicBackend(ctl.backend()) as symbolic_backend:
                symbolic_backend.add_rule([a], [b], [c])
                symbolic_backend.add_rule([b])
        >>> ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
        Answer: a b
        SAT

    The `SymbolicBackend` can also be used in combination with the `Backend`
    that it wraps. In this case, it is the `Backend` that must be used with
    Python's `with` statement:

        >>> import clingo
        >>> from clingox.backends import SymbolicBackend
        >>> ctl = clingo.Control()
        >>> a = clingo.Function("a")
        >>> b = clingo.Function("b")
        >>> c = clingo.Function("c")
        >>> with ctl.backend() as backend:
                symbolic_backend = SymbolicBackend(backend)
                symbolic_backend.add_rule([a], [b], [c])
                atom_b = backend.add_atom(b)
                backend.add_rule([atom_b])
        >>> ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
        Answer: a b
        SAT
    '''
    def __init__(self, backend: Backend):
        self.backend: Backend = backend

    def __enter__(self):
        '''
        Initialize the backend.

        Returns
        -------
        Backend
            Returns the backend itself.

        Notes
        -----
        Must be called before using the backend.
        '''
        self.backend.__enter__()
        return self

    def __exit__(self, type_, value, traceback):
        '''
        Finalize the backend.

        Notes
        -----
        Follows Python's __exit__ conventions. Does not suppress exceptions.
        '''
        return self.backend.__exit__(type_, value, traceback)

    def add_acyc_edge(self, node_u: int, node_v: int, pos_condition: Sequence[Symbol],
                      neg_condition: Sequence[Symbol]) -> None:
        '''
        Add an edge directive to the underlying backend.

        Parameters
        ----------
        node_u : int
            The start node represented as an unsigned integer.
        node_v : int
            The end node represented as an unsigned integer.
        pos_condition : Sequence[Symbol]
            List of program symbols forming positive part of the condition.
        neg_condition : Sequence[Symbol]
            List of program symbols forming negated part of the condition.

        Returns
        -------
        None
        '''
        condition = chain(self._add_lits(pos_condition, True), self._add_lits(neg_condition, False))
        self.backend.add_acyc_edge(node_u, node_v, list(condition))

    def add_assume(self, pos_atoms: Sequence[Symbol] = (), neg_atoms: Sequence[Symbol] = ()) -> None:
        '''
        Add assumptions to the underlying backend.

        Parameters
        ----------
        pos_atoms : Sequence[Symbol]
            Atoms to assume true.
        neg_atoms : Sequence[Symbol]
            Atoms to assume false.

        Returns
        -------
        None
        '''
        literals = chain(self._add_lits(pos_atoms, True), self._add_lits(neg_atoms, False))
        self.backend.add_assume(list(literals))

    def add_atom(self, symbol: Symbol) -> int:
        '''
        Return a fresh program atom or the atom associated with the given
        symbol.

        If the given symbol does not exist in the atom base, it is added first.
        Such atoms will be used in subsequents calls to ground for
        instantiation.

        Parameters
        ----------
        symbol : Optional[Symbol]=None
            The symbol associated with the atom.

        Returns
        -------
        int
            The program atom representing the atom.
        '''
        return self.backend.add_atom(symbol)

    def add_external(self, atom: Symbol, value: TruthValue = TruthValue.False_) -> None:
        '''
        Mark a program atom as external and set its truth value.

        Parameters
        ----------
        atom : Symbol
            The atom to mark as external.
        value : TruthValue=TruthValue.False_
            Optional truth value.

        Returns
        -------
        None

        Notes
        -----
        Can also be used to release an external atom using `TruthValue.Release`.
        '''
        return self.backend.add_external(self.backend.add_atom(atom), value)

    def add_heuristic(self, atom: Symbol, type_: HeuristicType, bias: int, priority: int,
                      pos_condition: Sequence[Symbol], neg_condition: Sequence[Symbol]) -> None:
        '''
        Add a heuristic directive to the underlying backend.

        Parameters
        ----------
        atom : Symbol
            Program atom to heuristically modify.
        type_ : HeuristicType
            The type of modification.
        bias : int
            A signed integer.
        priority : int
            An unsigned integer.
        pos_condition : Sequence[Symbol]
            List of program literals forming the positive part of the
            condition.
        neg_condition : Sequence[Symbol]
            List of program literals forming the negated part of the condition.

        Returns
        -------
        None
        '''
        condition = chain(self._add_lits(pos_condition, True), self._add_lits(neg_condition, False))
        return self.backend.add_heuristic(self.backend.add_atom(atom), type_, bias, priority, list(condition))

    def add_minimize(self, priority: int, pos_literals: Sequence[Tuple[Symbol, int]],
                     neg_literals: Sequence[Tuple[Symbol, int]]) -> None:
        '''
        Add a minimize constraint to the underlying backend.

        Parameters
        ----------
        priority : int
            Integer for the priority.
        pos_literals : Sequence[Tuple[Symbol,int]]
            List of pairs of program symbols and weights forming the positive
            part of the condition.
        neg_literals : Sequence[Tuple[Symbol,int]]
            List of pairs of program symbols and weights forming the negated
            part of the condition.

        Returns
        -------
        None
        '''
        literals = chain(self._add_wlits(pos_literals, True), self._add_wlits(neg_literals, False))
        return self.backend.add_minimize(priority, list(literals))

    def add_project(self, atoms: Sequence[Symbol]) -> None:
        '''
        Add a project statement to the underlying backend.

        Parameters
        ----------
        atoms : Sequence[Symbol]
            List of program symbols to project on.

        Returns
        -------
        None
        '''
        return self.backend.add_project(list(self._add_lits(atoms, True)))

    def add_rule(self, head: Sequence[Symbol] = (), pos_body: Sequence[Symbol] = (), neg_body: Sequence[Symbol] = (),
                 choice: bool = False) -> None:
        '''
        Add a disjuntive or choice rule to the underlying backend.

        Parameters
        ----------
        head : Sequence[Symbol]
            The program atoms forming the rule head.
        pos_body : Sequence[Symbol]=()
            The program symbols forming the positive body of the rule
        neg_body : Sequence[Symbol]=()
            The program symbols forming the negated body of the rule
        choice : bool=False
            Whether to add a disjunctive or choice rule.

        Returns
        -------
        None

        Notes
        -----
        Integrity constraints and normal rules can be added by using an empty or
        singleton head list, respectively.
        '''
        body = chain(self._add_lits(pos_body, True), self._add_lits(neg_body, False))
        return self.backend.add_rule(list(self._add_lits(head, True)), list(body), choice)

    def add_weight_rule(self, head: Sequence[Symbol], lower: int, pos_body: Sequence[Tuple[Symbol, int]],
                        neg_body: Sequence[Tuple[Symbol, int]], choice: bool = False) -> None:
        '''
        Add a disjunctive or choice rule with one weight constraint with a lower
        bound in the body to the underlying backend.

        Parameters
        ----------
        head : Sequence[int]
            The program atoms forming the rule head.
        lower : int
            The lower bound.
        pos_body : Sequence[Tuple[Symbol,int]]
            The pairs of program symbols and weights forming the elements of the
            positive body of the weight constraint.
        neg_body : Sequence[Tuple[Symbol,int]]
            The pairs of program symbols and weights forming the elements of the
            negative body of the weight constraint.
        choice : bool=False
            Whether to add a disjunctive or choice rule.

        Returns
        -------
        None
        '''
        body = chain(self._add_wlits(pos_body, True), self._add_wlits(neg_body, False))
        return self.backend.add_weight_rule(list(self._add_lits(head, True)), lower, list(body), choice)

    def _add_lits(self, atoms: Sequence[Symbol], sign: bool) -> Iterable[int]:
        '''
        Map the given atoms to program literals with the given sign.
        '''
        return (_add_sign(self.backend.add_atom(symbol), sign) for symbol in atoms)

    def _add_wlits(self, weighted_symbols: Sequence[Tuple[Symbol, int]], sign: bool) -> Iterable[Tuple[int, int]]:
        '''
        Map the given weighted atoms to weighted program literals with the
        given sign.
        '''
        return ((_add_sign(self.backend.add_atom(x), sign), w) for (x, w) in weighted_symbols)
