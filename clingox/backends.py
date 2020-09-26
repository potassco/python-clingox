'''
This module provides a backedn wrapper to work with symbols instead of integer literals.
'''
from typing import Iterable, Tuple
from itertools import chain
from clingo import HeuristicType, Symbol, Backend, TruthValue

class SymbolicBackend():
    '''
    Backend wrapper providing a interface to extend a logic program.
    It mirrors the interface of clingo's Backend, but using Symbols rather than integers to represent literals.

    Implements: ContextManager[SymbolicBackend].

    See Also
    --------
    clingo.Backend and Control.backend()

    Notes
    --------
    The `SymbolicBackend` is a context manager and must be used with Python's `with` statement.

    Examples
    --------
    The following example shows how to add the rules
        a :- b, not c.
        b.
    to a program:

        >>> import clingo
        >>> import clingox.backends.SymbolicBackend
        >>> ctl = clingo.Control()
        >>> a = clingo.Function("a")
        >>> b = clingo.Function("b")
        >>> c = clingo.Function("c")
        >>> with SymbolicBackend(ctl.backend()) as backend:
                backend.add_rule([a], [b], [c])
                backend.add_rule([b])
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

    # TODO: are node_u and node_v literals? do they need to be Symbols?
    def add_acyc_edge(self, node_u: int, node_v: int, pos_condition: Iterable[Symbol], neg_condition: Iterable[Symbol]) -> None:
        '''
        Add an edge directive to the underline backend.

        Parameters
        ----------
        node_u : int
            The start node represented as an unsigned integer.
        node_v : int
            The end node represented as an unsigned integer.
        pos_condition : Iterable[Symbol]
            List of program symbols forming positive part of the condition
        neg_condition : Iterable[Symbol]
            List of program symbols forming negated part of the condition
        Returns
        -------
        None
        '''
        condition = chain(self._add_symbols(pos_condition), self._add_negated_symbols(neg_condition))
        return self.backend.add_acyc_edge(node_u, node_v, condition)

    # TODO: can this get negative literals?
    def add_assume(self, literals: Iterable[Symbol]) -> None:
        '''
        Add assumptions to the underline backend.

        :param literals: The list of symbols to assume true.
        :returns: None
        '''
        return self.backend.add_assume(self._add_symbols(literals))

    def add_atom(self, symbol: Symbol) -> int:
        '''
        Return a fresh program atom or the atom associated with the given symbol.

        If the given symbol does not exist in the atom base, it is added first. Such
        atoms will be used in subequents calls to ground for instantiation.

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

    def add_external(self, symbol: Symbol, value: TruthValue = TruthValue.False_) -> None:
        '''
        Mark a program atom as external optionally fixing its truth value.

        Parameters
        ----------
        atom : Symbol
            The program atom associated with symbol to mark as external.
        value : TruthValue=TruthValue.False_
            Optional truth value.

        Returns
        -------
        None

        Notes
        -----
        Can also be used to release an external atom using `TruthValue.Release`.
        '''
        return self.backend.add_external(self.backend.add_atom(symbol), value)

    def add_heuristic(self, symbol: Symbol, type_: HeuristicType, bias: int, priority: int, pos_condition: Iterable[Symbol], neg_condition: Iterable[Symbol]) -> None:
        '''
        Add a heuristic directive to the underline backend.

        Parameters
        ----------
        atom : int
            Program atom to heuristically modify.
        type : HeuristicType
            The type of modification.
        bias : int
            A signed integer.
        priority : int
            An unsigned integer.
        pos_condition : Iterable[Symbol]
            List of program literals forming the positive part of the condition.
        neg_condition : Iterable[Symbol]
            List of program literals forming the negated part of the condition.

        Returns
        -------
        None
    '''
        atom = self.backend.add_atom(symbol)
        condition = chain(self._add_symbols(pos_condition), self._add_negated_symbols(neg_condition))
        return self.backend.add_heuristic(atom, type_, bias, priority, condition)

    def add_minimize(self, priority: int, pos_literals: Iterable[Tuple[Symbol, int]], neg_literals: Iterable[Tuple[Symbol, int]]) -> None:
        '''
        Add a minimize constraint to the underline backend.

        Parameters
        ----------
        priority : int
            Integer for the priority.
        pos_literals : Iterable[Tuple[Symbol,int]]
            List of pairs of program symbols and weights forming the positive part of the condition.
        neg_literals : Iterable[Tuple[Symbol,int]]
            List of pairs of program symbols and weights forming the negated part of the condition.
        Returns
        -------
        None
        '''
        literals = chain(self._add_symbols_weights(pos_literals), self._add_symbols_weights(neg_literals))
        return self.backend.add_minimize(priority, literals)

    def add_project(self, symbols: Iterable[Symbol]) -> None:
        '''
        Add a project statement to the underline backend.

        Parameters
        ----------
        atoms : Iterable[Symbol]
            List of program symbols to project on.

        Returns
        -------
        None
        '''
        atoms = (self._add_symbols(symbols))
        return self.backend.add_project(atoms)

    # TODO: can the head also have negative literals?
    def add_rule(self, head: Iterable[Symbol] = (), pos_body: Iterable[Symbol] = (), neg_body: Iterable[Symbol] = (), choice: bool = False) -> None:
        '''
        Add a disjuntive or choice rule to the underline backend.

        Parameters
        ----------
        head : Iterable[Symbol]
            The program atoms forming the rule head.
        pos_body : Iterable[Symbol]=()
            The program symbols forming the positive body of the rule
        neg_body : Iterable[Symbol]=()
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
        head_ = (self._add_symbols(head))
        body = chain(self._add_symbols(pos_body), self._add_negated_symbols(neg_body))
        return self.backend.add_rule(head_, body, choice)

    def add_weight_rule(self, head: Iterable[int], lower: int, pos_body: Iterable[Tuple[Symbol, int]], neg_body: Iterable[Tuple[Symbol, int]], choice: bool=False) -> None:
        '''
        Add a disjuntive or choice rule with one weight constraint with a lower bound
        in the body to the underline backend.

        Parameters
        ----------
        head : Iterable[int]
            The program atoms forming the rule head.
        lower : int
            The lower bound.
        pos_body : Iterable[Tuple[Symbol,int]]
            The pairs of program symbols and weights forming the elements of the
            positive body of the weight constraint.
        neg_body : Iterable[Tuple[Symbol,int]]
            The pairs of program symbols and weights forming the elements of the
            negative body of the weight constraint.
        choice : bool=False
            Whether to add a disjunctive or choice rule.

        Returns
        -------
        None
        '''
        head_ = (self._add_symbols(head))
        body = chain(self._add_symbols_weights(pos_body), self._add_symbols_weights(neg_body))
        return self.backend.add_weight_rule(head_, lower, body, choice)

    def _add_symbols(self, symbols: Iterable[Symbol]) -> Iterable[int]:
        '''
        Return a fresh program atom or the atom associated with each symbol in `symbols`.

        If the given symbol does not exist in the atom base, it is added first. Such
        atoms will be used in subequents calls to ground for instantiation.

        Parameters
        ----------
        symbols : Iterable[Symbol]
            The symbols associated with the atoms.

        Returns
        -------
        Iterable[int]
            The program atoms representing the atoms.
        '''
        return (self.backend.add_atom(symbol) for symbol in symbols)

    def _add_negated_symbols(self, symbols: Iterable[Symbol]):
        '''
        Return a fresh program negated literal or the negation of the atom associated with each symbol in `symbols`.

        If the given symbol does not exist in the atom base, it is added first. Such
        atoms will be used in subequents calls to ground for instantiation.

        Parameters
        ----------
        symbols : Iterable[Symbol]
            The symbols associated with the atoms.

        Returns
        -------
        Iterable[int]
            The program literals representing the negated atoms.
        '''
        return (-x for x in self._add_symbols(symbols))

    def _add_symbols_weights(self, weighted_symbols: Iterable[Tuple[Symbol, int]]) -> Iterable[Tuple[int, int]]:
        '''
        Return a pair composing of an atom and its associated weith for each pair in `weighted_symbols`.
        The first component is a fresh program atom or the atom associated with a given symbol for each symbol in the weighted_symbol.
        The second component is the associated weight unchanged.

        If the given symbol does not exist in the atom base, it is added first. Such
        atoms will be used in subequents calls to ground for instantiation.

        Parameters
        ----------
        symbols : Iterable[Tuple[Symbol,int]]
            The symbols associated with the atoms.

        Returns
        -------
        Iterable[Tuple[int,int]]
            The pairs representing the weighted atoms.
        '''
        return ((self.backend.add_atom(symbol), w) for (symbol, w) in weighted_symbols)

    def _add_symbols_weights(self, weighted_symbols: Iterable[Tuple[Symbol, int]]):
        '''
        Return a pair composing of an negated literal and its associated weith for each pair in `weighted_symbols`.
        The first component is a fresh negated literal or the negation of the atom associated with the symbol in the weighted_symbol.
        The second component is the associated weight unchanged.

        If the given symbol does not exist in the atom base, it is added first. Such
        atoms will be used in subequents calls to ground for instantiation.

        Parameters
        ----------
        symbols : Iterable[Tuple[Symbol,int]]
            The symbols associated with the atoms.

        Returns
        -------
        Iterable[Tuple[int,int]]
            The pairs representing the weighted atoms.
        '''
        return ((-x, w) for (x, w) in self._add_symbols_weights(weighted_symbols))
