'''
This module provides a backedn wrapper to work with symbols instead of integer literals.
'''
from typing import Iterable, Tuple
from itertools import chain
from clingo import HeuristicType, Symbol, Backend, TruthValue

class SymbolicBackend():
    def __init__(self, backend: Backend):
        self.backend: Backend = backend

    def __enter__(self):
        self.backend.__enter__()
        return self

    def __exit__(self, type_, value, traceback):
        return self.backend.__exit__(type_, value, traceback)

    # TODO: are node_u and node_v literals? do they need to be Symbols?
    def add_acyc_edge(self, node_u: int, node_v: int, pos_condition: Iterable[Symbol], neg_condition: Iterable[Symbol]) -> None:
        condition = chain(self._add_symbols_and_return_their_codes(pos_condition), self._add_symbols_and_return_their_negated_codes(neg_condition))
        return self.backend.add_acyc_edge(node_u, node_v, condition)

    # TODO: can this get negative literals?
    def add_assume(self, literals: Iterable[Symbol]) -> None:
        return self.backend.add_assume(self._add_symbols_and_return_their_codes(literals))

    def add_atom(self, symbol: Symbol) -> int:
        return self.backend.add_atom(symbol)

    def add_external(self, symbol: Symbol, value: TruthValue = TruthValue.False_) -> None:
        return self.backend.add_external(self.backend.add_atom(symbol), value)

    def add_heuristic(self, symbol: Symbol, type_: HeuristicType, bias: int, priority: int, pos_condition: Iterable[Symbol], neg_condition: Iterable[Symbol]) -> None:
        atom = self.backend.add_atom(symbol)
        condition = chain(self._add_symbols_and_return_their_codes(pos_condition), self._add_symbols_and_return_their_negated_codes(neg_condition))
        return self.backend.add_heuristic(atom, type_, bias, priority, condition)

    def add_minimize(self, priority: int, pos_literals: Iterable[Tuple[Symbol, int]], neg_literals: Iterable[Tuple[Symbol, int]]) -> None:
        literals = chain(self._add_symbols_and_return_their_codes_with_weights(pos_literals), self._add_symbols_and_return_their_negated_codes_with_weights(neg_literals))
        return self.backend.add_minimize(priority, literals)

    def add_project(self, symbols: Iterable[Symbol]) -> None:
        atoms = (self._add_symbols_and_return_their_codes(symbols))
        return self.backend.add_project(atoms)

    # TODO: can the head also have negative literals?
    def add_rule(self, head: Iterable[Symbol] = (), pos_body: Iterable[Symbol] = (), neg_body: Iterable[Symbol] = (), choice: bool = False) -> None:
        head_ = (self._add_symbols_and_return_their_codes(head))
        body = chain(self._add_symbols_and_return_their_codes(pos_body), self._add_symbols_and_return_their_negated_codes(neg_body))
        return self.backend.add_rule(head_, body, choice)

    def _add_symbols_and_return_their_codes(self, symbols: Iterable[Symbol]):
        return (self.backend.add_atom(symbol) for symbol in symbols)

    def _add_symbols_and_return_their_negated_codes(self, symbols: Iterable[Symbol]):
        return (-x for x in self._add_symbols_and_return_their_codes(symbols))

    def _add_symbols_and_return_their_codes_with_weights(self, symbols: Iterable[Tuple[Symbol, int]]):
        return ((self.backend.add_atom(symbol), w) for (symbol, w) in symbols)

    def _add_symbols_and_return_their_negated_codes_with_weights(self, symbols: Iterable[Tuple[Symbol, int]]):
        return ((-x, w) for (x, w) in self._add_symbols_and_return_their_codes_with_weights(symbols))
