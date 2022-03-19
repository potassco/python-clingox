'''
Document me!!!
'''

from typing import Sequence, Tuple

from clingo.backend import HeuristicType, Observer, TruthValue
from clingo.symbol import Function, Number, Symbol

__all__ = ['Reifier']

class _StepData:
    def __init__(self):
        self.atom_tuples = {}
        self.lit_tuples = {}
        self.wlit_tuples = {}


class Reifier(Observer):
    '''
    Document me!!!
    '''

    def __init__(self, cb):
        self._cb = cb
        self._calculate_sccs = False
        self._reify_steps = False
        self._step_data = _StepData()

    def _add_edges(self, head, body):
        if self._calculate_sccs:
            raise RuntimeError("impplement me!!!")

    def _output(self, name, args):
        self._cb(Function(name, args))

    def _tuple(self, name, snmap, elems):
        # TODO: refactor to account for weighted literal tuples too
        s = tuple(sorted(elems))
        n = len(snmap)
        i = Number(snmap.setdefault(s, n))
        if n == i.number:
            for atm in s:
                self._cb(Function(name, [i, Number(atm)]))
        return i

    def _atom_tuple(self, atoms):
        return self._tuple("atom_tuple", self._step_data.atom_tuples, atoms)

    def _lit_tuple(self, atoms):
        return self._tuple("lit_tuple", self._step_data.lit_tuples, atoms)

    def init_program(self, incremental: bool) -> None:
        RuntimeError("impplement me!!!")

    def begin_step(self) -> None:
        RuntimeError("impplement me!!!")

    def rule(self, choice: bool, head: Sequence[int], body: Sequence[int]) -> None:
        hn = "choice" if choice else "disjunction"
        hd = Function(hn, [self._atom_tuple(head)])
        bd = Function("normal", self._lit_tuple(body))
        self._output("rule", [hd, bd])
        self._add_edges(head, body)

    def weight_rule(self, choice: bool, head: Sequence[int], lower_bound: int,
                    body: Sequence[Tuple[int,int]]) -> None:
        RuntimeError("impplement me!!!")

    def minimize(self, priority: int, literals: Sequence[Tuple[int,int]]) -> None:
        RuntimeError("impplement me!!!")

    def project(self, atoms: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def output_atom(self, symbol: Symbol, atom: int) -> None:
        RuntimeError("impplement me!!!")

    def output_term(self, symbol: Symbol, condition: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def output_csp(self, symbol: Symbol, value: int,
                   condition: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def external(self, atom: int, value: TruthValue) -> None:
        RuntimeError("impplement me!!!")

    def assume(self, literals: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def heuristic(self, atom: int, type_: HeuristicType, bias: int,
                  priority: int, condition: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def acyc_edge(self, node_u: int, node_v: int,
                  condition: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def theory_term_number(self, term_id: int, number: int) -> None:
        RuntimeError("impplement me!!!")

    def theory_term_string(self, term_id: int, name: str) -> None:
        RuntimeError("impplement me!!!")

    def theory_term_compound(self, term_id: int, name_id_or_type: int,
                             arguments: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def theory_element(self, element_id: int, terms: Sequence[int],
                       condition: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def theory_atom(self, atom_id_or_zero: int, term_id: int,
                    elements: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def theory_atom_with_guard(self, atom_id_or_zero: int, term_id: int,
                               elements: Sequence[int], operator_id: int,
                               right_hand_side_id: int) -> None:
        RuntimeError("impplement me!!!")

    def end_step(self) -> None:
        RuntimeError("impplement me!!!")

