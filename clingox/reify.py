'''
Document me!!!
'''

from typing import Any, Callable, Dict, Generic, List, Mapping, Sequence, Tuple, TypeVar
from dataclasses import dataclass, field

from clingo.backend import HeuristicType, Observer, TruthValue
from clingo.symbol import Function, Number, Symbol

__all__ = ['Reifier']

T = TypeVar('T')
U = TypeVar('U', int, Tuple[int, int])


@dataclass
class _Vertex(Generic[T]):
    '''
    Vertex data to calculate SCCs of a graph.
    '''
    name: T
    visited: int
    index: int = 0
    edges: List[int] = field(default_factory=list)

class _Graph(Generic[T]):
    '''
    Simple class to compute strongly connected components using Tarjan's
    algorithm.
    '''
    _names: Dict[T, int]
    _vertices: List[_Vertex]
    _phase: bool

    def __init__(self):
        self._names = {}
        self._vertices = []
        self._phase = True

    def _visited(self, key_u: int) -> bool:
        return self._vertices[key_u].visited != int(not self._phase)

    def _active(self, key_u: int) -> bool:
        return self._vertices[key_u].visited != int(self._phase)

    def _add_vertex(self, val_u: T) -> int:
        n = len(self._vertices)
        key_u = self._names.setdefault(val_u, n)
        if n == key_u:
            self._vertices.append(_Vertex(val_u, int(not self._phase)))
        return key_u

    def add_edge(self, val_u: T, val_v: T) -> None:
        '''
        Add an edge to the graph.
        '''
        key_u = self._add_vertex(val_u)
        key_v = self._add_vertex(val_v)
        self._vertices[key_u].edges.append(key_v)

    def tarjan(self) -> List[List[T]]:
        '''
        Returns the stringly connected components of the graph.
        '''
        sccs: List[List[T]] = []
        stack = []
        trail = []
        index = 1
        def push(key_u: int):
            nonlocal index
            index += 1
            vtx_u = self._vertices[key_u]
            vtx_u.visited = index
            vtx_u.index = 0
            stack.append(key_u)
            trail.append(key_u)

        for key_u in range(len(self._vertices)):
            if self._visited(key_u):
                continue
            index = 1
            push(key_u)
            while stack:
                key_v = stack[-1]
                vtx_v = self._vertices[key_v]
                len_v = len(vtx_v.edges)
                while vtx_v.index < len_v:
                    key_w = vtx_v.edges[vtx_v.index]
                    vtx_v.index += 1
                    if not self._visited(key_w):
                        push(key_w)
                        break
                else:
                    stack.pop()
                    root = True
                    for key_w in vtx_v.edges:
                        vtx_w = self._vertices[key_w]
                        if self._active(key_w) and vtx_w.visited < vtx_v.visited:
                            root = False
                            vtx_v.visited = vtx_w.visited
                    if root:
                        key_last = None
                        sccs.append([])
                        while key_last != key_v:
                            key_last = trail[-1]
                            vtx_last = self._vertices[key_last]
                            sccs[-1].append(vtx_last.name)
                            vtx_last.visited = int(self._phase)
                            trail.pop()
                        if len(sccs[-1]) == 1:
                            sccs.pop()

        self._phase = not self._phase
        return sccs

@dataclass
class _StepData:
    atom_tuples: Dict[Sequence[int], int] = field(default_factory=dict)
    lit_tuples: Dict[Sequence[int], int] = field(default_factory=dict)
    wlit_tuples: Dict[Sequence[Tuple[int, int]], int] = field(default_factory=dict)
    graph: _Graph = field(default_factory=_Graph)

def _lit(i: Symbol, lit: int) -> Sequence[Symbol]:
    return [i, Number(lit)]

def _wlit(i: Symbol, wlit: Tuple[int, int]) -> Sequence[Symbol]:
    return [i, Number(wlit[0]), Number(wlit[1])]

class Reifier(Observer):
    '''
    Document me!!!
    '''
    _step: int
    # Bug in mypy???
    #_cb: Callable[[Symbol], None]
    _calculate_sccs: bool
    _reify_steps: bool
    _step_data: _StepData

    def __init__(self, cb: Callable[[Symbol], None], calculate_sccs: bool = False):
        self._step = 0
        self._cb = cb
        self._calculate_sccs = calculate_sccs
        self._reify_steps = False
        self._step_data = _StepData()

    def calculate_sccs(self) -> None:
        '''
        Trigger computation of SCCs.

        SCCs can only be computed if the Reifier has been initialized with
        `calculate_sccs=True`, This function is called automatically if
        `reify_steps=True` has been set when initializing the Reifier.
        '''
        for idx, scc in enumerate(self._step_data.graph.tarjan()):
            for atm in scc:
                self._output('scc', [Number(idx), Number(atm)])

    def _add_edges(self, head: Sequence[int], body: Sequence[int]):
        if self._calculate_sccs:
            for u in head:
                for v in body:
                    if v > 0:
                        self._step_data.graph.add_edge(u, v)

    def _output(self, name: str, args: Sequence[Symbol]):
        if self._reify_steps:
            args = list(args) + [Number(self._step)]
        self._cb(Function(name, args))

    def _tuple(self, name: str,
               snmap: Dict[Sequence[U], int],
               elems: Sequence[U],
               afun: Callable[[Symbol, U], Sequence[Symbol]]) -> Symbol:
        s = tuple(sorted(elems))
        n = len(snmap)
        i = Number(snmap.setdefault(s, n))
        if n == i.number:
            self._output(name, [i])
            for atm in s:
                self._output(name, afun(i, atm))
        return i

    def _atom_tuple(self, atoms: Sequence[int]):
        return self._tuple("atom_tuple", self._step_data.atom_tuples, atoms, _lit)

    def _lit_tuple(self, lits: Sequence[int]):
        return self._tuple("literal_tuple", self._step_data.lit_tuples, lits, _lit)

    def _wlit_tuple(self, wlits: Sequence[Tuple[int, int]]):
        return self._tuple("weighted_literal_tuple", self._step_data.wlit_tuples, wlits, _wlit)

    def init_program(self, incremental: bool) -> None:
        if incremental:
            self._output("tag", [Function("incremental")])

    def begin_step(self) -> None:
        pass

    def rule(self, choice: bool, head: Sequence[int], body: Sequence[int]) -> None:
        hn = "choice" if choice else "disjunction"
        hd = Function(hn, [self._atom_tuple(head)])
        bd = Function("normal", [self._lit_tuple(body)])
        self._output("rule", [hd, bd])
        self._add_edges(head, body)

    def weight_rule(self, choice: bool, head: Sequence[int], lower_bound: int,
                    body: Sequence[Tuple[int, int]]) -> None:
        hn = "choice" if choice else "disjunction"
        hd = Function(hn, [self._atom_tuple(head)])
        bd = Function("sum", [self._wlit_tuple(body), Number(lower_bound)])
        self._output("rule", [hd, bd])
        self._add_edges(head, [l for l, w in  body])

    def minimize(self, priority: int, literals: Sequence[Tuple[int, int]]) -> None:
        RuntimeError("impplement me!!!")

    def project(self, atoms: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def output_atom(self, symbol: Symbol, atom: int) -> None:
        self._output("output", [symbol, self._lit_tuple([atom])])

    def output_term(self, symbol: Symbol, condition: Sequence[int]) -> None:
        self._output("output", [symbol, self._lit_tuple(condition)])

    def output_csp(self, symbol: Symbol, value: int,
                   condition: Sequence[int]) -> None:
        self._output("output", [symbol, self._lit_tuple(condition)])

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
        if self._reify_steps:
            self.calculate_sccs()
            self._step += 1
            self._step_data = _StepData()
