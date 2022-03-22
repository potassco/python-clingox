'''
Document me!!!
'''

from typing import Callable, Dict, Generic, List, Sequence, Tuple, TypeVar
from dataclasses import dataclass, field
import clingo
from clingo.backend import HeuristicType, Observer, TruthValue
from clingo.symbol import Function, Number, Symbol, String

__all__ = ['Reifier','get_theory_symbols']

T = TypeVar('T') #pylint:disable=invalid-name
U = TypeVar('U', int, Tuple[int, int]) #pylint:disable=invalid-name


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
    theory_tuples: Dict[Sequence[int], int] = field(default_factory=dict)
    theory_element_tuples: Dict[Sequence[int], int] = field(default_factory=dict)
    graph: _Graph = field(default_factory=_Graph)

def _theory(i: Symbol, pos: int, lit: int) -> Sequence[Symbol]:
    return [i, Number(pos), Number(lit)]

def _lit(i: Symbol, lit: int) -> Sequence[Symbol]:
    return [i, Number(lit)]

def _wlit(i: Symbol, wlit: Tuple[int, int]) -> Sequence[Symbol]:
    return [i, Number(wlit[0]), Number(wlit[1])]

class Reifier(Observer):
    '''
    Document me!!!
    '''
    #pylint:disable=too-many-public-methods
    _step: int
    # Bug in mypy???
    #_cb: Callable[[Symbol], None]
    _calculate_sccs: bool
    _reify_steps: bool
    _step_data: _StepData

    def __init__(self, cb: Callable[[Symbol], None], calculate_sccs: bool = False, reify_steps: bool = False):
        self._step = 0
        self._cb = cb
        self._calculate_sccs = calculate_sccs
        self._reify_steps = reify_steps
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
               afun: Callable,
               ordered: bool = False) -> Symbol:
        ident = tuple(elems) if ordered else tuple(sorted(set(elems)))
        n = len(snmap)
        i = Number(snmap.setdefault(ident, n))
        if n == i.number:
            self._output(name, [i])
            for idx, atm in enumerate(elems):
                if ordered:
                    self._output(name, afun(i, idx, atm))
                else:
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
            self._cb(Function("tag", [Function("incremental")]))

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
        self._output("minimize", [Number(priority), self._wlit_tuple(literals)])


    def project(self, atoms: Sequence[int]) -> None:
        self._output("project", [Number(atoms[0])])

    def output_atom(self, symbol: Symbol, atom: int) -> None:
        if atom == 0:
            self._output("output", [symbol, Number(0)])
        else:
            self._output("output", [symbol, self._lit_tuple([atom])])

    def output_term(self, symbol: Symbol, condition: Sequence[int]) -> None:
        self._output("output", [symbol, self._lit_tuple(condition)])

    def output_csp(self, symbol: Symbol, value: int,
                   condition: Sequence[int]) -> None:
        self._output("output", [symbol, self._lit_tuple(condition)])

    def external(self, atom: int, value: TruthValue) -> None:
        n = 'true' if value == TruthValue.True_ else 'false' if TruthValue.False_ else 'free'
        self._output("external", [self._lit_tuple([atom]), Function(n,[])])
        

    def assume(self, literals: Sequence[int]) -> None:
        self._output("assume", [self._lit_tuple(literals)])

    def heuristic(self, atom: int, type_: HeuristicType, bias: int,
                  priority: int, condition: Sequence[int]) -> None:
        type_name = str(type_).replace('HeuristicType.', '').lower().strip('_')
        condition_lit = self._lit_tuple(condition)
        self._output("heuristic", [Number(atom), 
                                    Function(type_name), 
                                    Number(bias),
                                    Number(priority),
                                    condition_lit])
        

    def acyc_edge(self, node_u: int, node_v: int,
                  condition: Sequence[int]) -> None:
        RuntimeError("impplement me!!!")

    def theory_term_number(self, term_id: int, number: int) -> None:
        self._output("theory_number", [Number(term_id),Number(number)])

    def theory_term_string(self, term_id: int, name: str) -> None:
        self._output("theory_string", [Number(term_id),String(name)])

    def theory_term_compound(self, term_id: int, name_id_or_type: int,
                             arguments: Sequence[int]) -> None:
        names = {-1:"tuple",-2:"set",-3:"list"}
        if name_id_or_type in names:
            n = ("theory_sequence",Function(names[name_id_or_type],[]))
        else:
            n = ("theory_function",Number(name_id_or_type))
        tuple_id = self._tuple("theory_tuple", self._step_data.theory_tuples, arguments, _theory, True)
        self._output(n[0], [Number(term_id), n[1], tuple_id])

    def theory_element(self, element_id: int, terms: Sequence[int],
                       condition: Sequence[int]) -> None:
        tuple_id = self._tuple("theory_tuple", self._step_data.theory_tuples, terms, _theory, True)
        lit_con_id = self._tuple("literal_tuple", self._step_data.lit_tuples, condition, _lit)
        self._output("theory_element", [Number(element_id), tuple_id, lit_con_id])

    def theory_atom(self, atom_id_or_zero: int, term_id: int,
                    elements: Sequence[int]) -> None:
        tuple_e_id = self._tuple("theory_element_tuple", self._step_data.theory_element_tuples, elements, _lit)
        self._output("theory_atom", [Number(atom_id_or_zero), Number(term_id), tuple_e_id])


    def theory_atom_with_guard(self, atom_id_or_zero: int, term_id: int,
                               elements: Sequence[int], operator_id: int,
                               right_hand_side_id: int) -> None:
        tuple_e_id = self._tuple("theory_element_tuple", self._step_data.theory_element_tuples, elements, _lit)
        self._output("theory_atom", [Number(atom_id_or_zero),
                                    Number(term_id),
                                    tuple_e_id,
                                    Number(operator_id),
                                    Number(right_hand_side_id)
                                    ])

    def end_step(self) -> None:
        if self._reify_steps:
            self.calculate_sccs()
            self._step += 1
            self._step_data = _StepData()


class _SymbolData:
    t_basic: Dict[int, Symbol] = {}
    t_tuple: Dict[int, List[Symbol]] = {}

def _is_op(op:str):
    return op[0] in "/!<=>+-*\\?&@|:;~^."

def get_theory_symbols(reification_symbols : List[Symbol]):
    """
    Gets new predicate for the reification: `theory_symbol`,
    containing the mapping from the index of a `theory_formula` to the associated symbol
    when the theory formula is not constructed with a theory operator.
    """
    t_basic: Dict[int, Symbol] = {}
    t_tuple: Dict[int, List[Symbol]] = {}
    for s in reification_symbols:
        name = s.name
        if not name.startswith('theory_'):
            continue

        idx = s.arguments[0].number
        if name == "theory_string":
            val = s.arguments[1].string
            if not _is_op(val):
                t_basic.setdefault(idx, Function(val,[]))
        elif name == "theory_number":
            t_basic.setdefault(idx,s.arguments[1])
        elif name == "theory_tuple":
            if len(s.arguments)==1:
                t_tuple.setdefault(idx,[])
            else:
                if not idx in t_tuple:
                    continue
                l = t_tuple[idx]
                if not s.arguments[2].number in t_basic:
                    del t_tuple[idx]
                    continue
                l.append(t_basic[s.arguments[2].number])
        elif name == "theory_function":
            if not s.arguments[1].number in t_basic:
                continue
            s = Function(t_basic[s.arguments[1].number].name,t_tuple[s.arguments[2].number])
            t_basic.setdefault(idx,s)
        elif name == "theory_sequence":
            if  s.arguments[1].name != 'tuple':
                raise RuntimeError(f"Not supported {str(s)}")
            s = Function("",t_tuple[s.arguments[2].number])
            t_basic.setdefault(idx,s)

    new_symbols = []
    for idx, s in t_basic.items():
        if s.type!=clingo.SymbolType.Number:
            new_symbols.append(Function("theory_symbol",[Number(idx),s]))

    return new_symbols
