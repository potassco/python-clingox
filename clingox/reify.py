'''
Document me!!!
'''

from typing import Dict, Generic, List, Sequence, Tuple, TypeVar

from clingo.backend import HeuristicType, Observer, TruthValue
from clingo.symbol import Function, Number, Symbol

__all__ = ['Reifier']


class Node:
    def __init__(self, name, visited):
        self.name = name
        self.visited = visited
        self.index = 0
        self.edges = []

T = TypeVar('T')

class Graph(Generic[T]):
    def __init__(self):
        self._names: Dict[T, int] = {}
        self._vertices: List[Node] = []
        self._phase: bool = True

    def _visited(self, key_u):
        return self._vertices[key_u].visited != (not self._phase)

    def _active(self, key_u):
        return self._vertices[key_u].visited != self._phase

    def _add_vertex(self, val_u: T) -> int:
        n = len(self._vertices)
        key_u = self._names.setdefault(val_u, n)
        if n == key_u:
            self._vertices.append(Node(val_u, not self._phase))
        return key_u

    def add_edge(self, val_u: T, val_v: T) -> None:
        key_u = self._add_vertex(val_u)
        key_v = self._add_vertex(val_v)
        self._vertices[key_u].edges.append(key_v)

    def tarjan(self) -> List[List[T]]:
        '''
        // Check!!!
        SCCVec sccs;
        NodeVec stack;
        NodeVec trail;
        for (auto &x : nodes_) {
            if (x.visited_ == nphase()) {
                unsigned index = 1;
                auto push = [&stack, &trail, &index](Node &x) {
                    x.visited_  = ++index;
                    x.finished_ = x.edges_.begin();
                    stack.emplace_back(&x);
                    trail.emplace_back(&x);
                };
                push(x);
                while (!stack.empty()) {
                    auto &y = stack.back();
                    auto end = y->edges_.end();
                    for (; y->finished_ != end && (*y->finished_)->visited_ != nphase(); ++y->finished_) { }
                    if (y->finished_ != end) { push(**y->finished_++); }
                    else {
                        stack.pop_back();
                        bool root = true;
                        for (auto &z : y->edges_) {
                            if (z->visited_ != phase_ && z->visited_ < y->visited_) {
                                root = false;
                                y->visited_ = z->visited_;
                            }
                        }
                        if (root) {
                            sccs.emplace_back();
                            do {
                                sccs.back().emplace_back(trail.back());
                                trail.back()->visited_ = phase_;
                                trail.pop_back();
                            }
                            while (sccs.back().back() != y);
                        }
                    }
                }
            }
        }
        phase_ = nphase();
        return sccs;
        '''
        sccs: List[List[T]] = []
        stack = []
        trail = []
        index = 1
        def push(key_u):
            nonlocal index
            index += 1
            stack.append(key_u)
            trail.append(key_u)

        for key_u in range(len(self._vertices)):
            if not self._visited(key_u):
                continue
            index = 1
            push(key_u)
            while stack:
                key_v = stack[-1]
                vtx_v = self._vertices[key_v]
                len_v = len(vtx_v.edges)
                while vtx_v.index < len_v:
                    if self._vertices[vtx_v.index].visited == (not self._phase):
                        push(vtx_v.edges[vtx_v.index])
                        vtx_v.index += 1
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
                            vtx_last.visited = self._phase
                            trail.pop()

        self._phase = not self._phase
        return sccs

class _StepData:
    def __init__(self):
        self.atom_tuples = {}
        self.lit_tuples = {}
        self.wlit_tuples = {}
        self.graph = Graph()

def _lit(i, lit):
    return [i, Number(lit)]

def _wlit(i, wlit):
    return [i, Number(wlit[0]), Number(wlit[1])]

class Reifier(Observer):
    '''
    Document me!!!
    '''

    def __init__(self, cb):
        self._step = 0
        self._cb = cb
        self._calculate_sccs = False
        self._reify_steps = False
        self._step_data = _StepData()

    def _add_edges(self, head, body):
        if self._calculate_sccs:
            for u in head:
                for v in body:
                    self._step_data.graph.add_edge(u, v)

    def _output(self, name, args):
        if self._reify_steps:
            args = args + [self._step]
        self._cb(Function(name, args))

    def _tuple(self, name, snmap, elems, afun=_lit):
        s = tuple(sorted(elems))
        n = len(snmap)
        i = Number(snmap.setdefault(s, n))
        if n == i.number:
            self._cb(Function(name, [i]))
            for atm in s:
                self._cb(Function(name, afun(i, atm)))
        return i

    def _atom_tuple(self, atoms):
        return self._tuple("atom_tuple", self._step_data.atom_tuples, atoms)

    def _lit_tuple(self, atoms):
        return self._tuple("literal_tuple", self._step_data.lit_tuples, atoms)

    def _wlit_tuple(self, atoms):
        return self._tuple("weighted_literal_tuple", self._step_data.wlit_tuples, atoms, _wlit)

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
                    body: Sequence[Tuple[int,int]]) -> None:
        hn = "choice" if choice else "disjunction"
        hd = Function(hn, [self._atom_tuple(head)])
        bd = Function("sum", [self._wlit_tuple(body), Number(lower_bound)])
        self._output("rule", [hd, bd])
        self._add_edges(head, body)

    def minimize(self, priority: int, literals: Sequence[Tuple[int,int]]) -> None:
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
            self._step += 1
            self._step_data = _StepData()
