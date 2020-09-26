'''
This module provides functions to work with ground programs.

This includes constructing a ground representation using an observer, pretty
printing the ground representation, and adding ground program to control
objects via the backend.
'''

from typing import (Callable, Iterable, List, Mapping, MutableMapping, MutableSequence, NamedTuple, Optional, Sequence,
                    Tuple, TypeVar)
from dataclasses import dataclass, field
from functools import singledispatch
from itertools import chain
from copy import copy

from clingo import Backend, HeuristicType, Observer, Symbol, TruthValue

Atom = int
Literal = int
Weight = int
OutputTable = Mapping[Atom, Symbol]
LiteralMap = Callable[[Literal], Literal]
Statement = TypeVar('Statement', 'Fact', 'Show', 'Rule', 'WeightRule', 'Heuristic', 'Edge', 'Minimize', 'External',
                    'Project')

@singledispatch
def pretty_str(arg: Statement, output_atoms: OutputTable) -> str: # pylint: disable=unused-argument
    '''
    Pretty print program constructs.
    '''
    assert False, 'unexpected type'

@singledispatch
def remap(arg: Statement, mapping: Mapping) -> Statement: # pylint: disable=unused-argument
    '''
    Add statements or programs to the backend using the provided mapping to map
    literals.
    '''
    assert False, 'unexpected type'

@singledispatch
def add_to_backend(arg: Statement, backend: Backend) -> None: # pylint: disable=unused-argument
    '''
    Add statements or programs to the backend using the provided mapping to map
    literals.
    '''
    assert False, 'unexpected type'

# ------------------------------------------------------------------------------

def _pretty_str_lit(arg: Literal, output_atoms: OutputTable) -> str:
    '''
    Pretty print literals and atoms.
    '''
    atom = abs(arg)
    if atom in output_atoms:
        atom_str = str(output_atoms[atom])
    else:
        atom_str = f'__x{atom}'

    return f'not {atom_str}' if arg < 0 else atom_str

def _pretty_str_rule_head(choice: bool, has_body: bool, head: Sequence[Atom], output_atoms: OutputTable) -> str:
    '''
    Pretty print the head of a rule including the implication symbol if
    necessary.
    '''
    ret = ''

    if choice:
        ret += '{'
    ret += '; '.join(_pretty_str_lit(lit, output_atoms) for lit in head)
    if choice:
        ret += '}'

    if has_body or (not head and not choice):
        ret += ' :- '

    return ret

def _pretty_str_truth_value(arg: TruthValue):
    '''
    Pretty print a truth value.
    '''
    if arg == TruthValue.False_:
        return 'False'
    if arg == TruthValue.True_:
        return 'True'
    return 'Free'

def _remap_seq(literals: Sequence[Literal], mapping: LiteralMap):
    '''
    Apply the mapping to a sequence of literals or atoms.
    '''
    return [mapping(lit) for lit in literals]

def _remap_wseq(literals: Sequence[Tuple[Literal, Weight]], mapping: LiteralMap):
    '''
    Apply the mapping to a sequence of weighted literals or atoms.
    '''
    return [(mapping(lit), weight) for lit, weight in literals]

def _remap_stms(stms: MutableSequence[Statement], mapping: LiteralMap):
    '''
    Remap the given statements.
    '''
    for i, stm in enumerate(stms):
        stms[i] = remap(stm, mapping)

def _add_stms_to_backend(stms: Iterable[Statement], backend: Backend, mapping: Optional[LiteralMap]):
    '''
    Remap the given statements returning a list with the result.
    '''
    for stm in stms:
        if mapping:
            add_to_backend(remap(stm, mapping), backend)
        else:
            add_to_backend(stm, backend)

# ------------------------------------------------------------------------------

class Fact(NamedTuple):
    '''
    Ground representation of a fact.
    '''
    symbol: Symbol

@pretty_str.register
def _pretty_str_fact(arg: Fact, output_atoms: OutputTable) -> str: # pylint: disable=unused-argument
    '''
    Pretty print a fact.
    '''
    return f'{arg.symbol}'

@remap.register
def _remap_fact(arg: Fact, mapping: LiteralMap) -> Fact: # pylint: disable=unused-argument
    '''
    Remap a fact statement.
    '''
    return arg

@add_to_backend.register
def _add_to_backend_fact(arg: Fact, backend: Backend) -> None: # pylint: disable=unused-argument
    '''
    Add a fact to the backend.

    This does nothing to not interfere with the mapping of literals. If facts
    are to be mapped, then this should be done manually beforehand.
    '''

# ------------------------------------------------------------------------------

class Show(NamedTuple):
    '''
    Ground representation of a show statements.
    '''
    symbol: Symbol
    condition: Sequence[Literal]

@pretty_str.register
def _pretty_str_show(arg: Show, output_atoms: OutputTable) -> str:
    '''
    Pretty print a fact.
    '''
    body = ', '.join(_pretty_str_lit(lit, output_atoms) for lit in arg.condition)
    return f'#show {arg.symbol}{": " if body else ""}{body}.'

@remap.register
def _remap_show(arg: Show, mapping: LiteralMap) -> Show:
    '''
    Remap a show statetment.
    '''
    return Show(arg.symbol, _remap_seq(arg.condition, mapping))

@add_to_backend.register
def _add_to_backend_show(arg: Show, backend: Backend) -> None: # pylint: disable=unused-argument
    '''
    Add a show statement to the backend.

    Note that this currently does nothing because backend does not yet support
    adding to the symbol table.
    '''

# ------------------------------------------------------------------------------

class Rule(NamedTuple):
    '''
    Ground representation of disjunctive and choice rules.
    '''
    choice: bool
    head: Sequence[Atom]
    body: Sequence[Literal]

@pretty_str.register(Rule)
def _pretty_str_rule(arg: Rule, output_atoms: OutputTable) -> str:
    '''
    Pretty print a rule.
    '''
    head = _pretty_str_rule_head(arg.choice, bool(arg.body), arg.head, output_atoms)
    body = ', '.join(_pretty_str_lit(lit, output_atoms) for lit in arg.body)

    return f'{head}{body}.'

@remap.register
def _remap_rule(arg: Rule, mapping: LiteralMap) -> Rule:
    '''
    Remap literals in a rule.
    '''
    return Rule(arg.choice, _remap_seq(arg.head, mapping), _remap_seq(arg.body, mapping))

@add_to_backend.register
def _add_to_backend_rule(arg: Rule, backend: Backend) -> None:
    '''
    Add a rule to the backend.
    '''
    backend.add_rule(arg.head, arg.body, arg.choice)

# ------------------------------------------------------------------------------

class WeightRule(NamedTuple):
    '''
    Ground representation of rules with a weight constraint in the body.
    '''
    choice: bool
    head: Sequence[Atom]
    lower_bound: Weight
    body: Sequence[Tuple[Literal, Weight]]

@pretty_str.register(WeightRule)
def _pretty_str_weight_rule(arg: WeightRule, output_atoms: OutputTable) -> str:
    '''
    Pretty print a rule or weight rule.
    '''
    head = _pretty_str_rule_head(arg.choice, bool(arg.body), arg.head, output_atoms)
    body = ', '.join(f'{weight},{i}: {_pretty_str_lit(literal, output_atoms)}'
                     for i, (literal, weight) in enumerate(arg.body))

    return f'{head}{arg.lower_bound}{{{body}}}.'

@remap.register
def _remap_weight_rule(arg: WeightRule, mapping: LiteralMap) -> WeightRule:
    '''
    Remap literals in a weight rule.
    '''
    return WeightRule(arg.choice, _remap_seq(arg.head, mapping), arg.lower_bound, _remap_wseq(arg.body, mapping))

@add_to_backend.register
def _add_to_backend_weight_rule(arg: WeightRule, backend: Backend) -> None:
    '''
    Add a weight rule to the backend.
    '''
    backend.add_weight_rule(arg.head, arg.lower_bound, arg.body, arg.choice)

# ------------------------------------------------------------------------------

class Project(NamedTuple):
    '''
    Ground representation of project statements.
    '''
    atom: Atom

@pretty_str.register(Project)
def _pretty_str_project(arg: Project, output_atoms: OutputTable) -> str:
    '''
    Pretty print a project statement.
    '''
    return f'#project {_pretty_str_lit(arg.atom, output_atoms)}.'

@remap.register
def _remap_project(arg: Project, mapping: LiteralMap):
    '''
    Remap project statement.
    '''
    return Project(mapping(arg.atom))

@add_to_backend.register
def _add_to_backend_project(arg: Project, backend: Backend):
    '''
    Add a project statement to the backend.
    '''
    backend.add_project([arg.atom])

# ------------------------------------------------------------------------------

class External(NamedTuple):
    '''
    Ground representation of external atoms.
    '''
    atom: Atom
    value: TruthValue

@pretty_str.register(External)
def _pretty_print_external(arg: External, output_atoms: OutputTable) -> str:
    '''
    Pretty print an external.
    '''
    return f'#external {_pretty_str_lit(arg.atom, output_atoms)}. [{_pretty_str_truth_value(arg.value)}]'

@remap.register
def _remap_external(arg: External, mapping: LiteralMap) -> External:
    '''
    Remap the external.
    '''
    return External(mapping(arg.atom), arg.value)

@add_to_backend.register
def _add_to_backend_external(arg: External, backend: Backend):
    '''
    Add an external statement to the backend remapping its atom.
    '''
    backend.add_external(arg.atom, arg.value)

# ------------------------------------------------------------------------------

class Minimize(NamedTuple):
    '''
    Ground representation of a minimize statement.
    '''
    priority: Weight
    literals: Sequence[Tuple[Literal, Weight]]

@pretty_str.register(Minimize)
def _pretty_print_minimize(arg, output_atoms) -> str:
    '''
    Pretty print a minimize statement.
    '''
    body = '; '.join(f'{weight}@{arg.priority},{i}: {_pretty_str_lit(literal, output_atoms)}'
                     for i, (literal, weight) in enumerate(arg.literals))
    return f'#minimize{{{body}}}.'

@remap.register
def _remap_minimize(arg: Minimize, mapping: LiteralMap) -> Minimize:
    '''
    Remap the literals in the minimize statement.
    '''
    return Minimize(arg.priority, _remap_wseq(arg.literals, mapping))

@add_to_backend.register
def _add_to_backend_minimize(arg: Minimize, backend: Backend):
    '''
    Add a minimize statement to the backend.
    '''
    backend.add_minimize(arg.priority, arg.literals)

# ------------------------------------------------------------------------------

class Heuristic(NamedTuple):
    '''
    Ground representation of a heuristic statement.
    '''
    atom: Atom
    type_: HeuristicType
    bias: Weight
    priority: Weight
    condition: Sequence[Literal]

@pretty_str.register(Heuristic)
def _pretty_str_heuristic(arg: Heuristic, output_atoms: OutputTable) -> str:
    '''
    Pretty print a heuristic statement.
    '''
    body = ', '.join(_pretty_str_lit(lit, output_atoms) for lit in arg.condition)
    head = _pretty_str_lit(arg.atom, output_atoms)
    return f'#heuristic {head}{": " if body else ""}{body}. [{arg.bias}@{arg.priority}, {arg.type_}]'

@remap.register
def _remap_heuristic(arg: Heuristic, mapping: LiteralMap) -> Heuristic:
    '''
    Remap the heuristic statement.
    '''
    return Heuristic(mapping(arg.atom), arg.type_, arg.bias, arg.priority, _remap_seq(arg.condition, mapping))

@add_to_backend.register
def _add_to_backend_heuristic(arg: Heuristic, backend: Backend) -> None:
    '''
    Add a heurisitic statement to the backend.
    '''
    backend.add_heuristic(arg.atom, arg.type_, arg.bias, arg.priority, arg.condition)

# ------------------------------------------------------------------------------

class Edge(NamedTuple):
    '''
    Ground representation of a heuristic statement.
    '''
    u: int
    v: int
    condition: Sequence[Literal]

@pretty_str.register(Edge)
def _pretty_str_edge(arg: Edge, output_atoms: OutputTable) -> str:
    '''
    Pretty print a heuristic statement.
    '''
    body = ', '.join(_pretty_str_lit(lit, output_atoms) for lit in arg.condition)
    return f'#edge ({arg.u},{arg.v}){": " if body else ""}{body}.'

@remap.register
def _remap_edge(arg: Edge, mapping: LiteralMap) -> Edge:
    '''
    Remap an edge statement.
    '''
    return Edge(arg.u, arg.v, _remap_seq(arg.condition, mapping))

@add_to_backend.register
def _add_to_backend_edge(arg: Edge, backend: Backend) -> None:
    '''
    Add an edge statement to the backend remapping its literals.
    '''
    backend.add_acyc_edge(arg.u, arg.v, arg.condition)

# ------------------------------------------------------------------------------

@dataclass
class Program: # pylint: disable=too-many-instance-attributes
    '''
    Ground program representation.

    Although inefficient, the string representation of this program is parsable
    by clingo.
    '''
    output_atoms: MutableMapping[Atom, Symbol] = field(default_factory=dict)
    shows: List[Show] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    weight_rules: List[WeightRule] = field(default_factory=list)
    heuristics: List[Heuristic] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    minimizes: List[Minimize] = field(default_factory=list)
    externals: List[External] = field(default_factory=list)
    projects: Optional[List[Project]] = None
    assumptions: List[Literal] = field(default_factory=list)

    def _pretty_stms(self, arg: Iterable[Statement], sort: bool) -> Iterable[str]:
        if sort:
            arg = sorted(arg)
        return (pretty_str(x, self.output_atoms) for x in arg)

    def _pretty_assumptions(self, sort: bool) -> Iterable[str]:
        if not self.assumptions:
            return []
        arg = sorted(self.assumptions) if sort else self.assumptions
        assumptions = (_pretty_str_lit(lit, self.output_atoms) for lit in arg)
        return [f'% assumptions: {", ".join(assumptions)}']

    def _pretty_projects(self, sort: bool) -> Iterable[str]:
        if self.projects is None:
            return []
        # This is to inform that there is an empty projection statement.
        # It might be worth to allow writing just #project.
        if not self.projects:
            return ['#project x: #false.']
        arg = sorted(self.projects) if sort else self.projects
        return (pretty_str(project, self.output_atoms) for project in arg)

    def sort(self) -> 'Program':
        '''
        Sort the statements in the program inplace.

        Returns a reference to self.

        Note: It might also be nice to sort statement bodies and conditions.
        '''
        self.shows.sort()
        self.facts.sort()
        self.rules.sort()
        self.weight_rules.sort()
        self.heuristics.sort()
        self.edges.sort()
        self.minimizes.sort()
        self.externals.sort()
        if self.projects is not None:
            self.projects.sort()
        self.assumptions.sort()

        return self

    def remap(self, mapping: LiteralMap) -> 'Program':
        '''
        Remap the literals in the program inplace.

        Returns a reference to self.

        Note: It might also be nice to sort statement bodies and conditions.
        '''
        _remap_stms(self.shows, mapping)
        _remap_stms(self.facts, mapping)
        _remap_stms(self.rules, mapping)
        _remap_stms(self.weight_rules, mapping)
        _remap_stms(self.heuristics, mapping)
        _remap_stms(self.edges, mapping)
        _remap_stms(self.minimizes, mapping)
        _remap_stms(self.externals, mapping)
        if self.projects is not None:
            _remap_stms(self.projects, mapping)
        for i, lit in enumerate(self.assumptions):
            self.assumptions[i] = mapping(lit)
        self.output_atoms = {mapping(lit): sym for lit, sym in self.output_atoms.items()}

        return self

    def add_to_backend(self, backend: Backend, mapping: Optional[LiteralMap] = None) -> 'Program':
        '''
        Add the program to the given backend with an optional mapping.

        Returns a reference to self.
        '''

        _add_stms_to_backend(self.shows, backend, mapping)
        _add_stms_to_backend(self.facts, backend, mapping)
        _add_stms_to_backend(self.rules, backend, mapping)
        _add_stms_to_backend(self.weight_rules, backend, mapping)
        _add_stms_to_backend(self.heuristics, backend, mapping)
        _add_stms_to_backend(self.edges, backend, mapping)
        _add_stms_to_backend(self.minimizes, backend, mapping)
        _add_stms_to_backend(self.externals, backend, mapping)
        if self.projects is not None:
            _add_stms_to_backend(self.projects, backend, mapping)

        backend.add_assume(mapping(lit) if mapping else lit
                           for lit in self.assumptions)

        return self

    def pretty_str(self, sort: bool = True) -> str:
        '''
        Return a readable string represenation of the program.
        '''
        return '\n'.join(chain(
            self._pretty_stms(self.shows, sort),
            self._pretty_stms(self.facts, sort),
            self._pretty_stms(self.rules, sort),
            self._pretty_stms(self.weight_rules, sort),
            self._pretty_stms(self.heuristics, sort),
            self._pretty_stms(self.edges, sort),
            self._pretty_stms(self.minimizes, sort),
            self._pretty_stms(self.externals, sort),
            self._pretty_projects(sort),
            self._pretty_assumptions(sort)))

    def copy(self) -> 'Program':
        '''
        Return a shallow copy of the program.

        This copies all mutable state.
        '''
        return copy(self)

    def __str__(self) -> str:
        '''
        Return a readable string represenation of the program.
        '''
        return self.pretty_str()

# ------------------------------------------------------------------------------

class Remapping:
    '''
    This class maps existing literals to fresh literals as created by the
    backend.
    '''
    _backend: Backend
    _map: MutableMapping[Atom, Atom]

    def __init__(self, backend: Backend, output_atoms: OutputTable, facts: Iterable[Fact] = None):
        '''
        Initializes the mapping with the literals in the given output table.

        Furthemore, it associates a fresh literal with each given fact.
        '''
        self._backend = backend
        self._map = {}
        for atom, sym in output_atoms.items():
            assert atom not in self._map
            self._map[atom] = self._backend.add_atom(sym)
        if facts is not None:
            for fact in facts:
                backend.add_rule([backend.add_atom(fact.symbol)])

    def __call__(self, lit: Literal) -> Literal:
        '''
        Map the given literal to the corresponding literal in the backend.

        If the literal was not mapped during initialization, a new literal is
        associated with it.
        '''
        atom = abs(lit)
        if atom not in self._map:
            self._map[atom] = self._backend.add_atom()

        ret = self._map[atom]
        return -ret if lit < 0 else -ret

# ------------------------------------------------------------------------------

class ProgramObserver(Observer):
    '''
    Program observer to build a ground program representation while grounding.

    This class explicitly ignores theory atoms because they already have a
    ground representation.
    '''
    _program: Program

    def __init__(self, program: Program):
        self._program = program

    def begin_step(self) -> None:
        '''
        Resets the assumptions.
        '''
        self._program.assumptions.clear()

    def output_atom(self, symbol: Symbol, atom: Atom) -> None:
        '''
        Add the given atom to the list of facts or output table.
        '''
        if atom != 0:
            self._program.output_atoms[atom] = symbol
        else:
            self._program.facts.append(Fact(symbol))

    def output_term(self, symbol: Symbol, condition: Sequence[Literal]) -> None:
        '''
        Add a term to the output table.
        '''
        self._program.shows.append(Show(symbol, condition))

    def rule(self, choice: bool, head: Sequence[Atom], body: Sequence[Literal]) -> None:
        '''
        Add a rule to the ground representation.
        '''
        self._program.rules.append(Rule(choice, head, body))

    def weight_rule(self, choice: bool, head: Sequence[Atom], lower_bound: Weight,
                    body: Sequence[Tuple[Literal, Weight]]) -> None:
        '''
        Add a weight rule to the ground representation.
        '''
        self._program.weight_rules.append(WeightRule(choice, head, lower_bound, body))

    def project(self, atoms: Sequence[Atom]) -> None:
        '''
        Add a project statement to the ground representation.
        '''
        if self._program.projects is None:
            self._program.projects = []
        self._program.projects.extend(Project(atom) for atom in atoms)

    def external(self, atom: Atom, value: TruthValue) -> None:
        '''
        Add an external statement to the ground representation.
        '''
        self._program.externals.append(External(atom, value))

    def assume(self, literals: Sequence[Literal]) -> None:
        '''
        Extend the program with the given assumptions.
        '''
        self._program.assumptions.extend(literals)

    def minimize(self, priority: Weight, literals: Sequence[Tuple[Literal, Weight]]) -> None:
        '''
        Add a minimize statement to the ground representation.
        '''
        self._program.minimizes.append(Minimize(priority, literals))

    def acyc_edge(self, node_u: int, node_v: int, condition: Sequence[Literal]) -> None:
        '''
        Add an edge statement to the gronud representation.
        '''
        self._program.edges.append(Edge(node_u, node_v, condition))

    def heuristic(self, atom: Atom, type_: HeuristicType, bias: Weight, priority: Weight,
                  condition: Sequence[Literal]) -> None:
        '''
        Add heurisitic statement to the gronud representation.
        '''
        self._program.heuristics.append(Heuristic(atom, type_, bias, priority, condition))
