'''
This module provides functions to work with ground programs.

This includes constructing a ground representation using an observer, pretty
printing the ground representation, and adding ground program to control
objects via the backend.

Example
-------

The following example shows how to:

- use the `ProgramObserver` to construct a `Program`, and
- add it to another `clingo.control.Control` object.

```python-repl
>>> from clingo.control import Control
>>> from clingox.program import Program, ProgramObserver, Remapping
>>>
>>> prg = Program()
>>> ctl_a = Control()
>>> ctl_a.register_observer(ProgramObserver(prg))
>>>
>>> ctl_a.add('base', [], 'a. {b}. c :- b.')
>>> ctl_a.ground([('base', [])])
>>> print(prg)
a.
__x1.
c :- b.
{b}.
>>>
>>> ctl_b = Control(['0'])
>>> with ctl_b.backend() as backend:
...     mapping = Remapping(backend, prg.output_atoms, prg.facts)
...     prg.add_to_backend(backend, mapping)
...
>>> ctl_b.solve(on_model=print)
a
b c a
```
'''

from typing import (Callable, Iterable, List, Mapping, MutableMapping, MutableSequence, NamedTuple, Optional, Sequence,
                    Tuple, TypeVar)
from dataclasses import dataclass, field
from functools import singledispatch
from itertools import chain
from copy import copy

from clingo import Backend, HeuristicType, Observer, Symbol, TruthValue

__all__ = ['add_to_backend', 'pretty_str', 'remap', 'Edge', 'External', 'Fact',
           'Heuristic', 'Minimize', 'Program', 'ProgramObserver', 'Project',
           'Remapping', 'Rule', 'Show', 'WeightRule']
__pdoc__ = {}

Atom = int
Literal = int
Weight = int
OutputTable = Mapping[Atom, Symbol]
AtomMap = Callable[[Atom], Atom]
Statement = TypeVar('Statement', 'Fact', 'Show', 'Rule', 'WeightRule', 'Heuristic', 'Edge', 'Minimize', 'External',
                    'Project')

@singledispatch
def pretty_str(stm: Statement, output_atoms: OutputTable) -> str: # pylint: disable=unused-argument
    '''
    Pretty print statements.

    Parameters
    ----------
    stm
        The statement to convert to a string.
    output_atoms
        A mapping from program atoms to symbols.

    Returns
    -------
    The string representation of the statement.
    '''
    assert False, 'unexpected type'

@singledispatch
def remap(stm: Statement, mapping: AtomMap) -> Statement: # pylint: disable=unused-argument
    '''
    Remap literals in the given statement with the provided mapping.

    Parameters
    ----------
    stm
        The statement to remap.
    mapping
        The mapping function to remap literals.

    Returns
    -------
    The updated statement.

    See Also
    --------
    Remapping
    '''
    assert False, 'unexpected type'

@singledispatch
def add_to_backend(stm: Statement, backend: Backend) -> None: # pylint: disable=unused-argument
    '''
    Add statements to the backend using the provided mapping to map literals.

    Parameters
    ----------
    stm
        The statement to add to the backend.
    backend
        The backend.
    '''
    assert False, 'unexpected type'

# ------------------------------------------------------------------------------

def _pretty_str_lit(stm: Literal, output_atoms: OutputTable) -> str:
    '''
    Pretty print literals and atoms.
    '''
    atom = abs(stm)
    if atom in output_atoms:
        atom_str = str(output_atoms[atom])
    else:
        atom_str = f'__x{atom}'

    return f'not {atom_str}' if stm < 0 else atom_str

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

def _pretty_str_truth_value(stm: TruthValue):
    '''
    Pretty print a truth value.
    '''
    if stm == TruthValue.False_:
        return 'False'
    if stm == TruthValue.True_:
        return 'True'
    return 'Free'

def _remap_lit(literal: Literal, mapping: AtomMap) -> Atom:
    return -mapping(-literal) if literal < 0 else mapping(literal)

def _remap_seq(literals: Sequence[Literal], mapping: AtomMap):
    '''
    Apply the mapping to a sequence of literals or atoms.
    '''
    return [_remap_lit(lit, mapping) for lit in literals]

def _remap_wseq(literals: Sequence[Tuple[Literal, Weight]], mapping: AtomMap):
    '''
    Apply the mapping to a sequence of weighted literals or atoms.
    '''
    return [(_remap_lit(lit, mapping), weight) for lit, weight in literals]

def _remap_stms(stms: MutableSequence[Statement], mapping: AtomMap):
    '''
    Remap the given statements.
    '''
    for i, stm in enumerate(stms):
        stms[i] = remap(stm, mapping)

def _add_stms_to_backend(stms: Iterable[Statement], backend: Backend, mapping: Optional[AtomMap]):
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
def _pretty_str_fact(stm: Fact, output_atoms: OutputTable) -> str: # pylint: disable=unused-argument
    '''
    Pretty print a fact.
    '''
    return f'{stm.symbol}.'

@remap.register
def _remap_fact(stm: Fact, mapping: AtomMap) -> Fact: # pylint: disable=unused-argument
    '''
    Remap a fact statement.
    '''
    return stm

@add_to_backend.register
def _add_to_backend_fact(stm: Fact, backend: Backend) -> None: # pylint: disable=unused-argument
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
def _pretty_str_show(stm: Show, output_atoms: OutputTable) -> str:
    '''
    Pretty print a fact.
    '''
    body = ', '.join(_pretty_str_lit(lit, output_atoms) for lit in stm.condition)
    return f'#show {stm.symbol}{": " if body else ""}{body}.'

@remap.register
def _remap_show(stm: Show, mapping: AtomMap) -> Show:
    '''
    Remap a show statetment.
    '''
    return Show(stm.symbol, _remap_seq(stm.condition, mapping))

@add_to_backend.register
def _add_to_backend_show(stm: Show, backend: Backend) -> None: # pylint: disable=unused-argument
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
def _pretty_str_rule(stm: Rule, output_atoms: OutputTable) -> str:
    '''
    Pretty print a rule.
    '''
    head = _pretty_str_rule_head(stm.choice, bool(stm.body), stm.head, output_atoms)
    body = ', '.join(_pretty_str_lit(lit, output_atoms) for lit in stm.body)

    return f'{head}{body}.'

@remap.register
def _remap_rule(stm: Rule, mapping: AtomMap) -> Rule:
    '''
    Remap literals in a rule.
    '''
    return Rule(stm.choice, _remap_seq(stm.head, mapping), _remap_seq(stm.body, mapping))

@add_to_backend.register
def _add_to_backend_rule(stm: Rule, backend: Backend) -> None:
    '''
    Add a rule to the backend.
    '''
    backend.add_rule(stm.head, stm.body, stm.choice)

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
def _pretty_str_weight_rule(stm: WeightRule, output_atoms: OutputTable) -> str:
    '''
    Pretty print a rule or weight rule.
    '''
    head = _pretty_str_rule_head(stm.choice, bool(stm.body), stm.head, output_atoms)
    body = ', '.join(f'{weight},{i}: {_pretty_str_lit(literal, output_atoms)}'
                     for i, (literal, weight) in enumerate(stm.body))

    return f'{head}{stm.lower_bound}{{{body}}}.'

@remap.register
def _remap_weight_rule(stm: WeightRule, mapping: AtomMap) -> WeightRule:
    '''
    Remap literals in a weight rule.
    '''
    return WeightRule(stm.choice, _remap_seq(stm.head, mapping), stm.lower_bound, _remap_wseq(stm.body, mapping))

@add_to_backend.register
def _add_to_backend_weight_rule(stm: WeightRule, backend: Backend) -> None:
    '''
    Add a weight rule to the backend.
    '''
    backend.add_weight_rule(stm.head, stm.lower_bound, stm.body, stm.choice)

# ------------------------------------------------------------------------------

class Project(NamedTuple):
    '''
    Ground representation of project statements.
    '''
    atom: Atom

@pretty_str.register(Project)
def _pretty_str_project(stm: Project, output_atoms: OutputTable) -> str:
    '''
    Pretty print a project statement.
    '''
    return f'#project {_pretty_str_lit(stm.atom, output_atoms)}.'

@remap.register
def _remap_project(stm: Project, mapping: AtomMap):
    '''
    Remap project statement.
    '''
    return Project(mapping(stm.atom))

@add_to_backend.register
def _add_to_backend_project(stm: Project, backend: Backend):
    '''
    Add a project statement to the backend.
    '''
    backend.add_project([stm.atom])

# ------------------------------------------------------------------------------

class External(NamedTuple):
    '''
    Ground representation of external atoms.
    '''
    atom: Atom
    value: TruthValue

@pretty_str.register(External)
def _pretty_print_external(stm: External, output_atoms: OutputTable) -> str:
    '''
    Pretty print an external.
    '''
    return f'#external {_pretty_str_lit(stm.atom, output_atoms)}. [{_pretty_str_truth_value(stm.value)}]'

@remap.register
def _remap_external(stm: External, mapping: AtomMap) -> External:
    '''
    Remap the external.
    '''
    return External(mapping(stm.atom), stm.value)

@add_to_backend.register
def _add_to_backend_external(stm: External, backend: Backend):
    '''
    Add an external statement to the backend remapping its atom.
    '''
    backend.add_external(stm.atom, stm.value)

# ------------------------------------------------------------------------------

class Minimize(NamedTuple):
    '''
    Ground representation of a minimize statement.
    '''
    priority: Weight
    literals: Sequence[Tuple[Literal, Weight]]

@pretty_str.register(Minimize)
def _pretty_print_minimize(stm, output_atoms) -> str:
    '''
    Pretty print a minimize statement.
    '''
    body = '; '.join(f'{weight}@{stm.priority},{i}: {_pretty_str_lit(literal, output_atoms)}'
                     for i, (literal, weight) in enumerate(stm.literals))
    return f'#minimize{{{body}}}.'

@remap.register
def _remap_minimize(stm: Minimize, mapping: AtomMap) -> Minimize:
    '''
    Remap the literals in the minimize statement.
    '''
    return Minimize(stm.priority, _remap_wseq(stm.literals, mapping))

@add_to_backend.register
def _add_to_backend_minimize(stm: Minimize, backend: Backend):
    '''
    Add a minimize statement to the backend.
    '''
    backend.add_minimize(stm.priority, stm.literals)

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

def _pretty_str_heuristic_type(type_):
    return str(type_).replace('HeuristicType.', '')

@pretty_str.register(Heuristic)
def _pretty_str_heuristic(stm: Heuristic, output_atoms: OutputTable) -> str:
    '''
    Pretty print a heuristic statement.
    '''
    body = ', '.join(_pretty_str_lit(lit, output_atoms) for lit in stm.condition)
    head = _pretty_str_lit(stm.atom, output_atoms)
    type_ = _pretty_str_heuristic_type(stm.type_)
    return f'#heuristic {head}{": " if body else ""}{body}. [{stm.bias}@{stm.priority}, {type_}]'

@remap.register
def _remap_heuristic(stm: Heuristic, mapping: AtomMap) -> Heuristic:
    '''
    Remap the heuristic statement.
    '''
    return Heuristic(mapping(stm.atom), stm.type_, stm.bias, stm.priority, _remap_seq(stm.condition, mapping))

@add_to_backend.register
def _add_to_backend_heuristic(stm: Heuristic, backend: Backend) -> None:
    '''
    Add a heurisitic statement to the backend.
    '''
    backend.add_heuristic(stm.atom, stm.type_, stm.bias, stm.priority, stm.condition)

# ------------------------------------------------------------------------------

class Edge(NamedTuple):
    '''
    Ground representation of a heuristic statement.
    '''
    u: int
    v: int
    condition: Sequence[Literal]

@pretty_str.register(Edge)
def _pretty_str_edge(stm: Edge, output_atoms: OutputTable) -> str:
    '''
    Pretty print a heuristic statement.
    '''
    body = ', '.join(_pretty_str_lit(lit, output_atoms) for lit in stm.condition)
    return f'#edge ({stm.u},{stm.v}){": " if body else ""}{body}.'

@remap.register
def _remap_edge(stm: Edge, mapping: AtomMap) -> Edge:
    '''
    Remap an edge statement.
    '''
    return Edge(stm.u, stm.v, _remap_seq(stm.condition, mapping))

@add_to_backend.register
def _add_to_backend_edge(stm: Edge, backend: Backend) -> None:
    '''
    Add an edge statement to the backend remapping its literals.
    '''
    backend.add_acyc_edge(stm.u, stm.v, stm.condition)

# ------------------------------------------------------------------------------

@dataclass
class Program: # pylint: disable=too-many-instance-attributes
    '''
    Ground program representation.

    Although inefficient, the string representation of this program is parsable
    by clingo.
    '''
    output_atoms: MutableMapping[Atom, Symbol] = field(default_factory=dict)
    '''
    A mapping from program atoms to symbols.
    '''
    shows: List[Show] = field(default_factory=list)
    '''
    A list of show statements.
    '''
    facts: List[Fact] = field(default_factory=list)
    '''
    A list of facts.
    '''
    rules: List[Rule] = field(default_factory=list)
    '''
    A list of rules.
    '''
    weight_rules: List[WeightRule] = field(default_factory=list)
    '''
    A list of weight rules.
    '''
    heuristics: List[Heuristic] = field(default_factory=list)
    '''
    A list of heuristic statements.
    '''
    edges: List[Edge] = field(default_factory=list)
    '''
    A list of edge statements.
    '''
    minimizes: List[Minimize] = field(default_factory=list)
    '''
    A list of minimize statements.
    '''
    externals: List[External] = field(default_factory=list)
    '''
    A list of external statements.
    '''
    projects: Optional[List[Project]] = None
    '''
    A list of project statements.
    '''
    assumptions: List[Literal] = field(default_factory=list)
    '''
    A list of assumptions in form of program literals.
    '''

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

        Returns
        -------
        A reference to self.
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

    def remap(self, mapping: AtomMap) -> 'Program':
        '''
        Remap the literals in the program inplace.

        Parameters
        ----------
        mapping
            A function to remap program atoms.

        Returns
        -------
        A reference to self.

        See Also
        --------
        remap
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
            self.assumptions[i] = _remap_lit(lit, mapping)
        self.output_atoms = {mapping(lit): sym for lit, sym in self.output_atoms.items()}

        return self

    def add_to_backend(self, backend: Backend, mapping: Optional[AtomMap] = None) -> 'Program':
        '''
        Add the program to the given backend with an optional mapping.

        Note that the output table cannot be added to the backend for technical
        reasons. This has to be taken care of by the user. See for example the
        `Remapping` class, which provides functionality for this.

        Parameters
        ----------
        backend
            The backend.
        mapping
            A mapping function to remap literals.

        Returns
        -------
        A reference to self.

        See Also
        --------
        add_to_backend
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
            if self.projects:
                _add_stms_to_backend(self.projects, backend, mapping)
            else:
                backend.add_project([])

        backend.add_assume([_remap_lit(lit, mapping) if mapping else lit
                            for lit in self.assumptions])

        return self

    def pretty_str(self, sort: bool = True) -> str:
        '''
        Return a readable string represenation of the program.

        Parameters
        ----------
        sort
            Whether to sort the statements in the program befor printing.

        Returns
        -------
        The string representation of the program.
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
        Return a shallow copy of the program copying all mutable state.

        Returns
        -------
        A shallow copy of the program.
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

    Parameters
    ----------
    backend
        The backend used to introduce fresh atoms.
    output_atoms
        The output table to initialize the mapping with.
    facts
        A list of facts each of which will receive a fresh program atom.
    '''
    _backend: Backend
    _map: MutableMapping[Atom, Atom]

    def __init__(self, backend: Backend, output_atoms: OutputTable, facts: Iterable[Fact] = None):
        self._backend = backend
        self._map = {}
        for atom, sym in output_atoms.items():
            assert atom not in self._map
            self._map[atom] = self._backend.add_atom(sym)
        if facts is not None:
            for fact in facts:
                backend.add_rule([backend.add_atom(fact.symbol)])

    def __call__(self, atom: Atom) -> Atom:
        '''
        Map the given program atom to the corresponding atom in the backend.

        If the literal was not mapped during initialization, a new literal is
        associated with it.

        Parameters
        ----------
        atom
            The atom to remap.

        Returns
        -------
        The remapped program atom.
        '''
        if atom not in self._map:
            self._map[atom] = self._backend.add_atom()

        return self._map[atom]

__pdoc__['Remapping.__call__'] = True

# ------------------------------------------------------------------------------

class ProgramObserver(Observer):
    '''
    Program observer to build a ground program representation while grounding.

    This class explicitly ignores theory atoms because they already have a
    ground representation.

    Parameters
    ----------
    program
        The program to add statements to.
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

        Parameters
        ----------
        choice
            Determines if the head is a choice or a disjunction.
        head
            List of program atoms forming the rule head.
        body
            List of program literals forming the rule body.
        '''
        self._program.rules.append(Rule(choice, head, body))

    def weight_rule(self, choice: bool, head: Sequence[Atom], lower_bound: Weight,
                    body: Sequence[Tuple[Literal, Weight]]) -> None:
        '''
        Add a weight rule to the ground representation.

        Parameters
        ----------
        choice
            Determines if the head is a choice or a disjunction.
        head
            List of program atoms forming the head of the rule.
        lower_bound
            The lower bound of the weight constraint in the rule body.
        body
            List of weighted literals (pairs of literal and weight) forming the
            elements of the weight constraint.
        '''
        self._program.weight_rules.append(WeightRule(choice, head, lower_bound, body))

    def project(self, atoms: Sequence[Atom]) -> None:
        '''
        Add a project statement to the ground representation.

        Parameters
        ----------
        atoms
            The program atoms to project on.
        '''
        if self._program.projects is None:
            self._program.projects = []
        self._program.projects.extend(Project(atom) for atom in atoms)

    def external(self, atom: Atom, value: TruthValue) -> None:
        '''
        Add an external statement to the ground representation.

        Parameters
        ----------
        atom
            The external atom in form of a program literal.
        value
            The truth value of the external statement.
        '''
        self._program.externals.append(External(atom, value))

    def assume(self, literals: Sequence[Literal]) -> None:
        '''
        Extend the program with the given assumptions.

        Parameters
        ----------
        literals
            The program literals to assume (positive literals are true and
            negative literals false for the next solve call).
        '''
        self._program.assumptions.extend(literals)

    def minimize(self, priority: Weight, literals: Sequence[Tuple[Literal, Weight]]) -> None:
        '''
        Add a minimize statement to the ground representation.

        Parameters
        ----------
        priority
            The priority of the directive.
        literals
            List of weighted literals whose sum to minimize (pairs of literal
            and weight).
        '''
        self._program.minimizes.append(Minimize(priority, literals))

    def acyc_edge(self, node_u: int, node_v: int, condition: Sequence[Literal]) -> None:
        '''
        Add an edge statement to the gronud representation.

        Parameters
        ----------
        node_u
            The start vertex of the edge (in form of an integer).
        node_v
            Тhe end vertex of the edge (in form of an integer).
        condition
            The list of program literals forming th condition under which to
            add the edge.
        '''
        self._program.edges.append(Edge(node_u, node_v, condition))

    def heuristic(self, atom: Atom, type_: HeuristicType, bias: Weight, priority: Weight,
                  condition: Sequence[Literal]) -> None:
        '''
        Add heurisitic statement to the gronud representation.

        Parameters
        ----------
        atom
            The program atom heuristically modified.
        type_
            The type of the modification.
        bias
            A signed integer.
        priority
            An unsigned integer.
        condition
            List of program literals.
        '''
        self._program.heuristics.append(Heuristic(atom, type_, bias, priority, condition))
