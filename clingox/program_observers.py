'''
This module provides functions to work with ground programs.

This includes construting a ground representation using an observer, pretty
printing the ground representation, and adding ground program to control
objects via the backend.
'''

from typing import Any, List, Mapping, MutableMapping, NamedTuple, Sequence, Tuple, Union
from dataclasses import dataclass, field
from functools import singledispatch
from itertools import chain
from textwrap import dedent

from clingo import HeuristicType, Symbol, Observer, TruthValue

Atom = int
Literal = int
Weight = int
OutputTable = Mapping[int, Symbol]


@singledispatch
def pretty_str(arg: Any, output_atoms: OutputTable): # pylint: disable=unused-argument
    '''
    Pretty print program constructs.
    '''
    assert False, "unexpected type"


@pretty_str.register
def _pretty_str_lit(arg: Literal, output_atoms: OutputTable):
    '''
    Pretty print literals and atoms.
    '''
    atom = abs(arg)
    if atom in output_atoms:
        atom_str = str(output_atoms[atom])
    else:
        atom_str = f"__x{atom}"

    return f"not {atom_str}" if arg < 0 else atom_str

@pretty_str.register(TruthValue)
def _pretty_str_truth_value(arg: TruthValue, output_atoms: OutputTable):
    '''
    Pretty print a truth value.
    '''
    if arg == TruthValue.False_:
        return "False"
    if arg == TruthValue.True_:
        return "True"
    return "Free"


class Fact(NamedTuple):
    '''
    Ground representation of a fact.
    '''
    symbol: Symbol

@pretty_str.register
def _pretty_str_fact(arg: Fact, output_atoms: OutputTable): # pylint: disable=unused-argument
    '''
    Pretty print a fact.
    '''
    return f"{arg.symbol}"


class Show(NamedTuple):
    '''
    Ground representation of a show statements.
    '''
    symbol: Symbol
    condition: Sequence[Literal]

@pretty_str.register
def _pretty_str_show(arg: Show, output_atoms: OutputTable):
    '''
    Pretty print a fact.
    '''
    body = ', '.join(pretty_str(lit, output_atoms) for lit in arg.condition)
    return f'#show {arg.symbol}{": " if body else ""}{body}.'


class Rule(NamedTuple):
    '''
    Ground representation of disjunctive and choice rules.
    '''
    choice: bool
    head: Sequence[Atom]
    body: Sequence[Literal]

class WeightRule(NamedTuple):
    '''
    Ground representation of rules with a weight constraint in the body.
    '''
    choice: bool
    head: Sequence[Atom]
    lower_bound: Weight
    body: Sequence[Tuple[Literal, Weight]]

@pretty_str.register(Rule)
@pretty_str.register(WeightRule)
def _pretty_str_rule(arg: Union[Rule, WeightRule], output_atoms: OutputTable):
    '''
    Pretty print a rule or weight rule.
    '''
    ret = ""
    if arg.choice:
        ret += "{"
    ret += "; ".join(pretty_str(lit, output_atoms) for lit in arg.head)
    if arg.choice:
        ret += "}"
    if arg.body or (not arg.head and not arg.choice):
        ret += " :- "

    if isinstance(arg, WeightRule):
        body = ", ".join(f'{weight},{i}: {pretty_str(literal, output_atoms)}'
                         for i, (literal, weight) in enumerate(arg.body))
        ret += f'{arg.lower_bound}{{{body}}}'
    else:
        ret += ", ".join(pretty_str(lit, output_atoms) for lit in arg.body)
    ret += "."
    return ret


class Project(NamedTuple):
    '''
    Ground representation of project statements.
    '''
    atoms: Sequence[Atom]

@pretty_str.register(Project)
def _pretty_str_project(arg: Project, output_atoms: OutputTable):
    '''
    Pretty print a project statement.
    '''
    atoms = ', '.join(pretty_str(atom, output_atoms) for atom in arg.atoms)
    return f'#project{" " if atoms else ""}{atoms}.'


class External(NamedTuple):
    '''
    Ground representation of external atoms.
    '''
    atom: Atom
    value: TruthValue

@pretty_str.register(External)
def _pretty_print_external(arg: External, output_atoms: OutputTable):
    '''
    Pretty print an external.
    '''
    return f"#external {pretty_str(arg.atom, output_atoms)}. [{pretty_str(arg.value, output_atoms)}]"


class Minimize(NamedTuple):
    '''
    Ground representation of a minimize statement.
    '''
    priority: Weight
    literals: Sequence[Tuple[Literal, Weight]]

@pretty_str.register(Minimize)
def _pretty_print_minimize(arg, output_atoms):
    '''
    Pretty print a minimize statement.
    '''
    body = "; ".join(f"{weight}@{arg.priority},{i}: {pretty_str(literal, output_atoms)}"
                     for i, (literal, weight) in enumerate(arg.literals))
    return f"#minimize{{{body}}}."


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
def _pretty_str_heuristic(arg: Heuristic, output_atoms: OutputTable):
    '''
    Pretty print a heuristic statement.
    '''
    body = ', '.join(pretty_str(lit, output_atoms) for lit in arg.condition)
    return f'#heuristic {arg.atom}{": " if body else ""}{body}. [{arg.bias}@{arg.priority}, {arg.type_}]'


class Edge(NamedTuple):
    '''
    Ground representation of a heuristic statement.
    '''
    u: int
    v: int
    condition: Sequence[Literal]

@pretty_str.register(Edge)
def _pretty_str_edge(arg: Edge, output_atoms: OutputTable):
    '''
    Pretty print a heuristic statement.
    '''
    body = ', '.join(pretty_str(lit, output_atoms) for lit in arg.condition)
    return f'#edge ({arg.u},{arg.v}){": " if body else ""}{body}.'


class Assume(NamedTuple):
    '''
    Ground representation of an assume statement.
    '''
    literals: Sequence[Literal]

@pretty_str.register(Assume)
def _pretty_str_assume(arg: Assume, output_atoms: OutputTable):
    '''
    Pretty print a assume statement.
    '''
    body = ', '.join(pretty_str(lit, output_atoms) for lit in arg.literals)
    return f'#assume{" " if body else ""}{body}.'


@dataclass
class Program: # pylint: disable=too-many-instance-attributes
    '''
    Ground program representation.

    Although inefficient, the string representation of this program is parsable
    by clingo.
    '''
    output_atoms: MutableMapping[int, Symbol] = field(default_factory=dict)
    shows: List[Show] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    weight_rules: List[WeightRule] = field(default_factory=list)
    heuristics: List[Heuristic] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    minimizes: List[Minimize] = field(default_factory=list)
    externals: List[External] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    assumes: List[Assume] = field(default_factory=list)

    def __repr__(self):
        return dedent(f'''\
            Program(output_atoms={repr(self.output_atoms)},
                    shows={repr(self.shows)},
                    facts={repr(self.facts)},
                    rules={repr(self.rules)},
                    weight_rules={repr(self.rules)},
                    heuristics={repr(self.heuristics)},
                    edges={repr(self.edges)},
                    minimizes={repr(self.minimizes)},
                    externals={repr(self.externals)},
                    projects={repr(self.projects)},
                    assumes={repr(self.assumes)})''')

    def __str__(self):
        return "\n".join(pretty_str(stm, self.output_atoms) for stm in chain(
            sorted(self.shows),
            sorted(self.facts),
            sorted(self.rules),
            sorted(self.weight_rules),
            sorted(self.heuristics),
            sorted(self.edges),
            sorted(self.minimizes),
            sorted(self.externals),
            sorted(self.projects),
            sorted(self.assumes)))


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
        self._program.assumes.clear()

    def output_atom(self, symbol: Symbol, atom: int) -> None:
        '''
        Add the given atom to the list of facts or output table.
        '''
        if atom != 0:
            self._program.output_atoms[atom] = symbol
        else:
            self._program.facts.append(Fact(symbol))

    def output_term(self, symbol: Symbol, condition: Sequence[int]) -> None:
        '''
        Add a term to the output table.
        '''
        self._program.shows.append(Show(symbol, condition))

    def rule(self, choice: bool, head: Sequence[int], body: Sequence[int]) -> None:
        '''
        Add a rule to the ground representation.
        '''
        self._program.rules.append(Rule(choice, head, body))

    def weight_rule(self, choice: bool, head: Sequence[int], lower_bound: int, body: Sequence[Tuple[int, int]]) -> None:
        '''
        Add a weight rule to the ground representation.
        '''
        self._program.weight_rules.append(WeightRule(choice, head, lower_bound, body))

    def project(self, atoms: Sequence[int]) -> None:
        '''
        Add a project statement to the ground representation.
        '''
        self._program.projects.append(Project(atoms))

    def external(self, atom: int, value: TruthValue) -> None:
        '''
        Add an external statement to the ground representation.
        '''
        self._program.externals.append(External(atom, value))

    def minimize(self, priority: int, literals: Sequence[Tuple[int, int]]) -> None:
        '''
        Add a minimize statement to the ground representation.
        '''
        self._program.minimizes.append(Minimize(priority, literals))

    def acyc_edge(self, node_u: int, node_v: int, condition: Sequence[int]) -> None:
        '''
        Add an edge statement to the gronud representation.
        '''
        self._program.edges.append(Edge(node_u, node_v, condition))
