from typing import Any, Iterable, List, NamedTuple, MutableMapping, Sequence, Tuple
from dataclasses import dataclass, field
from functools import singledispatch
from itertools import chain

from clingo import HeuristicType, Symbol, Function, Observer, TruthValue


@singledispatch
def pretty_str(lit: Any, output_atoms): # pylint: disable=unused-argument
    assert "unexpected type" # pylint: disable=assert-on-string-literal

@pretty_str.register
def _(arg: int, output_atoms):
    """
    Pretty printliterals and atoms.
    """
    atom = abs(arg)
    if atom in output_atoms:
        atom_str = str(output_atoms[atom])
    else:
        atom_str = f"__x{atom}"

    return f"not {atom_str}" if arg < 0 else atom_str


class Rule(NamedTuple):
    choice: bool
    head: List[int]
    body: List[int]

class WeightRule(NamedTuple):
    choice: bool
    head: Sequence[int]
    lower_bound: int
    body: Sequence[Tuple[int, int]]

@pretty_str.register(Rule)
@pretty_str.register(WeightRule)
def _(arg, output_atoms):
    ret = ""
    if arg.choice:
        ret += "{"
    ret += "; ".join(pretty_str(lit, output_atoms) for lit in arg.head)
    if arg.choice:
        ret += "}"
    if arg.body or (not arg.head and not arg.choice):
        ret += " :- "
         
    if isinstance(arg, WeightRule):
        body = ", ".join(str(x[1]) + ":" + pretty_str(x[0], output_atoms) for x in arg.body)
        ret += str(arg.lower_bound) + "{" + body + "}"
    else:
        ret += ", ".join(pretty_str(lit, output_atoms) for lit in arg.body)
    ret += "."
    return ret

class Project(NamedTuple):
    atoms: Sequence[int]

@pretty_str.register(Project)
def _(arg, output_atoms):
    atoms = ','.join(pretty_str(atom, output_atoms) for atom in arg.atoms)
    if atoms:
        return '#project ' + atoms  + '.'
    else:
        return '#project.'

class External(NamedTuple):
    atom: int
    value: TruthValue

@pretty_str.register(TruthValue)
def _(arg, output_atoms):
    if arg == TruthValue.False_:
        return "False"
    elif arg == TruthValue.True_:
        return "True"
    else:
        return str(arg)

@pretty_str.register(External)
def _(arg, output_atoms):
    return f"#external {pretty_str(arg.atom, output_atoms)}. % value={pretty_str(arg.value, output_atoms)}"

class Minimize(NamedTuple):
    priority: int
    literals: Iterable[Tuple[int,int]]

@pretty_str.register(Minimize)
def _(arg, output_atoms):
    body = ((pretty_str(x[0], output_atoms),x[1]) for x in arg.literals)
    body = "; ".join(f"{x[1]}@{arg.priority},{x[0]}:{x[0]}" for x in body)
    return f"#minimize{{{body}}}."

class Heuristic(NamedTuple):
    atom: int
    type_: HeuristicType
    bias: int
    priority: int
    condition: Iterable[int]

class Assume(NamedTuple):
    literals: Iterable[int]

# This combines the pretty program and program classes and should have some more methods.
@dataclass
class Program:
    output_atoms: MutableMapping[int, Symbol] = field(default_factory=dict)
    facts: List[Rule] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    minimizes: List[Rule] = field(default_factory=list)
    projects: List[Rule] = field(default_factory=list)
    externals: List[Rule] = field(default_factory=list)

    def __repr__(self):
        return f"""Program(output_atoms={repr(self.output_atoms)}, 
                           rules={repr(self.rules)}, 
                           minimizes={repr(self.minimizes)},
                           projects={repr(self.projects)},
                           externals={repr(self.externals)},
                          )"""

    def __str__(self):
        ret = "\n".join(f"{stm}." for stm in sorted(self.facts))
        if ret:
            ret += "\n"
        ret += "\n".join(pretty_str(stm, self.output_atoms) for stm in chain(
            # advanced sorting can be implemented here
            sorted(self.rules),
            sorted(self.minimizes),
            sorted(self.projects),
            sorted(self.externals)
            ))
        return ret
            
class ProgramObserver(Observer):
    def __init__(self, program):
        self._program = program

    def output_atom(self, symbol: Symbol, atom: int) -> None:
        if atom != 0:
            self._program.output_atoms[atom] = symbol
        else:
            self._program.facts.append(symbol)
        

    def rule(self, choice: bool, head: Sequence[int], body: Sequence[int]) -> None:
        # categorization of rules can be handled here
        # there can then be a property than returns the chained rules
        self._program.rules.append(Rule(choice, list(head), list(body)))

    def weight_rule(self, choice: bool, head: Sequence[int], lower_bound: int, body: Sequence[Tuple[int, int]]) -> None:
        self._program.rules.append(WeightRule(choice, list(head), lower_bound, list(body)))

    def project(self, atoms: Sequence[int]) -> None:
        self._program.projects.append(Project(atoms))

    def external(self, atom: int, value: TruthValue) -> None:
        self._program.externals.append(External(atom, value))

    def minimize(self, priority: int, literals: Sequence[Tuple[int,int]]) -> None:
        self._program.minimizes.append(Minimize(priority, list(literals)))


