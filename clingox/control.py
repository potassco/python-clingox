from clingox.ast_repr import ast_repr
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
import textwrap
from typing import Any, Callable, Dict
from typing import Iterable as IterableType
from typing import List, Sequence, Set, Tuple, Union

from clingo import Function, MessageCode, Symbol, TruthValue, Observer, Control, parse_program
from clingo.ast import Sign, AST # pylint: disable=import-error, disable=no-member


@dataclass(eq=True, unsafe_hash=True, order=True)
class Literal:
    atom: Symbol
    sign: Sign

    def __init__(self, atom: Symbol, sign: Union[Sign, bool]):
        self.atom = atom
        if isinstance(sign, bool):
            if sign:
                sign = Sign.NoSign
            else:
                sign = Sign.Negation
        self.sign = sign

    def __repr__(self):
        return repr(self.sign) + repr(self.atom)

    @classmethod
    def from_int(cls, literal, atom_to_symbol: Dict[int, Symbol], symbols: Set[Symbol]) -> 'Literal':
        if abs(literal) in atom_to_symbol:
            lit = atom_to_symbol[abs(literal)]
        else:
            i = abs(literal)
            while True:
                lit = Function('x_' + str(i))
                if lit not in symbols:
                    break
                i += 1
        return cls(lit, literal >= 0)


class ClingoObject(object):
    order: int = 0

    def __lt__(self, other):
        if isinstance(other, ClingoObject):
            return self.order < other.order
        raise Exception("Incomparable type")


@dataclass
class ClingoOutputAtom(ClingoObject):
    symbol: Symbol
    atom: int
    order: int = 0

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return (self.symbol, self.atom) < (other.symbol, other.atom)
        return super().__lt__(other)


class ClingoRuleABC(ClingoObject):
    pass

@dataclass
class ClingoRule(ClingoRuleABC):
    choice: bool
    head: Sequence[int]
    body: Sequence[int]
    order: int = 1

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return (self.choice, self.head, self.body) < (other.choice, other.head, other.body)
        return super().__lt__(other)

@dataclass
class ClingoWeightRule(ClingoRuleABC):
    choice: bool
    head: Sequence[int]
    body: Sequence[Tuple[int, int]]
    lower: int
    order: int = 2

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return (self.choice, self.head, self.body, self.lower) < (other.choice, other.head, other.body, other.lower)
        return super().__lt__(other)

@dataclass
class ClingoProject(ClingoObject):
    atoms: Sequence[int]
    order: int = 3

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return self.atoms < other.atoms
        elif isinstance(other, ClingoObject):
            return self.order < other.order
        raise Exception("Incomparable type")

ClingoExternal   = namedtuple('ClingoExternal', ['atom', 'value'])
ClingoHeuristic  = namedtuple('ClingoHeuristic', ['atom', 'type', 'bias', 'priority', 'condition'])
ClingoMinimize   = namedtuple('ClingoMinimize', ['priority', 'literals'])
ClingoAssume     = namedtuple('ClingoAssume', ['literals'])

@dataclass
class GroundProgram():
    objects : List[ClingoObject]

    def __init__(self, objects: IterableType[ClingoObject] = ()):
        self.objects = list(objects)

    def add_rule(self, choice: bool = False, head: IterableType[int] = (), body: IterableType[int] = ()) -> None:
        self.objects.append(ClingoRule(choice=choice, head=list(head), body=list(body)))

    def add_rules(self, rules: IterableType[ClingoRule]) -> None:
        self.objects.extend(rules)

    def add_project(self, atoms: List[int] = ()) -> None:
        self.objects.append(ClingoProject(list(atoms)))

    def add(self, obj: Union[ClingoObject, IterableType[ClingoObject]]) -> None:
        if isinstance(obj, ClingoObject):
            self.objects.append(obj)
        elif isinstance(obj, Iterable): # pylint: disable=isinstance-second-argument-not-valid-type
            self.objects.extend(obj)

    def pretty(self):
        return PrettyGroundProgram(self.objects)

    def __str__(self):
        return str(self.pretty())

    def __iter__(self):
        return iter(self.objects)


class PrettyClingoOject:
    pass


class PrettyRule(PrettyClingoOject):

    def __init__(self, choice: bool = False, head: IterableType[Literal] = (), body: IterableType[Literal] = ()):
        head = list(head)
        body = list(body)
        self.choice = choice
        self.head   = head
        self.body   = body

    def __repr__(self):
        if self.head:
            head = ', '.join(str(x) for x in self.head)
            if self.choice:
                head = '{' + head + '}'
        else:
            head = ''
        if self.body:
            body = ':- ' + ', '.join(str(x) for x in self.body)
        else:
            body = ''
        if head and body:
            return head + ' ' + body + '.'
        else:
            return head + body + '.'

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return (self.choice, self.head, self.body) < (other.choice, other.head, other.body)
        raise Exception("Incomparable type")

    @classmethod
    def from_rule(cls, rule: ClingoRule, atom_to_symbol, symbols) -> 'PrettyRule':
        return cls._from_rule(rule.choice, rule.head, rule.body, atom_to_symbol, symbols)

    @classmethod
    def _from_rule(cls, choice: bool, head, body, atom_to_symbol, symbols) -> 'PrettyRule':
        head = [Literal.from_int(literal, atom_to_symbol, symbols) for literal in head]
        body = [Literal.from_int(literal, atom_to_symbol, symbols) for literal in body]
        return cls(choice, head, body)


class PrettyProjection(PrettyClingoOject):

    def __init__(self, atoms: IterableType[Literal]):
        if not isinstance(atoms, set):
            atoms = set(atoms)
        self.atoms = atoms

    def __repr__(self):
        atoms = ','.join(repr(atom) for atom in self.atoms)
        if atoms:
            return '#project ' + atoms  + '.'
        else:
            return '#project.'

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return self.atoms < other.atoms
        raise Exception("Incomparable type")

    @classmethod
    def from_projection(cls, projection: ClingoProject, atom_to_symbol, symbols) -> 'PrettyProjection':
        return cls(Literal.from_int(atom, atom_to_symbol, symbols) for atom in projection.atoms)


class PrettyExternal(PrettyClingoOject):

    def __init__(self, atom: Literal, value: TruthValue):
        self.atom  = atom
        self.value = value

    def __repr__(self):
        return '#external ' + repr(self.atom) + ' [' + ('True' if self.value else 'False') + '].'

    @classmethod
    def from_external(self, external: ClingoExternal, atom_to_symbol, symbols) -> 'PrettyExternal':
        return PrettyExternal(Literal.from_int(external.atom, atom_to_symbol, symbols), external.value)

class PrettyGroundProgram():

    def __init__(self, program: IterableType[ClingoObject]):
        self.symbols        : Set[Symbol] = set()
        self.atom_to_symbol : Dict[int, Symbol] = dict()
        self.facts          : List[Symbol]  = list()
        self.cfacts         : List[PrettyRule]  = list()
        self.dfacts         : List[PrettyRule]  = list()
        self.rules          : List[PrettyRule]  = list()
        self.projections    : List[PrettyProjection] = list()
        self.assumtions     : List[ClingoAssume]     = list()
        self.externals      : List[PrettyExternal]   = list()
        self.heuristics     : List[ClingoHeuristic]  = list()
        self.minimizes      : List[ClingoMinimize]   = list()
        self.weight_rules   : List[ClingoWeightRule] = list()
        self.add(program)

    def add(self, program: IterableType[ClingoObject]) -> None:
        output_atoms: List[ClingoOutputAtom] = []
        others: List[ClingoObject] = []

        for obj in program:
            if isinstance(obj, ClingoOutputAtom):
                output_atoms.append(obj)
            else:
                others.append(obj)

        for output_atom in output_atoms:
            self.symbols.add(output_atom.symbol)
            if output_atom.atom != 0:
                self.atom_to_symbol.update({output_atom.atom : output_atom.symbol})
            else:
                self.facts.append(output_atom.symbol)

        self._add(others)
        # discard anonymous facts
        self.facts = list(fact for fact in self.facts if fact in self.symbols)

    def _add(self, obj: Union['ClingoObject', IterableType['ClingoObject']]) -> None:
        if isinstance(obj, ClingoRule):
            self.add_rule(obj)
        if isinstance(obj, ClingoProject):
            self.add_projection(obj)
        elif isinstance(obj, ClingoAssume):
            self.assumtions.append(obj)
        elif isinstance(obj, ClingoExternal):
            self.add_external(obj)
        elif isinstance(obj, ClingoHeuristic):
            self.heuristics.append(obj)
        elif isinstance(obj, ClingoMinimize):
            self.minimizes.append(obj)
        elif isinstance(obj, ClingoWeightRule):
            self.weight_rules.append(obj)
        elif isinstance(obj, Iterable):
            for obj2 in obj:
                self._add(obj2)

    def add_rule(self, rule: ClingoRule) -> PrettyRule:
        pretty_rule = PrettyRule.from_rule(rule, self.atom_to_symbol, self. symbols)
        self.__add_rule(pretty_rule)
        return pretty_rule

    def __add_rule(self, rule: PrettyRule) -> None:
        if not rule.body and len(rule.head) == 1:
            if rule.choice:
                self.cfacts.append(rule)
            else:
                self.facts.append(next(iter(rule.head)).atom)
        elif not rule.body:
            self.dfacts.append(rule)
        else:
            self.rules.append(rule)

    def add_rules(self, rules: IterableType[ClingoRule]) -> None:
        for rule in rules:
            self.add_rule(rule)

    def add_projection(self, projection: ClingoProject) -> None:
        pretty_projection = PrettyProjection.from_projection(projection, self.atom_to_symbol, self. symbols)
        self.projections.append(pretty_projection)

    def add_external(self, external: 'ClingoExternal') -> None:
        pretty_external = PrettyExternal.from_external(external, self.atom_to_symbol, self. symbols)
        self.externals.append(pretty_external)

    def add_project(self, atoms: List[int]) -> None:
        self.add_projection(ClingoProject(atoms))

    def sort(self) -> None:
        self.facts.sort()
        self.cfacts.sort()
        self.dfacts.sort()
        self.rules.sort()
        self.projections.sort()
        self.assumtions.sort()
        self.externals.sort()
        self.heuristics.sort()
        self.minimizes.sort()
        self.weight_rules.sort()

    def as_list(self):
        return [ PrettyRule(False, [fact]) for fact in self.facts ] + \
               self.cfacts + \
               self.dfacts + \
               self.rules + \
               self.weight_rules + \
               self.assumtions + \
               self.assumtions + \
               self.heuristics + \
               self.minimizes + \
               self.projections

    def __repr__(self):
        self.sort()
        facts = '.\n'.join(repr(x) for x in self.facts)
        if facts:
            result = facts + '.'
        else:
            result = ''
        if self.cfacts:
            if result:
                result += '\n\n'
            result += '\n'.join(repr(x) for x in self.cfacts)
        if self.dfacts:
            if result:
                result += '\n\n'
            result += '\n'.join(repr(x) for x in self.dfacts)
        if self.rules:
            if result:
                result += '\n\n'
            result += '\n'.join(repr(x) for x in self.rules)
        if self.weight_rules:
            result += '\n'.join(repr(x) for x in self.weight_rules)
        if self.assumtions:
            if result:
                result += '\n\n'
            result += '\n'.join(repr(x) for x in self.assumtions)
        if self.externals:
            if result:
                result += '\n\n'
            result += '\n'.join(repr(x) for x in self.externals)
        if self.heuristics:
            if result:
                result += '\n\n'
            result += '\n'.join(repr(x) for x in self.heuristics)
        if self.minimizes:
            if result:
                result += '\n\n'
            result += '\n'.join(repr(x) for x in self.minimizes)
        if self.projections:
            if result:
                result += '\n\n'
            result += '\n'.join(repr(x) for x in self.projections)
        return result

class ProgramBuilder():

    def __init__(self, builder, program):
        self.builder = builder
        self.program = program

    def __enter__(self):
        self.builder.__enter__()
        return self

    def __exit__(self, type_, value, traceback):
        return self.builder.__exit__(type_, value, traceback)

    def add(self, statement: AST):
        self.program.append(statement)
        try:
            return self.builder.add(statement)
        except RuntimeError as error:
            if len(error.args) != 1:
                raise error
            if error.args[0] == 'literal expected':
                error.args = ('literal expected, got\n' + textwrap.indent(ast_repr(statement), 13*' '), )
            raise error
        except AttributeError as error:
            if error.args[0] == "'list' object has no attribute 'location'":
                error.args = (error.args[0] + '\n' + textwrap.indent(ast_repr(statement), 13*' '), )
            raise error


class SymbolicBackend():
    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        self.backend.__enter__()
        return self

    def __exit__(self, type_, value, traceback):
        return self.backend.__exit__(type_, value, traceback)

    def add_atom(self, symbol: Symbol) -> None:
        self.backend.add_atom(symbol)

    def add_rule(self, head: IterableType[Symbol] = (), pos_body: IterableType[Symbol] = (), neg_body: IterableType[Symbol] = (), choice: bool = False) -> None:  # pylint: disable=dangerous-default-value
        head = list(self._add_symbols_and_return_their_codes(head))
        body = list(self._add_symbols_and_return_their_codes(pos_body))
        body.extend(self._add_symbols_and_return_their_negated_codes(neg_body))
        return self.backend.add_rule(head, body, choice)

    def add_project(self, symbols: IterableType[Symbol]) -> None:
        atoms = list(self._add_symbols_and_return_their_codes(symbols))
        return self.backend.add_project(atoms)

    def _add_symbol_and_return_its_code(self, symbol: Symbol):
        return self.backend.add_atom(symbol)

    def _add_symbols_and_return_their_codes(self, symbols: IterableType[Symbol]):
        return (self._add_symbol_and_return_its_code(symbol) for symbol in symbols)

    def _add_symbols_and_return_their_negated_codes(self, symbols: IterableType[Symbol]):
        return (-x for x in self._add_symbols_and_return_their_codes(symbols))


class GroundProgramObserver(Observer):

    def __init__(self, program):
        self.program = program

    def rule(self, choice: bool, head: Sequence[int], body: Sequence[int]) -> None:
        self.program.objects.append(ClingoRule(choice=choice, head=head, body=body))

    def output_atom(self, symbol: Symbol, atom: int) -> None:
        self.program.objects.append(ClingoOutputAtom(symbol=symbol, atom=atom))

    def weight_rule(self, choice: bool, head: Sequence[int], lower_bound: int, body: Sequence[Tuple[int, int]]) -> None:
        self.program.objects.append(ClingoWeightRule(choice, head, body, lower_bound))

    def project(self, atoms: Sequence[int]) -> None:
        self.program.objects.append(ClingoProject(atoms))

    def external(self, atom: int, value: TruthValue) -> None:
        self.program.objects.append(ClingoExternal(atom, value))


class Controlx(object):

    def __init__(self, control: Control):
        self.control = control
        self.parsed_program: List[AST] = []
        self.ground_program = GroundProgram()
        self.control.register_observer(GroundProgramObserver(self.ground_program))

    def add_program(self, program: str) -> None:
        with self.builder() as builder:
            parse_program(program, builder.add)

    def builder(self) -> ProgramBuilder:
        return ProgramBuilder(self.control.builder(), self.parsed_program)

    def ground(self, parts: IterableType[Tuple[str, IterableType[Symbol]]], context: Any = None) -> None:
        self.control.ground(parts, context)

    def symbolic_backend(self) -> SymbolicBackend:
        return SymbolicBackend(self.control.backend())

    def facts(self) -> IterableType[Symbol]:
        for symbolic_atom in self.control.symbolic_atoms:
            if symbolic_atom.is_fact:
                yield symbolic_atom.symbol

    def atom_to_symbol_mapping(self) -> Dict[int, Symbol]:
        mapping = dict()
        for symbolic_atom in self.control.symbolic_atoms:
            if not symbolic_atom.is_fact:
                mapping.update({symbolic_atom.literal : symbolic_atom.symbol})
        return mapping

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.control, attr)