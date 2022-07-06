"""
Simple tests for ast manipulation.
"""

# pylint: disable=too-many-lines

from unittest import TestCase
from typing import (
    Callable,
    Container,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import clingo
from clingo import Function
from clingo.ast import (
    AST,
    Location,
    Position,
    Sign,
    Transformer,
    Variable,
    parse_string,
    ASTType,
    AggregateFunction,
)
from .. import ast
from ..ast import (
    ASTPredicate,
    clingo_literal_parser,
    ast_to_dict,
    clingo_term_parser,
    dict_to_ast,
    location_to_str,
    partition_body_literals,
    prefix_symbolic_atoms,
    str_to_location,
    TheoryAtomType,
    TheoryParser,
    filter_body_literals,
    parse_theory,
    reify_symbolic_atoms,
    theory_term_to_literal,
    theory_term_to_term,
)

TERM_TABLE = {"t": clingo_term_parser()}

ATOM_TABLE = {
    ("p", 0): (TheoryAtomType.Head, "t", None),
    ("q", 1): (TheoryAtomType.Body, "t", None),
    ("r", 0): (TheoryAtomType.Directive, "t", (["<"], "t")),
}

TEST_THEORY = """\
#theory test {
    t {
        -  : 3, unary;
        ** : 2, binary, right;
        *  : 1, binary, left;
        /  : 1, binary, left;
        +  : 0, binary, left;
        -  : 0, binary, left
    };
    &p/0 : t, head;
    &q/1 : t, body;
    &r/0 : t, { < }, t, directive
}\
"""

LOC = Location(Position("a", 1, 2), Position("a", 1, 2))


class Extractor(Transformer):
    """
    Simple visitor returning the first theory term in a program.
    """

    # pylint: disable=invalid-name
    atom: Optional[AST]

    def __init__(self):
        self.atom = None

    def visit_TheoryAtom(self, x: AST):
        """
        Extract theory atom.
        """
        self.atom = x
        return x


def theory_atom(s: str) -> AST:
    """
    Convert string to theory term.
    """
    v = Extractor()

    def visit(stm):
        v(stm)

    parse_string(f"{s}.", visit)
    return cast(AST, v.atom)


def last_stm(s: str) -> AST:
    """
    Convert string to rule.
    """
    stm = None

    def set_stm(x):
        nonlocal stm
        stm = x

    parse_string(s, set_stm)

    return cast(AST, stm)


def parse_theory_term(s: str) -> AST:
    """
    Parse the given theory term using a simple parse table for testing.
    """
    return clingo_term_parser()(theory_atom(f"&p {{{s}}}").elements[0].terms[0])


def parse_theory_term_as_literal(s: str) -> AST:
    """
    Parse the given theory term using a simple parse table for testing.
    """
    return clingo_literal_parser()(theory_atom(f"&p {{{s}}}").elements[0].terms[0])


def parse_clingo_term(s: str) -> AST:
    """
    Parse the given theory term using a simple parse table for testing.
    """
    return last_stm(f"p({s}).").head.atom.symbol.arguments[0]


def parse_clingo_literal(s: str) -> AST:
    """
    Parse the given theory term using a simple parse table for testing.
    """
    return last_stm(f"{s}.").head


def parse_atom(s: str, parser: Optional[TheoryParser] = None) -> str:
    """
    Parse the given theory atom using a simple parse table for testing.
    """
    if parser is None:
        parser = TheoryParser(TERM_TABLE, ATOM_TABLE)
    return str(parser(theory_atom(s)))


def parse_stm(s: str, parser: Optional[TheoryParser] = None) -> str:
    """
    Parse the given theory atom using a simple parse table for testing.
    """
    if parser is None:
        parser = TheoryParser(TERM_TABLE, ATOM_TABLE)
    return str(parser(last_stm(s)))


def parse_with(s: str, f: Callable[[AST], AST] = lambda x: x) -> Sequence[str]:
    """
    Parse the given program and apply the given function to it.
    """
    prg: List[str]
    prg = []

    def append(stm):
        nonlocal prg
        ret = f(stm)
        if ret is not None:
            prg.append(str(ret))

    parse_string(s, append)
    return prg


def test_rename(s: str) -> Sequence[str]:
    """
    Parse the given program and rename symbolic atoms in it.
    """
    return parse_with(s, lambda s: prefix_symbolic_atoms(s, "u_"))


def test_reify(
    s: str, f: Callable[[AST], Sequence[AST]] = lambda x: [x], st: bool = False
) -> Sequence[str]:
    """
    Parse the given program and reify symbolic atoms in it.
    """
    return parse_with(s, lambda x: reify_symbolic_atoms(x, "u", f, st))


def test_ast_dict(tc: TestCase, s: str):
    """
    Parse and transform a program to its dictionary representation.
    """
    prg: list = []
    parse_string(s, prg.append)
    ret = [ast_to_dict(x) for x in prg]
    preamble = {
        "ast_type": "Program",
        "location": "<string>:1:1",
        "name": "base",
        "parameters": [],
    }
    tc.assertEqual(ret[0], preamble)
    tc.assertEqual(prg, [dict_to_ast(x) for x in ret])
    return ret[1:]


def get_body(
    stm: AST,
    symbolic_atom_predicate: ASTPredicate = True,
    theory_atom_predicate: ASTPredicate = True,
    aggregate_predicate: ASTPredicate = True,
    conditional_literal_predicate: ASTPredicate = True,
    signs: Container[Sign] = (Sign.NoSign, Sign.Negation, Sign.DoubleNegation),
) -> Iterable[AST]:
    """
    Returns the body of a statement applying optional filters.
    """
    return list(
        filter_body_literals(
            stm.body,
            symbolic_atom_predicate,
            theory_atom_predicate,
            aggregate_predicate,
            conditional_literal_predicate,
            signs,
        )
    )


def partition_body(
    stm: AST,
    symbolic_atom_predicate: ASTPredicate = True,
    theory_atom_predicate: ASTPredicate = True,
    aggregate_predicate: ASTPredicate = True,
    conditional_literal_predicate: ASTPredicate = True,
    signs: Container[Sign] = (Sign.NoSign, Sign.Negation, Sign.DoubleNegation),
) -> Tuple[List[AST], List[AST]]:
    """
    Returns the body of a statement applying optional filters.
    """
    t1, t2 = partition_body_literals(
        stm.body,
        symbolic_atom_predicate,
        theory_atom_predicate,
        aggregate_predicate,
        conditional_literal_predicate,
        signs,
    )
    return list(t1), list(t2)


class TestAST(TestCase):
    """
    Tests for AST manipulation.
    """

    def test_loc(self):
        """
        Test string representation of location.
        """
        loc = LOC
        self.assertEqual(location_to_str(loc), "a:1:2")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc = Location(loc.begin, Position(loc.end.filename, loc.end.line, 4))
        self.assertEqual(location_to_str(loc), "a:1:2-4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc = Location(loc.begin, Position(loc.end.filename, 3, loc.end.column))
        self.assertEqual(location_to_str(loc), "a:1:2-3:4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc = Location(loc.begin, Position("b", loc.end.line, loc.end.column))
        self.assertEqual(location_to_str(loc), "a:1:2-b:3:4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        loc = Location(
            Position(r"a:1:2-3\:", loc.begin.line, loc.begin.column),
            Position("b:1:2-3", loc.end.line, loc.end.column),
        )
        self.assertEqual(location_to_str(loc), r"a\:1\:2-3\\\::1:2-b\:1\:2-3:3:4")
        self.assertEqual(str_to_location(location_to_str(loc)), loc)
        self.assertRaises(RuntimeError, str_to_location, "a:1:2-")

    def test_parse_term(self):
        """
        Test parsing of theory terms.
        """
        self.assertEqual(str(parse_theory_term("1+2")), "+(1,2)")
        self.assertEqual(str(parse_theory_term("1+2+3")), "+(+(1,2),3)")
        self.assertEqual(str(parse_theory_term("1+2*3")), "+(1,*(2,3))")
        self.assertEqual(str(parse_theory_term("1**2**3")), "**(1,**(2,3))")
        self.assertEqual(str(parse_theory_term("-1+2")), "+(-(1),2)")
        self.assertEqual(str(parse_theory_term("f(1+2)+3")), "+(f(+(1,2)),3)")
        self.assertRaises(RuntimeError, parse_theory_term, "1++2")

    def test_parse_atom(self):
        """
        Test parsing of theory atoms.
        """
        self.assertEqual(parse_atom("&p {1+2}"), "&p { +(1,2) }")
        self.assertEqual(parse_atom("&p {1+2+3}"), "&p { +(+(1,2),3) }")
        self.assertEqual(parse_atom("&q(1+2+3) { }"), "&q(((1+2)+3)) { }")
        self.assertEqual(parse_atom("&r { } < 1+2+3"), "&r { } < +(+(1,2),3)")

    def test_parse_atom_occ(self):
        """
        Test parsing of different theory atom types.
        """
        self.assertEqual(parse_stm("&p {1+2}."), "&p { +(1,2) }.")
        self.assertRaises(RuntimeError, parse_stm, ":- &p {1+2}.")
        self.assertRaises(RuntimeError, parse_stm, "&q(1+2+3) { }.")
        self.assertEqual(parse_stm(":- &q(1+2+3) { }."), "#false :- &q(((1+2)+3)) { }.")
        self.assertEqual(parse_stm("&r { } < 1+2+3."), "&r { } < +(+(1,2),3).")
        self.assertRaises(RuntimeError, parse_stm, "&r { } < 1+2+3 :- x.")
        self.assertRaises(RuntimeError, parse_stm, ":- &r { } < 1+2+3.")

    def test_parse_theory(self):
        """
        Test creating parsers from theory definitions.
        """
        with self.assertRaisesRegex(ValueError, "no theory definition found"):
            parse_theory("#program base")
        with self.assertRaisesRegex(ValueError, "multiple theory definitions"):
            parse_theory(TEST_THEORY + "." + TEST_THEORY)
        parser = parse_theory(TEST_THEORY)

        def pawp(s):
            return parse_atom(s, parser)

        def prwp(s):
            return parse_stm(s, parser)

        self.assertEqual(parse_atom("&p {1+2}", pawp), "&p { +(1,2) }")
        self.assertEqual(parse_atom("&p {1+2+3}", pawp), "&p { +(+(1,2),3) }")
        self.assertEqual(parse_atom("&q(1+2+3) { }", pawp), "&q(((1+2)+3)) { }")
        self.assertEqual(parse_atom("&r { } < 1+2+3", pawp), "&r { } < +(+(1,2),3)")

        self.assertEqual(prwp("&p {1+2}."), "&p { +(1,2) }.")
        self.assertEqual(prwp("#show x : &q(0) {1+2}."), "#show x : &q(0) { +(1,2) }.")
        self.assertEqual(prwp(":~ &q(0) {1+2}. [0]"), ":~ &q(0) { +(1,2) }. [0@0]")
        self.assertEqual(
            prwp("#edge (u, v) : &q(0) {1+2}."), "#edge (u,v) : &q(0) { +(1,2) }."
        )
        self.assertEqual(
            prwp("#heuristic a : &q(0) {1+2}. [sign,true]"),
            "#heuristic a : &q(0) { +(1,2) }. [sign@0,true]",
        )
        self.assertEqual(
            prwp("#project a : &q(0) {1+2}."), "#project a : &q(0) { +(1,2) }."
        )
        self.assertRaises(RuntimeError, prwp, ":- &p {1+2}.")
        self.assertRaises(RuntimeError, prwp, "&q(1+2+3) { }.")
        self.assertEqual(prwp(":- &q(1+2+3) { }."), "#false :- &q(((1+2)+3)) { }.")
        self.assertEqual(prwp("&r { } < 1+2+3."), "&r { } < +(+(1,2),3).")
        self.assertRaises(RuntimeError, prwp, "&r { } < 1+2+3 :- x.")
        self.assertRaises(RuntimeError, prwp, ":- &r { } < 1+2+3.")
        self.assertRaises(RuntimeError, prwp, "&s(1+2+3) { }.")
        self.assertRaises(RuntimeError, prwp, "&p { } <= 3.")
        self.assertRaises(RuntimeError, prwp, "&r { } <= 3.")

    def test_rename(self):
        """
        Test renaming symbolic atoms.
        """
        self.assertEqual(
            test_rename("a :- b(X,Y), not c(f(3,b))."),
            ["#program base.", "u_a :- u_b(X,Y); not u_c(f(3,b))."],
        )
        sym = clingo.ast.SymbolicAtom(
            clingo.ast.UnaryOperation(
                LOC,
                clingo.ast.UnaryOperator.Minus,
                clingo.ast.Function(LOC, "a", [], 0),
            )
        )
        self.assertEqual(str(prefix_symbolic_atoms(sym, "u_")), "-u_a")
        self.assertEqual(
            test_rename("-a :- -b(X,Y), not -c(f(3,b))."),
            ["#program base.", "-u_a :- -u_b(X,Y); not -u_c(f(3,b))."],
        )
        sym = ast.SymbolicAtom(ast.SymbolicTerm(LOC, Function("a", [Function("b")])))
        self.assertEqual(str(prefix_symbolic_atoms(sym, "u_")), "u_a(b)")
        sym = ast.SymbolicAtom(Variable(LOC, "B"))
        self.assertEqual(prefix_symbolic_atoms(sym, "u"), sym)

    def test_reify(self):
        """
        Test reifying symbolic atoms.
        """
        self.assertEqual(test_reify("a."), ["#program base.", "u(a)."])
        self.assertEqual(
            test_reify("a :- b(X,Y), not c(f(3,b))."),
            ["#program base.", "u(a) :- u(b(X,Y)); not u(c(f(3,b)))."],
        )
        sym = clingo.ast.SymbolicAtom(
            clingo.ast.UnaryOperation(
                LOC,
                clingo.ast.UnaryOperator.Minus,
                clingo.ast.Function(LOC, "a", [], 0),
            )
        )
        self.assertEqual(str(reify_symbolic_atoms(sym, "u")), "-u(a)")
        self.assertEqual(
            str(reify_symbolic_atoms(sym, "u", reify_strong_negation=True)), "u(-a)"
        )
        self.assertEqual(
            test_reify("-a :- -b(X,Y), not -c(f(3,b))."),
            ["#program base.", "-u(a) :- -u(b(X,Y)); not -u(c(f(3,b)))."],
        )
        self.assertEqual(
            test_reify(
                "-a :- b(X,Y), not -c(f(3,b)). a :- -b(X,Y), not c(f(3,b)).", st=True
            ),
            [
                "#program base.",
                "u(-a) :- u(b(X,Y)); not u(-c(f(3,b))).",
                "u(a) :- u(-b(X,Y)); not u(c(f(3,b))).",
            ],
        )
        self.assertEqual(
            test_reify(
                "a :- b(X,Y), not c(f(3,b)).",
                f=lambda x: [x, Variable(LOC, "T"), Variable(LOC, "I")],
            ),
            ["#program base.", "u(a,T,I) :- u(b(X,Y),T,I); not u(c(f(3,b)),T,I)."],
        )
        self.assertEqual(
            test_reify("-a :- -b(X,Y), &theory(X){ p(X): q(X), -r(X) }."),
            [
                "#program base.",
                "-u(a) :- -u(b(X,Y)); &theory(X) { p(X): u(q(X)), -u(r(X)) }.",
            ],
        )
        self.assertEqual(
            test_reify("-a :- -b(X,Y), &theory(X){ p(X): q(X), not r(X) }."),
            [
                "#program base.",
                "-u(a) :- -u(b(X,Y)); &theory(X) { p(X): u(q(X)), not u(r(X)) }.",
            ],
        )

        def fun(x):
            return [Variable(LOC, "T"), x, Variable(LOC, "I")]

        self.assertEqual(
            test_reify("a :- -b(X,Y), not c(f(3,b)).", f=fun, st=True),
            ["#program base.", "u(T,a,I) :- u(T,-b(X,Y),I); not u(T,c(f(3,b)),I)."],
        )

        self.assertEqual(
            test_reify("a :- -b(X,Y), not c(f(3,b)).", f=fun, st=True),
            ["#program base.", "u(T,a,I) :- u(T,-b(X,Y),I); not u(T,c(f(3,b)),I)."],
        )

        sym = ast.SymbolicAtom(ast.SymbolicTerm(LOC, Function("a", [Function("b")])))
        self.assertEqual(str(reify_symbolic_atoms(sym, "u")), "u(a(b))")
        sym = ast.SymbolicAtom(Variable(LOC, "B"))
        self.assertEqual(prefix_symbolic_atoms(sym, "u"), sym)

    def test_encode_term(self):
        """
        Test encoding of terms in AST.
        """
        self.assertEqual(
            test_ast_dict(self, "a(1)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-6",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-5",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-5",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:3-4",
                                        "symbol": "1",
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(X)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-6",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-5",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-5",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "Variable",
                                        "location": "<string>:1:3-4",
                                        "name": "X",
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(-1)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-7",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-6",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-6",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "UnaryOperation",
                                        "location": "<string>:1:3-5",
                                        "operator_type": 0,
                                        "argument": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:4-5",
                                            "symbol": "1",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(~1)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-7",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-6",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-6",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "UnaryOperation",
                                        "location": "<string>:1:3-5",
                                        "operator_type": 1,
                                        "argument": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:4-5",
                                            "symbol": "1",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(|1|)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-7",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "UnaryOperation",
                                        "location": "<string>:1:3-6",
                                        "operator_type": 2,
                                        "argument": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:4-5",
                                            "symbol": "1",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1+2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-7",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "BinaryOperation",
                                        "location": "<string>:1:3-6",
                                        "operator_type": 3,
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:5-6",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1-2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-7",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "BinaryOperation",
                                        "location": "<string>:1:3-6",
                                        "operator_type": 4,
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:5-6",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1*2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-7",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "BinaryOperation",
                                        "location": "<string>:1:3-6",
                                        "operator_type": 5,
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:5-6",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1/2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-7",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "BinaryOperation",
                                        "location": "<string>:1:3-6",
                                        "operator_type": 6,
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:5-6",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1\\2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-7",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "BinaryOperation",
                                        "location": "<string>:1:3-6",
                                        "operator_type": 7,
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:5-6",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1**2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-9",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-8",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-8",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "BinaryOperation",
                                        "location": "<string>:1:3-7",
                                        "operator_type": 8,
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:6-7",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1^2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-7",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "BinaryOperation",
                                        "location": "<string>:1:3-6",
                                        "operator_type": 0,
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:5-6",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1?2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-7",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "BinaryOperation",
                                        "location": "<string>:1:3-6",
                                        "operator_type": 1,
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:5-6",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1&2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-7",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "BinaryOperation",
                                        "location": "<string>:1:3-6",
                                        "operator_type": 2,
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:5-6",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1..2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-9",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-8",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-8",
                                "name": "a",
                                "arguments": [
                                    {
                                        "ast_type": "Interval",
                                        "location": "<string>:1:3-7",
                                        "left": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:3-4",
                                            "symbol": "1",
                                        },
                                        "right": {
                                            "ast_type": "SymbolicTerm",
                                            "location": "<string>:1:6-7",
                                            "symbol": "2",
                                        },
                                    }
                                ],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a(1;2)."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Pool",
                                "location": "<string>:1:1-7",
                                "arguments": [
                                    {
                                        "ast_type": "Function",
                                        "location": "<string>:1:1-7",
                                        "name": "a",
                                        "arguments": [
                                            {
                                                "ast_type": "SymbolicTerm",
                                                "location": "<string>:1:3-4",
                                                "symbol": "1",
                                            }
                                        ],
                                        "external": 0,
                                    },
                                    {
                                        "ast_type": "Function",
                                        "location": "<string>:1:1-7",
                                        "name": "a",
                                        "arguments": [
                                            {
                                                "ast_type": "SymbolicTerm",
                                                "location": "<string>:1:5-6",
                                                "symbol": "2",
                                            }
                                        ],
                                        "external": 0,
                                    },
                                ],
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )

    def test_encode_literal(self):
        """
        Tests for converting between python and ast representation of literals.
        """
        # Note: tests are simply skipped for older clingo versions
        if clingo.version() < (5, 6, 0):
            return  # nocoverage
        self.assertEqual(
            test_ast_dict(self, "a."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-3",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-2",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-2",
                                "name": "a",
                                "arguments": [],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "not a."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-7",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-6",
                        "sign": 1,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:5-6",
                                "name": "a",
                                "arguments": [],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "not not a."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-11",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-10",
                        "sign": 2,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:9-10",
                                "name": "a",
                                "arguments": [],
                                "external": 0,
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a <= b."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "Comparison",
                            "term": {
                                "ast_type": "SymbolicTerm",
                                "location": "<string>:1:1-2",
                                "symbol": "a",
                            },
                            "guards": [
                                {
                                    "ast_type": "Guard",
                                    "comparison": 2,
                                    "term": {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:6-7",
                                        "symbol": "b",
                                    },
                                }
                            ],
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a < b."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-7",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-6",
                        "sign": 0,
                        "atom": {
                            "ast_type": "Comparison",
                            "term": {
                                "ast_type": "SymbolicTerm",
                                "location": "<string>:1:1-2",
                                "symbol": "a",
                            },
                            "guards": [
                                {
                                    "ast_type": "Guard",
                                    "comparison": 1,
                                    "term": {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:5-6",
                                        "symbol": "b",
                                    },
                                }
                            ],
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a >= b."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "Comparison",
                            "term": {
                                "ast_type": "SymbolicTerm",
                                "location": "<string>:1:1-2",
                                "symbol": "a",
                            },
                            "guards": [
                                {
                                    "ast_type": "Guard",
                                    "comparison": 3,
                                    "term": {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:6-7",
                                        "symbol": "b",
                                    },
                                }
                            ],
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a > b."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-7",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-6",
                        "sign": 0,
                        "atom": {
                            "ast_type": "Comparison",
                            "term": {
                                "ast_type": "SymbolicTerm",
                                "location": "<string>:1:1-2",
                                "symbol": "a",
                            },
                            "guards": [
                                {
                                    "ast_type": "Guard",
                                    "comparison": 0,
                                    "term": {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:5-6",
                                        "symbol": "b",
                                    },
                                }
                            ],
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a = b."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-7",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-6",
                        "sign": 0,
                        "atom": {
                            "ast_type": "Comparison",
                            "term": {
                                "ast_type": "SymbolicTerm",
                                "location": "<string>:1:1-2",
                                "symbol": "a",
                            },
                            "guards": [
                                {
                                    "ast_type": "Guard",
                                    "comparison": 5,
                                    "term": {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:5-6",
                                        "symbol": "b",
                                    },
                                }
                            ],
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a != b."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-7",
                        "sign": 0,
                        "atom": {
                            "ast_type": "Comparison",
                            "term": {
                                "ast_type": "SymbolicTerm",
                                "location": "<string>:1:1-2",
                                "symbol": "a",
                            },
                            "guards": [
                                {
                                    "ast_type": "Guard",
                                    "comparison": 4,
                                    "term": {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:6-7",
                                        "symbol": "b",
                                    },
                                }
                            ],
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "a : b."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-7",
                    "head": {
                        "ast_type": "Disjunction",
                        "location": "<string>:1:1-6",
                        "elements": [
                            {
                                "ast_type": "ConditionalLiteral",
                                "location": "<string>:1:1-2",
                                "literal": {
                                    "ast_type": "Literal",
                                    "location": "<string>:1:1-2",
                                    "sign": 0,
                                    "atom": {
                                        "ast_type": "SymbolicAtom",
                                        "symbol": {
                                            "ast_type": "Function",
                                            "location": "<string>:1:1-2",
                                            "name": "a",
                                            "arguments": [],
                                            "external": 0,
                                        },
                                    },
                                },
                                "condition": [
                                    {
                                        "ast_type": "Literal",
                                        "location": "<string>:1:5-6",
                                        "sign": 0,
                                        "atom": {
                                            "ast_type": "SymbolicAtom",
                                            "symbol": {
                                                "ast_type": "Function",
                                                "location": "<string>:1:5-6",
                                                "name": "b",
                                                "arguments": [],
                                                "external": 0,
                                            },
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, ":- a : b."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-10",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-10",
                        "sign": 0,
                        "atom": {"ast_type": "BooleanConstant", "value": 0},
                    },
                    "body": [
                        {
                            "ast_type": "ConditionalLiteral",
                            "location": "<string>:1:4-9",
                            "literal": {
                                "ast_type": "Literal",
                                "location": "<string>:1:4-5",
                                "sign": 0,
                                "atom": {
                                    "ast_type": "SymbolicAtom",
                                    "symbol": {
                                        "ast_type": "Function",
                                        "location": "<string>:1:4-5",
                                        "name": "a",
                                        "arguments": [],
                                        "external": 0,
                                    },
                                },
                            },
                            "condition": [
                                {
                                    "ast_type": "Literal",
                                    "location": "<string>:1:8-9",
                                    "sign": 0,
                                    "atom": {
                                        "ast_type": "SymbolicAtom",
                                        "symbol": {
                                            "ast_type": "Function",
                                            "location": "<string>:1:8-9",
                                            "name": "b",
                                            "arguments": [],
                                            "external": 0,
                                        },
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#sum {1:a:b} <= 2."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-19",
                    "head": {
                        "ast_type": "HeadAggregate",
                        "location": "<string>:1:1-18",
                        "left_guard": {
                            "ast_type": "Guard",
                            "comparison": 3,
                            "term": {
                                "ast_type": "SymbolicTerm",
                                "location": "<string>:1:17-18",
                                "symbol": "2",
                            },
                        },
                        "function": 1,
                        "elements": [
                            {
                                "ast_type": "HeadAggregateElement",
                                "terms": [
                                    {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:7-8",
                                        "symbol": "1",
                                    }
                                ],
                                "condition": {
                                    "ast_type": "ConditionalLiteral",
                                    "location": "<string>:1:9-10",
                                    "literal": {
                                        "ast_type": "Literal",
                                        "location": "<string>:1:9-10",
                                        "sign": 0,
                                        "atom": {
                                            "ast_type": "SymbolicAtom",
                                            "symbol": {
                                                "ast_type": "Function",
                                                "location": "<string>:1:9-10",
                                                "name": "a",
                                                "arguments": [],
                                                "external": 0,
                                            },
                                        },
                                    },
                                    "condition": [
                                        {
                                            "ast_type": "Literal",
                                            "location": "<string>:1:11-12",
                                            "sign": 0,
                                            "atom": {
                                                "ast_type": "SymbolicAtom",
                                                "symbol": {
                                                    "ast_type": "Function",
                                                    "location": "<string>:1:11-12",
                                                    "name": "b",
                                                    "arguments": [],
                                                    "external": 0,
                                                },
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                        "right_guard": None,
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, ":- #sum {1:b} <= 2."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-20",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-20",
                        "sign": 0,
                        "atom": {"ast_type": "BooleanConstant", "value": 0},
                    },
                    "body": [
                        {
                            "ast_type": "Literal",
                            "location": "<string>:1:4-19",
                            "sign": 0,
                            "atom": {
                                "ast_type": "BodyAggregate",
                                "location": "<string>:1:4-19",
                                "left_guard": {
                                    "ast_type": "Guard",
                                    "comparison": 3,
                                    "term": {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:18-19",
                                        "symbol": "2",
                                    },
                                },
                                "function": 1,
                                "elements": [
                                    {
                                        "ast_type": "BodyAggregateElement",
                                        "terms": [
                                            {
                                                "ast_type": "SymbolicTerm",
                                                "location": "<string>:1:10-11",
                                                "symbol": "1",
                                            }
                                        ],
                                        "condition": [
                                            {
                                                "ast_type": "Literal",
                                                "location": "<string>:1:12-13",
                                                "sign": 0,
                                                "atom": {
                                                    "ast_type": "SymbolicAtom",
                                                    "symbol": {
                                                        "ast_type": "Function",
                                                        "location": "<string>:1:12-13",
                                                        "name": "b",
                                                        "arguments": [],
                                                        "external": 0,
                                                    },
                                                },
                                            }
                                        ],
                                    }
                                ],
                                "right_guard": None,
                            },
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#count {}."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-11",
                    "head": {
                        "ast_type": "HeadAggregate",
                        "location": "<string>:1:1-10",
                        "left_guard": None,
                        "function": 0,
                        "elements": [],
                        "right_guard": None,
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#min {}."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-9",
                    "head": {
                        "ast_type": "HeadAggregate",
                        "location": "<string>:1:1-8",
                        "left_guard": None,
                        "function": 3,
                        "elements": [],
                        "right_guard": None,
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#max {}."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-9",
                    "head": {
                        "ast_type": "HeadAggregate",
                        "location": "<string>:1:1-8",
                        "left_guard": None,
                        "function": 4,
                        "elements": [],
                        "right_guard": None,
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#sum+ {}."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-10",
                    "head": {
                        "ast_type": "HeadAggregate",
                        "location": "<string>:1:1-9",
                        "left_guard": None,
                        "function": 2,
                        "elements": [],
                        "right_guard": None,
                    },
                    "body": [],
                }
            ],
        )

    def test_encode_theory(self):
        """
        Tests for converting between python and ast representation of theory
        releated constructs.
        """
        self.assertEqual(
            test_ast_dict(self, "#theory t { }."),
            [
                {
                    "ast_type": "TheoryDefinition",
                    "location": "<string>:1:1-15",
                    "name": "t",
                    "terms": [],
                    "atoms": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#theory t { t { + : 1, unary } }."),
            [
                {
                    "ast_type": "TheoryDefinition",
                    "location": "<string>:1:1-34",
                    "name": "t",
                    "terms": [
                        {
                            "ast_type": "TheoryTermDefinition",
                            "location": "<string>:1:13-31",
                            "name": "t",
                            "operators": [
                                {
                                    "ast_type": "TheoryOperatorDefinition",
                                    "location": "<string>:1:17-29",
                                    "name": "+",
                                    "priority": 1,
                                    "operator_type": 0,
                                }
                            ],
                        }
                    ],
                    "atoms": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#theory t { t { + : 1, binary, left } }."),
            [
                {
                    "ast_type": "TheoryDefinition",
                    "location": "<string>:1:1-41",
                    "name": "t",
                    "terms": [
                        {
                            "ast_type": "TheoryTermDefinition",
                            "location": "<string>:1:13-38",
                            "name": "t",
                            "operators": [
                                {
                                    "ast_type": "TheoryOperatorDefinition",
                                    "location": "<string>:1:17-36",
                                    "name": "+",
                                    "priority": 1,
                                    "operator_type": 1,
                                }
                            ],
                        }
                    ],
                    "atoms": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#theory t { t { + : 1, binary, right } }."),
            [
                {
                    "ast_type": "TheoryDefinition",
                    "location": "<string>:1:1-42",
                    "name": "t",
                    "terms": [
                        {
                            "ast_type": "TheoryTermDefinition",
                            "location": "<string>:1:13-39",
                            "name": "t",
                            "operators": [
                                {
                                    "ast_type": "TheoryOperatorDefinition",
                                    "location": "<string>:1:17-37",
                                    "name": "+",
                                    "priority": 1,
                                    "operator_type": 2,
                                }
                            ],
                        }
                    ],
                    "atoms": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#theory t { &p/0 : t, any }."),
            [
                {
                    "ast_type": "TheoryDefinition",
                    "location": "<string>:1:1-29",
                    "name": "t",
                    "terms": [],
                    "atoms": [
                        {
                            "ast_type": "TheoryAtomDefinition",
                            "location": "<string>:1:13-26",
                            "atom_type": 2,
                            "name": "p",
                            "arity": 0,
                            "term": "t",
                            "guard": None,
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#theory t { &p/0 : t, head }."),
            [
                {
                    "ast_type": "TheoryDefinition",
                    "location": "<string>:1:1-30",
                    "name": "t",
                    "terms": [],
                    "atoms": [
                        {
                            "ast_type": "TheoryAtomDefinition",
                            "location": "<string>:1:13-27",
                            "atom_type": 0,
                            "name": "p",
                            "arity": 0,
                            "term": "t",
                            "guard": None,
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#theory t { &p/1 : t, body }."),
            [
                {
                    "ast_type": "TheoryDefinition",
                    "location": "<string>:1:1-30",
                    "name": "t",
                    "terms": [],
                    "atoms": [
                        {
                            "ast_type": "TheoryAtomDefinition",
                            "location": "<string>:1:13-27",
                            "atom_type": 1,
                            "name": "p",
                            "arity": 1,
                            "term": "t",
                            "guard": None,
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#theory t { &p/2 : t, { < }, t, directive }."),
            [
                {
                    "ast_type": "TheoryDefinition",
                    "location": "<string>:1:1-45",
                    "name": "t",
                    "terms": [],
                    "atoms": [
                        {
                            "ast_type": "TheoryAtomDefinition",
                            "location": "<string>:1:13-42",
                            "atom_type": 3,
                            "name": "p",
                            "arity": 2,
                            "term": "t",
                            "guard": {
                                "ast_type": "TheoryGuardDefinition",
                                "operators": ["<"],
                                "term": "t",
                            },
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "&p { }."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "TheoryAtom",
                        "location": "<string>:1:2-3",
                        "term": {
                            "ast_type": "Function",
                            "location": "<string>:1:2-3",
                            "name": "p",
                            "arguments": [],
                            "external": 0,
                        },
                        "elements": [],
                        "guard": None,
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, ":- &p { }."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-11",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-11",
                        "sign": 0,
                        "atom": {"ast_type": "BooleanConstant", "value": 0},
                    },
                    "body": [
                        {
                            "ast_type": "Literal",
                            "location": "<string>:1:4-10",
                            "sign": 0,
                            "atom": {
                                "ast_type": "TheoryAtom",
                                "location": "<string>:1:5-6",
                                "term": {
                                    "ast_type": "Function",
                                    "location": "<string>:1:5-6",
                                    "name": "p",
                                    "arguments": [],
                                    "external": 0,
                                },
                                "elements": [],
                                "guard": None,
                            },
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "&p { } > 2."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-12",
                    "head": {
                        "ast_type": "TheoryAtom",
                        "location": "<string>:1:2-3",
                        "term": {
                            "ast_type": "Function",
                            "location": "<string>:1:2-3",
                            "name": "p",
                            "arguments": [],
                            "external": 0,
                        },
                        "elements": [],
                        "guard": {
                            "ast_type": "TheoryGuard",
                            "operator_name": ">",
                            "term": {
                                "ast_type": "SymbolicTerm",
                                "location": "<string>:1:10-11",
                                "symbol": "2",
                            },
                        },
                    },
                    "body": [],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "&p { a,b: q }."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-15",
                    "head": {
                        "ast_type": "TheoryAtom",
                        "location": "<string>:1:2-3",
                        "term": {
                            "ast_type": "Function",
                            "location": "<string>:1:2-3",
                            "name": "p",
                            "arguments": [],
                            "external": 0,
                        },
                        "elements": [
                            {
                                "ast_type": "TheoryAtomElement",
                                "terms": [
                                    {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:6-7",
                                        "symbol": "a",
                                    },
                                    {
                                        "ast_type": "SymbolicTerm",
                                        "location": "<string>:1:8-9",
                                        "symbol": "b",
                                    },
                                ],
                                "condition": [
                                    {
                                        "ast_type": "Literal",
                                        "location": "<string>:1:11-12",
                                        "sign": 0,
                                        "atom": {
                                            "ast_type": "SymbolicAtom",
                                            "symbol": {
                                                "ast_type": "Function",
                                                "location": "<string>:1:11-12",
                                                "name": "q",
                                                "arguments": [],
                                                "external": 0,
                                            },
                                        },
                                    }
                                ],
                            }
                        ],
                        "guard": None,
                    },
                    "body": [],
                }
            ],
        )

    def test_encode_statement(self):
        """
        Tests for converting between python and ast representation of statements.
        """
        # Note: tests are simply skipped for older clingo versions
        if clingo.version() < (5, 6, 0):
            return  # nocoverage
        self.assertEqual(
            test_ast_dict(self, "a :- b."),
            [
                {
                    "ast_type": "Rule",
                    "location": "<string>:1:1-8",
                    "head": {
                        "ast_type": "Literal",
                        "location": "<string>:1:1-2",
                        "sign": 0,
                        "atom": {
                            "ast_type": "SymbolicAtom",
                            "symbol": {
                                "ast_type": "Function",
                                "location": "<string>:1:1-2",
                                "name": "a",
                                "arguments": [],
                                "external": 0,
                            },
                        },
                    },
                    "body": [
                        {
                            "ast_type": "Literal",
                            "location": "<string>:1:6-7",
                            "sign": 0,
                            "atom": {
                                "ast_type": "SymbolicAtom",
                                "symbol": {
                                    "ast_type": "Function",
                                    "location": "<string>:1:6-7",
                                    "name": "b",
                                    "arguments": [],
                                    "external": 0,
                                },
                            },
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#defined x/0."),
            [
                {
                    "ast_type": "Defined",
                    "location": "<string>:1:1-14",
                    "name": "x",
                    "arity": 0,
                    "positive": True,
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#show a : b."),
            [
                {
                    "ast_type": "ShowTerm",
                    "location": "<string>:1:1-13",
                    "term": {
                        "ast_type": "SymbolicTerm",
                        "location": "<string>:1:7-8",
                        "symbol": "a",
                    },
                    "body": [
                        {
                            "ast_type": "Literal",
                            "location": "<string>:1:11-12",
                            "sign": 0,
                            "atom": {
                                "ast_type": "SymbolicAtom",
                                "symbol": {
                                    "ast_type": "Function",
                                    "location": "<string>:1:11-12",
                                    "name": "b",
                                    "arguments": [],
                                    "external": 0,
                                },
                            },
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#show a/0."),
            [
                {
                    "ast_type": "ShowSignature",
                    "location": "<string>:1:1-11",
                    "name": "a",
                    "arity": 0,
                    "positive": True,
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#minimize { 1@2,a : b }."),
            [
                {
                    "ast_type": "Minimize",
                    "location": "<string>:1:13-22",
                    "weight": {
                        "ast_type": "SymbolicTerm",
                        "location": "<string>:1:13-14",
                        "symbol": "1",
                    },
                    "priority": {
                        "ast_type": "SymbolicTerm",
                        "location": "<string>:1:15-16",
                        "symbol": "2",
                    },
                    "terms": [
                        {
                            "ast_type": "SymbolicTerm",
                            "location": "<string>:1:17-18",
                            "symbol": "a",
                        }
                    ],
                    "body": [
                        {
                            "ast_type": "Literal",
                            "location": "<string>:1:21-22",
                            "sign": 0,
                            "atom": {
                                "ast_type": "SymbolicAtom",
                                "symbol": {
                                    "ast_type": "Function",
                                    "location": "<string>:1:21-22",
                                    "name": "b",
                                    "arguments": [],
                                    "external": 0,
                                },
                            },
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#script (python) blub! #end."),
            [
                {
                    "ast_type": "Script",
                    "location": "<string>:1:1-29",
                    "name": "python",
                    "code": "blub!",
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#script (lua) blub! #end."),
            [
                {
                    "ast_type": "Script",
                    "location": "<string>:1:1-26",
                    "code": "blub!",
                    "name": "lua",
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#program x(y)."),
            [
                {
                    "ast_type": "Program",
                    "location": "<string>:1:1-15",
                    "name": "x",
                    "parameters": [
                        {"ast_type": "Id", "location": "<string>:1:12-13", "name": "y"}
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#project a/0."),
            [
                {
                    "ast_type": "ProjectSignature",
                    "location": "<string>:1:1-14",
                    "name": "a",
                    "arity": 0,
                    "positive": True,
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#project a : b."),
            [
                {
                    "ast_type": "ProjectAtom",
                    "location": "<string>:1:1-16",
                    "atom": {
                        "ast_type": "SymbolicAtom",
                        "symbol": {
                            "ast_type": "Function",
                            "location": "<string>:1:10-11",
                            "name": "a",
                            "arguments": [],
                            "external": 0,
                        },
                    },
                    "body": [
                        {
                            "ast_type": "Literal",
                            "location": "<string>:1:14-15",
                            "sign": 0,
                            "atom": {
                                "ast_type": "SymbolicAtom",
                                "symbol": {
                                    "ast_type": "Function",
                                    "location": "<string>:1:14-15",
                                    "name": "b",
                                    "arguments": [],
                                    "external": 0,
                                },
                            },
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#external x : y. [X]"),
            [
                {
                    "ast_type": "External",
                    "location": "<string>:1:1-21",
                    "atom": {
                        "ast_type": "SymbolicAtom",
                        "symbol": {
                            "ast_type": "Function",
                            "location": "<string>:1:11-12",
                            "name": "x",
                            "arguments": [],
                            "external": 0,
                        },
                    },
                    "body": [
                        {
                            "ast_type": "Literal",
                            "location": "<string>:1:15-16",
                            "sign": 0,
                            "atom": {
                                "ast_type": "SymbolicAtom",
                                "symbol": {
                                    "ast_type": "Function",
                                    "location": "<string>:1:15-16",
                                    "name": "y",
                                    "arguments": [],
                                    "external": 0,
                                },
                            },
                        }
                    ],
                    "external_type": {
                        "ast_type": "Variable",
                        "location": "<string>:1:19-20",
                        "name": "X",
                    },
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#edge (u,v) : b."),
            [
                {
                    "ast_type": "Edge",
                    "location": "<string>:1:1-17",
                    "node_u": {
                        "ast_type": "SymbolicTerm",
                        "location": "<string>:1:8-9",
                        "symbol": "u",
                    },
                    "node_v": {
                        "ast_type": "SymbolicTerm",
                        "location": "<string>:1:10-11",
                        "symbol": "v",
                    },
                    "body": [
                        {
                            "ast_type": "Literal",
                            "location": "<string>:1:15-16",
                            "sign": 0,
                            "atom": {
                                "ast_type": "SymbolicAtom",
                                "symbol": {
                                    "ast_type": "Function",
                                    "location": "<string>:1:15-16",
                                    "name": "b",
                                    "arguments": [],
                                    "external": 0,
                                },
                            },
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            test_ast_dict(self, "#heuristic a : b. [p,X]"),
            [
                {
                    "ast_type": "Heuristic",
                    "location": "<string>:1:1-24",
                    "atom": {
                        "ast_type": "SymbolicAtom",
                        "symbol": {
                            "ast_type": "Function",
                            "location": "<string>:1:12-13",
                            "name": "a",
                            "arguments": [],
                            "external": 0,
                        },
                    },
                    "body": [
                        {
                            "ast_type": "Literal",
                            "location": "<string>:1:16-17",
                            "sign": 0,
                            "atom": {
                                "ast_type": "SymbolicAtom",
                                "symbol": {
                                    "ast_type": "Function",
                                    "location": "<string>:1:16-17",
                                    "name": "b",
                                    "arguments": [],
                                    "external": 0,
                                },
                            },
                        }
                    ],
                    "bias": {
                        "ast_type": "SymbolicTerm",
                        "location": "<string>:1:20-21",
                        "symbol": "p",
                    },
                    "priority": {
                        "ast_type": "SymbolicTerm",
                        "location": "<string>:1:1-24",
                        "symbol": "0",
                    },
                    "modifier": {
                        "ast_type": "Variable",
                        "location": "<string>:1:22-23",
                        "name": "X",
                    },
                }
            ],
        )

    def test_dict_ast_error(self):
        """
        Test error conditions when converting between ast and dict.
        """
        self.assertRaises(
            RuntimeError, dict_to_ast, {"ast_type": "Rule", "body": set()}
        )

    def helper_body_elements(
        self,
        stm: str,
        body: Sequence[str],
        signs: Container[Sign] = (Sign.NoSign,),
        symbolic_atom_predicate: Union[Callable[[AST], bool], bool] = True,
        theory_atom_predicate: Union[Callable[[AST], bool], bool] = True,
        aggregate_predicate: Union[Callable[[AST], bool], bool] = True,
        conditional_literal_predicate: Union[Callable[[AST], bool], bool] = True,
    ):
        """
        Helper for testing get_body.
        """
        parsed_stm = last_stm(stm)
        res = get_body(
            parsed_stm,
            symbolic_atom_predicate,
            theory_atom_predicate,
            aggregate_predicate,
            conditional_literal_predicate,
            signs,
        )
        self.assertListEqual(sorted(body), sorted(str(s) for s in res))
        res_true, res_false = partition_body(
            parsed_stm,
            symbolic_atom_predicate,
            theory_atom_predicate,
            aggregate_predicate,
            conditional_literal_predicate,
            signs,
        )
        self.assertListEqual(sorted(body), sorted(str(s) for s in res_true))
        full_body = get_body(
            parsed_stm,
            True,
            True,
            True,
            True,
            (Sign.NoSign, Sign.Negation, Sign.DoubleNegation),
        )
        body_false = [str(e) for e in full_body if str(e) not in body]
        self.assertListEqual(sorted(body_false), sorted(str(s) for s in res_false))

    def test_get_positive_body(self):
        """
        Test for get_body.
        """
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), not d(X), not not e(X,Y).", ["b(X)", "c(Y)"]
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }.",
            ["b(X)", "c(Y)", "Z = #sum { X: d(X) }"],
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), &sum { X: d(X) } = Z.",
            ["b(X)", "c(Y)", "&sum { X: d(X) } = Z"],
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = { d(X) }.", ["b(X)", "c(Y)", "Z = { d(X) }"]
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), d(Z): e(X,Z).", ["b(X)", "c(Y)", "d(Z): e(X,Z)"]
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), not d(X), not not e(X,Y).",
            ["b(X)", "c(Y)"],
            theory_atom_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }.",
            ["b(X)", "c(Y)", "Z = #sum { X: d(X) }"],
            theory_atom_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), &sum { X: d(X) } = Z.",
            ["b(X)", "c(Y)"],
            theory_atom_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = { d(X) }.",
            ["b(X)", "c(Y)", "Z = { d(X) }"],
            theory_atom_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), d(Z): e(X,Z).",
            ["b(X)", "c(Y)", "d(Z): e(X,Z)"],
            theory_atom_predicate=False,
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), not d(X), not not e(X,Y).",
            ["b(X)", "c(Y)"],
            aggregate_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }.",
            ["b(X)", "c(Y)"],
            aggregate_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), &sum { X: d(X) } = Z.",
            ["b(X)", "c(Y)", "&sum { X: d(X) } = Z"],
            aggregate_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = { d(X) }.",
            ["b(X)", "c(Y)"],
            aggregate_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), d(Z): e(X,Z).",
            ["b(X)", "c(Y)", "d(Z): e(X,Z)"],
            aggregate_predicate=False,
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), not d(X), not not e(X,Y).",
            ["b(X)", "c(Y)"],
            theory_atom_predicate=False,
            aggregate_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }.",
            ["b(X)", "c(Y)"],
            theory_atom_predicate=False,
            aggregate_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), &sum { X: d(X) } = Z.",
            ["b(X)", "c(Y)"],
            theory_atom_predicate=False,
            aggregate_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = { d(X) }.",
            ["b(X)", "c(Y)"],
            theory_atom_predicate=False,
            aggregate_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), d(Z): e(X,Z).",
            ["b(X)", "c(Y)", "d(Z): e(X,Z)"],
            theory_atom_predicate=False,
            aggregate_predicate=False,
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), not d(X), not not e(X,Y).",
            ["b(X)", "c(Y)", "not d(X)"],
            signs=(Sign.NoSign, Sign.Negation),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }.",
            ["b(X)", "c(Y)", "Z = #sum { X: d(X) }"],
            signs=(Sign.NoSign, Sign.Negation),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), &sum { X: d(X) } = Z.",
            ["b(X)", "c(Y)", "&sum { X: d(X) } = Z"],
            signs=(Sign.NoSign, Sign.Negation),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = { d(X) }.",
            ["b(X)", "c(Y)", "Z = { d(X) }"],
            signs=(Sign.NoSign, Sign.Negation),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), d(Z): e(X,Z).",
            ["b(X)", "c(Y)", "d(Z): e(X,Z)"],
            signs=(Sign.NoSign, Sign.Negation),
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), not d(X), not not e(X,Y).",
            ["b(X)", "c(Y)", "not not e(X,Y)"],
            signs=(Sign.NoSign, Sign.DoubleNegation),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }.",
            ["b(X)", "c(Y)", "Z = #sum { X: d(X) }"],
            signs=(Sign.NoSign, Sign.DoubleNegation),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), &sum { X: d(X) } = Z.",
            ["b(X)", "c(Y)", "&sum { X: d(X) } = Z"],
            signs=(Sign.NoSign, Sign.DoubleNegation),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = { d(X) }.",
            ["b(X)", "c(Y)", "Z = { d(X) }"],
            signs=(Sign.NoSign, Sign.DoubleNegation),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), d(Z): e(X,Z).",
            ["b(X)", "c(Y)", "d(Z): e(X,Z)"],
            signs=(Sign.NoSign, Sign.DoubleNegation),
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), not d(X), not not e(X,Y).",
            ["not d(X)", "not not e(X,Y)"],
            signs=(
                Sign.Negation,
                Sign.DoubleNegation,
            ),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }.",
            [],
            signs=(
                Sign.Negation,
                Sign.DoubleNegation,
            ),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), &sum { X: d(X) } = Z.",
            [],
            signs=(
                Sign.Negation,
                Sign.DoubleNegation,
            ),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = { d(X) }.",
            [],
            signs=(
                Sign.Negation,
                Sign.DoubleNegation,
            ),
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), d(Z): e(X,Z).",
            [],
            signs=(
                Sign.Negation,
                Sign.DoubleNegation,
            ),
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), not d(X), not not e(X,Y).",
            ["b(X)", "c(Y)"],
            conditional_literal_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }.",
            ["b(X)", "c(Y)", "Z = #sum { X: d(X) }"],
            conditional_literal_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), &sum { X: d(X) } = Z.",
            ["b(X)", "c(Y)", "&sum { X: d(X) } = Z"],
            conditional_literal_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = { d(X) }.",
            ["b(X)", "c(Y)", "Z = { d(X) }"],
            conditional_literal_predicate=False,
        )
        self.helper_body_elements(
            "a(X) :- b(X), c(Y), d(Z): e(X,Z).",
            ["b(X)", "c(Y)"],
            conditional_literal_predicate=False,
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }, Z = #count { X: d(X) }.",
            ["Z = #count { X: d(X) }", "Z = #sum { X: d(X) }"],
            symbolic_atom_predicate=False,
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }, Z = #count { X: d(X) }.",
            ["Z = #count { X: d(X) }", "Z = #sum { X: d(X) }", "b(X)"],
            symbolic_atom_predicate=lambda x: x.ast_type == ASTType.Function
            and x.name == "b",
        )

        self.helper_body_elements(
            "a(X) :- b(X), c(Y), Z = #sum { X: d(X) }, Z = #count { X: d(X) }.",
            ["Z = #count { X: d(X) }", "b(X)", "c(Y)"],
            aggregate_predicate=lambda x: x.ast_type == ASTType.BodyAggregate
            and x.function == AggregateFunction.Count,
        )

        self.helper_body_elements(
            "a(X) :- &k{ b(X) }, &k{ not c(X)}.",
            ["&k { b(X) }"],
            theory_atom_predicate=lambda x: not (
                x.elements
                and x.elements[0].terms
                and x.elements[0].terms[0].ast_type == ASTType.TheoryUnparsedTerm
                and x.elements[0].terms[0].elements
                and x.elements[0].terms[0].elements[0].ast_type
                == ASTType.TheoryUnparsedTermElement
                and x.elements[0].terms[0].elements[0].operators
                and x.elements[0].terms[0].elements[0].operators[0] == "not"
            ),
        )
        self.helper_body_elements(
            "a(X) :- b(X), not c(Y), d(Z): e(X,Z); not d(Z): e(X,Z).",
            ["b(X)", "d(Z): e(X,Z)"],
            signs=(Sign.NoSign,),
        )
        self.helper_body_elements(
            "a(X) :- b(X), not c(Y), d(Z): e(X,Z); not d(Z): e(X,Z).",
            ["b(X)", "d(Z): e(X,Z)"],
            signs=(Sign.NoSign,),
            conditional_literal_predicate=lambda x: x.literal.sign != Sign.Negation,
        )
        stm = last_stm("#show a.")
        self.assertListEqual(list(filter_body_literals([stm])), [stm])

    def _aux_theory_term_to_term(self, s: str) -> None:
        """
        Parse the given theory term using a simple parse table for testing.
        """
        parsed = parse_theory_term(s)
        unparsed = theory_atom(f"&p {{{s}}}").elements[0].terms[0]
        term = parse_clingo_term(s)

        self.assertEqual(theory_term_to_term(parsed, False), term, "without parsing")
        self.assertEqual(theory_term_to_term(unparsed, True), term, "with parsing")

    def test_theory_term_to_term(self):
        """
        Tests for converting theory terms into terms.
        """
        self._aux_theory_term_to_term("(1,-1,~1)")
        self._aux_theory_term_to_term("(1+X,1-X,1*X,1/X,1\\X,1**X,1&X,1?X,1^X)")
        self._aux_theory_term_to_term("1..X")
        self._aux_theory_term_to_term("f(X)")
        self._aux_theory_term_to_term("-1+ ~2-3*4/5\\6**7&8?9^10..11")

        with self.assertRaisesRegex(RuntimeError, "invalid term"):
            theory_term_to_term(parse_theory_term("[3*4]"))
        with self.assertRaisesRegex(RuntimeError, "invalid term"):
            theory_term_to_term(parse_theory_term("{3*4}"))

    def _aux_theory_term_to_literal(self, s: str, s_expected: Optional[str] = None):
        """
        Test parsing the given string representing a theory literal.
        """
        parsed = parse_theory_term_as_literal(s)
        unparsed = theory_atom(f"&p {{{s}}}").elements[0].terms[0]
        expected = parse_clingo_literal(s if s_expected is None else s_expected)

        self.assertEqual(
            theory_term_to_literal(parsed, False), expected, "without parsing"
        )
        self.assertEqual(
            theory_term_to_literal(unparsed, True), expected, "with parsing"
        )

    def test_theory_term_to_literal(self):
        """
        Tests for converting theory terms into terms.
        """
        self._aux_theory_term_to_literal("p")
        self._aux_theory_term_to_literal("p(1)")
        self._aux_theory_term_to_literal("not p(1+X,1-X,1*X,1/X,1\\X,1**X,1&X,1?X,1^X)")
        self._aux_theory_term_to_literal("not not p(1..X)")
        self._aux_theory_term_to_literal("-p(f(X))")
        self._aux_theory_term_to_literal("not -p(1)")
        self._aux_theory_term_to_literal("not not -p(1)")

        self._aux_theory_term_to_literal("not not not p(1)", "not p(1)")
        self._aux_theory_term_to_literal("not not not not p(1)", "not not p(1)")
        self._aux_theory_term_to_literal("- -p(1)", "p(1)")
        self._aux_theory_term_to_literal("- - -p(1)", "-p(1)")
        self._aux_theory_term_to_literal("- not p(1)", "not not p(1)")
        self._aux_theory_term_to_literal("- - not p(1)", "not p(1)")
        self._aux_theory_term_to_literal("- not not p(1)", "not p(1)")
        self._aux_theory_term_to_literal("- not not - not p(1)", "not p(1)")
        self._aux_theory_term_to_literal("- -not not p(1)", "not not p(1)")

        with self.assertRaisesRegex(RuntimeError, "cannot parse operator"):
            theory_term_to_literal(parse_theory_term_as_literal("p(not 1)"))
        with self.assertRaisesRegex(RuntimeError, "invalid literal"):
            theory_term_to_literal(parse_theory_term_as_literal("(a,b)"))
        with self.assertRaisesRegex(RuntimeError, "invalid literal"):
            theory_term_to_literal(parse_theory_term_as_literal("not 3*4"))
