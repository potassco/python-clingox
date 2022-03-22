'''
Test cases for the ground program and observer.
'''
from subprocess import Popen, PIPE
from unittest import TestCase

from typing import List, Optional

from clingo.control import Control
from clingo.symbol import Function, Number, Symbol

from ..reify import Reifier, get_theory_symbols

def _get_command_line_reification(prg):
    command = ["clingo --output=reify "]
    with Popen(command,
                        stdin=PIPE,
                        stdout=PIPE,
                        stderr=PIPE,
                        shell=True) as process:
        stdout = process.communicate(input=prg.encode())[0]
        return [s.strip('.') for s in stdout.decode().split('\n') if s !=""]

def _out(name, args, step: Optional[int]):
    return Function(name, args if step is None else args + [Number(step)])

def _at(idx: int, atoms: List[int], step: Optional[int] = None):
    ret = set()
    ret.add(_out('atom_tuple', [Number(idx)], step))
    for atm in atoms:
        ret.add(_out('atom_tuple', [Number(idx), Number(atm)], step))
    return ret

def _lt(idx: int, lits: List[int], step: Optional[int] = None):
    ret = set()
    ret.add(_out('literal_tuple', [Number(idx)], step))
    for lit in lits:
        ret.add(_out('literal_tuple', [Number(idx), Number(lit)], step))
    return ret

def _tag(inc: bool):
    return {_out('tag', [Function('incremental')], None)} if inc else set()

def _scc(idx: int, scc: List[int], step: Optional[int] = None):
    ret = set()
    for atm in scc:
        ret.add(_out('scc', [Number(idx), Number(atm)], step))
    return ret

def _rule(hd, bd, choice=False, step: Optional[int] = None):
    t = 'choice' if choice else 'disjunction'
    return {_out('rule', [Function(t, [Number(hd)]),
                          Function('normal', [Number(bd)])], step)}

def _output(sym: Symbol, lt: int, step: Optional[int] = None):
    return {_out('output', [sym, Number(lt)], step)}

GRAMMAR = """
#theory theory{
    term { <?  : 4, binary, left;
            >?  : 4, binary, left };
    &tel/0     : term, any;
    &tel2/0 : term, {=}, term, head }.
"""
class TestReifier(TestCase):
    '''
    Tests for the Reifier.

    TODO: Add more tests. To make it a bit easier, tests we can also use a meta
    encoding to ensure that the tested programs are equal.
    '''

    def test_simple_with_step(self):
        '''
        Test `a :- b. b :- a.`.
        '''
        ctl = Control()
        x = set()
        reifier = Reifier(x.add, True)
        ctl.register_observer(reifier)
        with ctl.backend() as bck:
            a = bck.add_atom(Function('a'))
            b = bck.add_atom(Function('b'))
            bck.add_rule([a], [b])
            bck.add_rule([b], [a])
        reifier.calculate_sccs()

        self.assertSetEqual(
            _tag(True) |
            _at(0, [1]) | _at(1, [2]) |
            _lt(0, [2]) | _lt(1, [1]) |
            _rule(0, 0) |
            _rule(1, 1) |
            _output(Function('a'), 1) |
            _output(Function('b'), 0) |
            _scc(0, [1, 2]),
            x)

    def test_incremental(self):
        '''
        Test `#step 0. a :- b. b :- a. #step 1. c :- d. d :- c. `.

        TODO: verify that this is relly what clingo is doing!!!!
        '''
        ctl = Control()
        x = set()
        reifier = Reifier(x.add, True, True)
        ctl.register_observer(reifier)
        with ctl.backend() as bck:
            a = bck.add_atom(Function('a'))
            b = bck.add_atom(Function('b'))
            bck.add_rule([a], [b])
            bck.add_rule([b], [a])
        ctl.solve()
        with ctl.backend() as bck:
            c = bck.add_atom(Function('c'))
            d = bck.add_atom(Function('d'))
            bck.add_rule([c], [d])
            bck.add_rule([d], [c])
        ctl.solve()
        self.assertSetEqual(
            _tag(True) |
            _at(0, [1], 0) | _at(1, [2], 0) |
            _lt(0, [2], 0) | _lt(1, [1], 0) |
            _rule(0, 0, step=0) |
            _rule(1, 1, step=0) |
            _output(Function('a'), 1, 0) |
            _output(Function('b'), 0, 0) |
            _scc(0, [1, 2], 0) |
            _at(0, [3], 1) | _at(1, [4], 1) |
            _lt(0, [4], 1) | _lt(1, [3], 1) |
            _rule(0, 0, step=1) |
            _rule(1, 1, step=1) |
            _output(Function('c'), 1, 1) |
            _output(Function('d'), 0, 1) |
            _scc(0, [3, 4], 1),
            x)

    def test_simple(self):
        '''
        Test simple programs without adding the step
        '''
        prgs = [
            '{a(1);a(2)}2. :-a(1..2).',
            ':- not b. {b}.'
        ]
        for prg in prgs:
            ctl = Control()
            x = []
            reifier = Reifier(x.append, False)
            ctl.register_observer(reifier)
            ctl.add("base",[],prg)
            ctl.ground([('base',[])])
            out_str = [str(s) for s in x if s.name!="tag"]
            r = _get_command_line_reification(prg)
            self.assertListEqual(out_str,r)

    def test_theory(self):
        """
        Test programs using theory
        """
        prgs = [
            '&tel{ a("s") <? b({2,3}) }.',
            '&tel{ a <? b([2,c(1)]) }.',
            '&tel{ a(s) <? b((2,3)) }.',
            '&tel2{ a <? b } = c.',
            '&tel{ a <? b: x}. {x}.'
        ]
        for p in prgs:
            ctl = Control()
            prg = GRAMMAR + p
            x = []
            reifier = Reifier(x.append, False)
            ctl.register_observer(reifier)
            ctl.add("base",[],prg)
            ctl.ground([('base',[])])
            out_str = [str(s) for s in x if s.name!="tag"]
            r = _get_command_line_reification(prg)

            self.assertListEqual(out_str,r)

    def test_theory_symbols(self):
        """
        Test function to get the theory_symbols
        """
        ctl = Control()
        prg = GRAMMAR + '&tel{ a(s) <? b((2,3)) }.'
        x = []
        reifier = Reifier(x.append, False)
        ctl.register_observer(reifier)
        ctl.add("base",[],prg)
        ctl.ground([('base',[])])
        theory_symbols = get_theory_symbols(x)
        expected_new_symbols = [
            "theory_symbol(0,tel)",
            "theory_symbol(3,s)",
            "theory_symbol(2,a)",
            "theory_symbol(4,a(s))",
            "theory_symbol(8,(2,3))",
            "theory_symbol(5,b)",
            "theory_symbol(9,b((2,3)))",
        ]
        out_str = [str(s) for s in theory_symbols]
        self.assertListEqual(out_str,expected_new_symbols)
