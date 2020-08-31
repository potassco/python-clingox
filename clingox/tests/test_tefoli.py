"""
Tests tefoli by loading the clingo-dl and clingcon libraries.

This module requires that the OS can load the required libaries. This can be
achieved by either installing clingo-dl and clingcon in a default location or
setting environment variables.
"""

import os
import sys
from tempfile import NamedTemporaryFile
from multiprocessing import Process, Queue
from unittest import TestCase
from typing import Callable, Sequence, List, Tuple

from clingo import (clingo_main, parse_files,
                    Application, ApplicationOptions, Control, Model, StatisticsMap, SymbolType)
from ..tefoli import Theory


class ClingoDL(Application):
    '''
    Example application providing clingo-dl functionality.
    '''
    program_name: str
    version: str
    _theory: Theory

    def __init__(self):
        self.program_name = "clingo-dl"
        self.version = "1.0"
        self._theory = Theory("clingodl", "clingo-dl")

    def register_options(self, options: ApplicationOptions):
        """
        Register options with theory.
        """
        self._theory.register_options(options)

    def validate_options(self):
        """
        Check options in theory.
        """
        self._theory.validate_options()
        return True

    def _hidden(self, symbol):
        return symbol.type == SymbolType.Function and symbol.name.startswith("__")

    def print_model(self, model: Model, printer: Callable[[], None]):
        """
        Print the assignment in nice text format along with the model.
        """
        # pylint: disable=unused-argument

        # print model
        symbols = model.symbols(shown=True)
        sys.stdout.write(" ".join(str(symbol) for symbol in sorted(symbols) if not self._hidden(symbol)))
        sys.stdout.write('\n')

        # print assignment
        sys.stdout.write("assignment:")
        for name, value in self._theory.assignment(model.thread_id):
            sys.stdout.write(" {}={}".format(name, value))
        sys.stdout.write("\n")

        sys.stdout.flush()

    def main(self, ctl: Control, files):
        """
        Main function repsonsible for registering the theory, loading and
        parsing files, grounding, and solving.
        """
        self._theory.register(ctl)

        with ctl.builder() as bld:
            parse_files(files, lambda stm: self._theory.rewrite_statement(stm, bld.add))

        ctl.ground([("base", [])])
        self._theory.prepare(ctl)

        ctl.solve(on_model=self._on_model, on_statistics=self._on_statistics)

    def _on_model(self, model: Model):
        """
        Callback to report models to theory and add additional output.
        """
        self._theory.on_model(model)

    def _on_statistics(self, step: StatisticsMap, accu: StatisticsMap):
        """
        Callback to gather statistics of theory.
        """
        self._theory.on_statistics(step, accu)


class ClingconApp(Application):
    '''
    Example application providing clingcon functionality.
    '''

    def __init__(self):
        self.program_name = "clingcon"
        self.version = "1.0"
        self._theory = Theory("clingcon", "clingcon")

    def register_options(self, options):
        """
        Register options with theory.
        """
        self._theory.register_options(options)

    def validate_options(self):
        """
        Check options in theory.
        """
        self._theory.validate_options()
        return True

    def _hidden(self, symbol):
        return symbol.type == SymbolType.Function and symbol.name.startswith("__")

    def print_model(self, model, default_printer):
        """
        Print the assignment in nice text format along with the model.
        """
        # pylint: disable=unused-argument

        # print model
        symbols = model.symbols(shown=True)
        sys.stdout.write(" ".join(str(symbol) for symbol in sorted(symbols) if not self._hidden(symbol)))
        sys.stdout.write('\n')

        # print assignment
        sys.stdout.write('Assignment:\n')
        symbols = model.symbols(theory=True)
        assignment = []
        cost = None
        for symbol in sorted(symbols):
            if symbol.match("__csp", 2):
                assignment.append("{}={}".format(*symbol.arguments))
            if symbol.match("__csp_cost", 1):
                cost = symbol.arguments[0].string()
        sys.stdout.write(" ".join(assignment))
        sys.stdout.write('\n')

        # print cost
        if cost is not None:
            sys.stdout.write("Cost: {}\n".format(cost))

        sys.stdout.flush()


    def main(self, ctl, files):
        """
        Main function repsonsible for registering the theory, loading and
        parsing files, grounding, and solving.
        """
        self._theory.register(ctl)

        with ctl.builder() as bld:
            parse_files(files, lambda stm: self._theory.rewrite_statement(stm, bld.add))

        ctl.ground([("base", [])])
        self._theory.prepare(ctl)

        ctl.solve(on_model=self._on_model, on_statistics=self._on_statistics)

    def _on_model(self, model):
        """
        Callback to report models to theory and add additional output.
        """
        self._theory.on_model(model)

    def _on_statistics(self, step, accu):
        self._theory.on_statistics(step, accu)


class TestClingoDL(ClingoDL):
    """
    This class extends the ClingoDL class to trace models and results.
    """
    __queue: Queue

    def __init__(self, queue: Queue):
        ClingoDL.__init__(self)
        self.__queue = queue

    def _on_model(self, model: Model):
        ClingoDL._on_model(self, model)

        symbols = [f'{symbol}' for symbol in model.symbols(shown=True)]
        assignments = [f'{name}={value}' for name, value in sorted(self._theory.assignment(model.thread_id))]
        self.__queue.put((symbols, assignments))


class TestClingconApp(ClingconApp):
    """
    This class extends the ClingconApp class to trace models and results.
    """
    __queue: Queue

    def __init__(self, queue: Queue):
        ClingconApp.__init__(self)
        self.__queue = queue

    def _on_model(self, model: Model):
        ClingconApp._on_model(self, model)

        symbols = [f'{symbol}' for symbol in model.symbols(shown=True)]
        assignments = [f'{name}={value}' for name, value in sorted(self._theory.assignment(model.thread_id))]
        self.__queue.put((symbols, assignments))


def _run_process(app: Callable[[Queue], Application], program: str, queue: Queue, args: Sequence[str]):
    '''
    Run clingo application with given program and intercept results.
    '''
    with NamedTemporaryFile(mode='wt', delete=False) as fp:
        name = fp.name
        fp.write(program)
    try:
        # Note: The multiprocess module does not allow for intercepting the
        # output. Thus, the output is simply disabled and we use the Queue
        # class to communicate results.
        ret = clingo_main(app(queue), (name, '--outf=3') + tuple(args))
        queue.put(int(ret))
    finally:
        os.unlink(name)


AppResult = Tuple[int, List[Tuple[List[str], List[str]]]]


def run_app(app: Callable[[Queue], Application], program: str, *args: Sequence[str]) -> AppResult:
    '''
    Run clingo application in subprocess via multiprocessing module.
    '''
    q: Queue
    q = Queue()
    p = Process(target=_run_process, args=(app, program, q, tuple(args)))

    p.start()
    seq: List[Tuple[List[str], List[str]]]
    seq, ret = [], -1
    while True:
        ret = q.get()
        if isinstance(ret, int):
            status = ret
            break
        seq.append(ret)
    p.join()

    seq.sort()
    return status, seq


def clingodl(program: str, *args: Sequence[str]) -> AppResult:
    """
    Run clingo-dl with given program and arguments in subproccess.
    """
    return run_app(TestClingoDL, program, *args)


def clingcon(program: str, *args: Sequence[str]) -> AppResult:
    """
    Run clingocon with given program and arguments in subproccess.
    """
    return run_app(TestClingconApp, program, *args)


class TestTefoli(TestCase):
    '''
    Test cases for running clingo-dl and clingcon via the tefoli module.
    '''

    def test_clingodl(self):
        '''
        Test simple clingodl usage.
        '''
        self.assertEqual(
            clingodl("1 {a; b} 1. &diff { x - 0 } = 1 :- a. &diff { x - 0 } = 2 :- b.", "0"),
            (30, [(['a'], ['x=1']), (['b'], ['x=2'])]))

    def test_clingcon(self):
        '''
        Test simple clingcon usage.
        '''
        self.assertEqual(
            clingcon("1 {a; b} 1. &sum { x - 0 } = 1 :- a. &sum { x - 0 } = 2 :- b.", "0"),
            (30, [(['a'], ['x=1']), (['b'], ['x=2'])]))
