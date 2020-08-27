#!/usr/bin/python
"""
This example show cases how to use tefoli to load the clingo-dl libary and use
its functionality.
"""

import sys
from typing import Callable
from clingo import (clingo_main,
                    Application, ApplicationOptions, Control, Model, StatisticsMap)
from clingox.tefoli import Theory, parse_files


class DLApp(Application):
    '''
    Example application providing clingo-dl functionality.
    '''
    program_name: str
    version: str
    __theory: Theory

    def __init__(self, name: str):
        self.program_name = name
        self.version = "1.0"
        self.__theory = Theory("clingodl", "clingo-dl")

    def __on_model(self, model: Model):
        """
        Callback to report models to theory and add additional output.
        """
        self.__theory.on_model(model)

    def print_model(self, model: Model, printer: Callable[[], None]):
        """
        Print the assignment in nice text format along with the model.
        """
        printer()
        sys.stdout.write("assignment:")
        for name, value in self.__theory.assignment(model.thread_id):
            sys.stdout.write(" {}={}".format(name, value))
        sys.stdout.write("\n")
        sys.stdout.flush()

    def register_options(self, options: ApplicationOptions):
        """
        Register options with theory.
        """
        self.__theory.register_options(options)

    def validate_options(self):
        """
        Check options in theory.
        """
        self.__theory.validate_options()
        return True

    def __on_statistics(self, step: StatisticsMap, accu: StatisticsMap):
        """
        Callback to gather statistics of theory.
        """
        self.__theory.on_statistics(step, accu)

    def main(self, ctl: Control, files):
        """
        Main function repsonsible for registering the theory, loading and
        parsing files, grounding, and solving.
        """
        self.__theory.register(ctl)

        with ctl.builder() as bld:
            parse_files(files, self.__theory, bld.add)

        ctl.ground([("base", [])])
        self.__theory.prepare(ctl)

        ctl.solve(on_model=self.__on_model, on_statistics=self.__on_statistics)


sys.exit(int(clingo_main(DLApp("clingo-dl"), sys.argv[1:])))
