#!/usr/bin/python

import sys
import clingo
import theory

class Application:
    def __init__(self, name):
        self.program_name = name
        self.version = "1.0"
        self.__theory = theory.Theory("clingodl", "clingo-dl")

    def __on_model(self, model):
        self.__theory.on_model(model)

    def register_options(self, options):
        self.__theory.register_options(options)

    def validate_options(self):
        self.__theory.validate_options()
        return True

    def __on_statistics(self, step, accu):
        self.__theory.on_statistics(step, accu)
        pass

    def main(self, prg, files):
        self.__theory.register(prg)
        if not files:
            files.append("-")
        for f in files:
            prg.load(f)

        prg.ground([("base", [])])
        self.__theory.prepare(prg)

        with prg.solve(on_model=self.__on_model, on_statistics=self.__on_statistics, yield_=True) as handle:
            for model in handle:
                sys.stdout.write("assignment:")
                for name, value in self.__theory.assignment(model.thread_id):
                    sys.stdout.write(" {}={}".format(name, value))
                sys.stdout.write("\n")

sys.exit(int(clingo.clingo_main(Application("test"), sys.argv[1:])))
