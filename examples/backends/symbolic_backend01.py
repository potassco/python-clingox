'''
Simple example showing how to attach the symbolic backend to an existing
backend.
'''
from clingo import Control, Function
from clingox.backends import SymbolicBackend


a, b, c = Function("a"), Function("b"), Function("c")

ctl = Control()

with SymbolicBackend(ctl.backend()) as symbolic_backend:
    symbolic_backend.add_rule([b])
    symbolic_backend.add_rule([a], [b], [c])

ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
