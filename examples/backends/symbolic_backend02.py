'''
Simple example showing how to attach the symbolic backend to an existing
backend.
'''

from clingo import Control, Function
from clingox.backends import SymbolicBackend


a, b, c = Function("a"), Function("b"), Function("c")

ctl = Control()

with ctl.backend() as backend:
    atom_b = backend.add_atom(b)
    backend.add_rule([atom_b])

    symbolic_backend = SymbolicBackend(backend)
    symbolic_backend.add_rule([a], [b], [c])

ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
