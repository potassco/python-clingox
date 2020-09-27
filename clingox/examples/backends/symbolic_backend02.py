
import clingo
from clingox.backends import SymbolicBackend
ctl = clingo.Control()
a = clingo.Function("a")
b = clingo.Function("b")
c = clingo.Function("c")
with ctl.backend() as backend:
    symbolic_backend = SymbolicBackend(backend)
    symbolic_backend.add_rule([a], [b], [c])
    atom_b = backend.add_atom(b)
    backend.add_rule([atom_b])
ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
