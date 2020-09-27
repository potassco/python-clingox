
import clingo
from clingox.backends import SymbolicBackend
ctl = clingo.Control()
a = clingo.Function("a")
b = clingo.Function("b")
c = clingo.Function("c")
with SymbolicBackend(ctl.backend()) as symbolic_backend:
    symbolic_backend.add_rule([a], [b], [c])
    symbolic_backend.add_rule([b])
ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))