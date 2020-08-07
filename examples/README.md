This example show how to use clingo-dl's propagator with the [clingo] python module.

The example should run out-of-the-box with clingo-dl and clingcon as provided
by [anaconda]. It emulates clingo-dl and clingcons printing along with some
extra information:

    export PYTHONPATH=.
    python examples/clingcon.py  -c n=132 examples/example.lp -t 2 --stats 2
    python examples/clingo-dl.py -c n=132 examples/example.lp --propagate partial -t 2 --stats 2

Note that the example requires Python 3 to run (the theory module uses type
annotations).

[clingo]: https://potassco.org/clingo/python-api/current/
[anaconda]: https://anaconda.org/potassco/
