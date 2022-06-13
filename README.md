This projects collects useful python functions that can be used along with the
clingo libary.

For examples how to use the library check the unit tests. The API documentation
is available [online][doc].

# Development

To improve code quality, we run a linters, type checkers, and unit tests. The
tools can be run using [nox]:

```bash
python -m pip install nox
nox
```

Note that `nox -r` to speed up subsequent runs avoiding the recreation of
virtual environments.

Furthermore, we auto format code using [black]. We provide a [pre-commit][pre]
config to automate this process. It can be set up using the following commands:

```bash
python -m pip install pre-commit
pre-commit install
```

This blackens the source code whenever `git commit` is used.

[doc]: https://potassco.org/clingo/python-api/current/
[nox]: https://nox.thea.codes/en/stable/index.html
[pre]: https://pre-commit.com/
[black]: https://black.readthedocs.io/en/stable/
