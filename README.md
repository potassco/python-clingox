# clingox: Auxiliary functions for the clingo library

This project collects useful Python functions that can be used along with the
clingo library.

## Documentation

For examples how to use the library, check the unit tests. The API documentation
is available [online][doc].

## Installation

We provide clingox packages for various package managers:

```bash
# pip
pip install clingox
# conda
conda install -c conda-forge python-clingox
# ubuntu
sudo add-apt-repository ppa:potassco/stable
sudo apt install python3-clingox
```

## Development

To improve code quality, we run linters, type checkers, and unit tests. The
tools can be run using [nox]:

```bash
python -m pip install nox
nox
```

Note that `nox -r` can be used to speed up subsequent runs. It avoids
recreating virtual environments.

Furthermore, we auto format code using [black]. We provide a [pre-commit][pre]
config to automate this process. It can be set up using the following commands:

```bash
python -m pip install pre-commit
pre-commit install
```

This blackens the source code whenever `git commit` is used.

There is also a format session for nox. It can be run as follows:

```bash
nox -rs format
nox -rs format -- check
```

The latter command can be used to inspect changes before applying them.

[doc]: https://potassco.org/clingo/python-api/current/
[nox]: https://nox.thea.codes/en/stable/index.html
[pre]: https://pre-commit.com/
[black]: https://black.readthedocs.io/en/stable/
