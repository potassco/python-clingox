import os

import nox

nox.options.sessions = "lint_flake8", "lint_pylint", "typecheck", "test"

PYTHON_VERSIONS = ["3.6", "3.9"] if "GITHUB_ACTIONS" in os.environ else None


@nox.session
def format(session):
    session.install("black", "isort", "autoflake")
    args = session.posargs if session.posargs else ["clingox"]
    session.run(
        "autoflake",
        "--in-place",
        "--imports=clingo,clingox",
        "--ignore-init-module-imports",
        "--remove-unused-variables",
        "-r",
        "clingox",
    )
    session.run("isort", "--profile", "black", "clingox")
    session.run("black", *args)


@nox.session
def lint_flake8(session):
    session.install("flake8", "flake8-black", "flake8-isort")
    session.run("flake8", "clingox")


@nox.session
@nox.parametrize("stable", [True, False])
def lint_pylint(session, stable):
    session.install(
        "-r", f".github/requirements{'-stable' if stable else ''}.txt", "pylint"
    )
    session.run("pylint", "clingox")


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session):
    session.install("-r", ".github/requirements.txt", "mypy")
    session.run("mypy", "-p", "clingox")


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("stable", [True, False])
def test(session, stable):
    if stable:
        session.install("-r", f".github/requirements{'-stable' if stable else ''}.txt")
        session.run("python", "-m", "unittest", "discover", "-v")
    else:
        session.install(
            "-r", f".github/requirements{'-stable' if stable else ''}.txt", "coverage"
        )
        session.run("coverage", "run", "-m", "unittest", "discover", "-v")
        session.run("coverage", "report", "-m", "--fail-under=100")
