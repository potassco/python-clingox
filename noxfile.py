import nox
import os

nox.options.sessions = "lint_flake8", "lint_pylint", "typecheck", "test"

PYTHON_VERSIONS = ["3.6", "3.9"] if 'GITHUB_ACTIONS' in os.environ else None


@nox.session
def format(session):
    session.install("black")
    session.run("black", "clingox")


@nox.session
def lint_flake8(session):
    session.install("flake8", "flake8-black")
    session.run("flake8", "clingox")


@nox.session
@nox.parametrize('stable', [True, False])
def lint_pylint(session, stable):
    session.install("-r", f".github/requirements{'-stable' if stable else ''}.txt", "pylint")
    session.run("pylint", "clingox")


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session):
    session.install("-r", ".github/requirements.txt", "mypy")
    session.run("mypy", "-p", "clingox")


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize('stable', [True, False])
def test(session, stable):
    if stable:
        session.install("-r", f".github/requirements{'-stable' if stable else ''}.txt")
        session.run("python", '-m', 'unittest', 'discover', '-v')
    else:
        session.install("-r", f".github/requirements{'-stable' if stable else ''}.txt", "coverage")
        session.run("coverage", 'run', '-m', 'unittest', 'discover', '-v')
        session.run("coverage", 'report', '--fail-under=100')
