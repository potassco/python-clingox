import os

import nox

nox.options.sessions = "lint_pylint", "typecheck", "test"

PYTHON_VERSIONS = None
if "GITHUB_ACTIONS" in os.environ:
    PYTHON_VERSIONS = ["3.7", "3.11"]


@nox.session
def format(session):
    session.install("black", "isort", "autoflake")
    check = "check" in session.posargs

    autoflake_args = [
        "--in-place",
        "--imports=clingo,clingox",
        "--ignore-init-module-imports",
        "--remove-unused-variables",
        "-r",
        "clingox",
    ]
    if check:
        autoflake_args.remove("--in-place")
    session.run("autoflake", *autoflake_args)

    isort_args = ["--profile", "black", "clingox"]
    if check:
        isort_args.insert(0, "--check")
        isort_args.insert(1, "--diff")
    session.run("isort", *isort_args)

    black_args = ["clingox"]
    if check:
        black_args.insert(0, "--check")
        black_args.insert(1, "--diff")
    session.run("black", *black_args)


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
