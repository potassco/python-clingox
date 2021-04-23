from textwrap import dedent
from setuptools import setup

setup(
    version='1.0.1',
    name='clingox',
    description='Support library for clingo.',
    long_description=dedent('''\
        Preliminary documentation can be found here:

        - https://www.cs.uni-potsdam.de/~kaminski/pyclingo-cffi/
        '''),
    long_description_content_type='text/markdown',
    author='Roland Kaminski',
    author_email='kaminski@cs.uni-potsdam.de',
    license='MIT',
    url='https://github.com/potassco/python-clingox',
    install_requires=['clingo-cffi'],
    packages=['clingox'],
    package_data={'clingox': ['py.typed']},
    zip_safe=False,
    python_requires=">=3.6"
)
