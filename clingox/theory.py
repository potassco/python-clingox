'''
This module provides functions to work with clingo's theories.
'''

from clingo import Symbol, Function, Tuple_, Number, SymbolType, TheoryTerm, TheoryTermType


def require_number(x: Symbol) -> int:
    '''
    Requires the argument to be a number returning the given number or throwing
    a type error.
    '''
    if x.type == SymbolType.Number:
        return x.number

    raise TypeError('number exepected')

def invert_symbol(x: Symbol) -> Symbol:
    '''
    Inverts a symbol.
    '''
    if x.type == SymbolType.Number:
        return Number(x.number - 1)

    if x.type == SymbolType.Function and x.name:
        return Function(x.name, x.arguments, not x.positive)

    raise TypeError('cannot invert symbol')

class TermEvaluator:
    '''
    Evaluates the operators in a theory term in the same fashion as clingo
    evaluates its arithmetic functions.

    This class can easily be extended for additional binary and unary
    operators.
    '''

    def evaluate_binary(self, f: str, x: Symbol, y: Symbol) -> Symbol:
        '''
        Evaluate binary terms as clingo would.
        '''
        if f == "+":
            return Number(require_number(x) + require_number(y))
        if f == "-":
            return Number(require_number(x) - require_number(y))
        if f == "*":
            return Number(require_number(x) * require_number(y))
        if f == "**":
            return Number(require_number(x) ** require_number(y))
        if f == "\\":
            if y == Number(0):
                raise RuntimeError("division by zero")
            return Number(require_number(x) % require_number(y))
        if f == "/":
            if y == Number(0):
                raise RuntimeError("division by zero")
            return Number(require_number(x) // require_number(y))

        return Function(f, [x, y])

    def evaluate_unary(self, f: str, x: Symbol):
        '''
        Evaluate unary terms as clingo would.
        '''
        if f == "+":
            return Number(require_number(x))
        if f == "-":
            return invert_symbol(x)

        return Function(f, [x])

    def __call__(self, term: TheoryTerm):
        '''
        Evaluate the given term.
        '''
        # tuples
        if term.type == TheoryTermType.Tuple:
            return Tuple_(self(x) for x in term.arguments)

        # functions and arithmetic operations
        if term.type == TheoryTermType.Function:
            arguments = [self(x) for x in term.arguments]
            # binary operations
            if len(arguments) == 2:
                return self.evaluate_binary(term.name, *arguments)

            # unary operations
            if len(arguments) == 1:
                return self.evaluate_unary(term.name, *arguments)

            # functions
            return Function(term.name, arguments)

        # constants
        if term.type == TheoryTermType.Symbol:
            return Function(term.name)

        # numbers
        if term.type == TheoryTermType.Number:
            return Number(term.number)

        raise RuntimeError("cannot evaluate term")


def evaluate(term: TheoryTerm) -> Symbol:
    '''
    Evaluates the operators in a theory term in the same fashion as clingo
    evaluates its arithmetic functions.
    '''
    return TermEvaluator()(term)
