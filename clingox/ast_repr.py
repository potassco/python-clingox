from clingo.ast import AST, Sign

class ASTInspector(object):
    def __init__(self):
        self.level = 0
        self.string = ""

    def _sign_repr(self, sign):
        if sign == Sign.NoSign:
            return "Sign.NoSign"
        elif sign == Sign.Negation:
            return "Sign.Negation"
        else: # sign == Sign.DoubleNegation:
            return "Sign.DoubleNegation"

    def visit_children(self, x, *args, **kwargs):
        """
        Visit all child nodes of the current node.
        """
        ident = 2*self.level*" "
        for key in x.keys():
            self.string += ident + key + " = "
            if key == "sign":
                self.string += self._sign_repr(getattr(x, key))
            if key in x.child_keys:
                self.visit(getattr(x, key), *args, **kwargs)
            else:
                self.string += repr(getattr(x, key))
            self.string += "\n"

    def visit_list(self, x, *args, **kwargs):
        """
        Visit a list of AST nodes.
        """
        ident = 2*self.level*" "
        if len(x) == 0:
            self.string += "[]"
        else:
            self.string += "[\n"
            for y in x:
                self.visit(y, *args, **kwargs)
            self.string += ident + "]\n"

    def visit_tuple(self, x, *args, **kwargs):
        """
        Visit a list of AST nodes.
        """
        for y in x:
            self.visit(y, *args, **kwargs)

    def visit_none(self, *args, **kwargs):
        """
        Visit none.

        This, is to handle optional arguments that do not have a visit method.
        """

    def visit(self, x, *args, **kwargs):
        """
        Default visit method to dispatch calls to child nodes.
        """
        if isinstance(x, AST):
            self.string += str(x.type) + "(    # " + str(x) + "\n"
            self.level += 1
            self.visit_children(x, *args, **kwargs)
            self.level -= 1
            ident = 2*self.level*" "
            self.string += ident + ")"
            return
        if isinstance(x, list):
            return self.visit_list(x, *args, **kwargs)
        if isinstance(x, tuple):
            return self.visit_tuple(x, *args, **kwargs)
        if x is None:
            return self.visit_none(x, *args, **kwargs)
        self.string += str(x)
        return

def ast_repr(stm):
    t = ASTInspector()
    t.visit(stm)
    return t.string
