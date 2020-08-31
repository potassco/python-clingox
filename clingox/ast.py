'''
This module provides highlevel functions to work with clingo's AST.

TODO:
- theory parsing as in telingo
- unpooling as in clingcon
'''

from typing import Any, List, Sequence, Mapping, Optional, cast
from copy import copy

from clingo.ast import AST


class Visitor:
    '''
    A visitor for clingo's abstart syntaxt tree.

    This class should be derived from. Implementing functions with name
    `visit_<type>` can be used to visit nodes of the given type.
    '''
    def visit_children(self, x: AST, *args: Sequence[Any], **kwargs: Mapping[str, Any]):
        '''
        Visit the children of an AST node.
        '''
        for key in x.child_keys:
            self.visit(getattr(x, key), *args, **kwargs)

    def visit_list(self, x: List[AST], *args: Sequence[Any], **kwargs):
        '''
        Visit a list of AST nodes.
        '''
        for y in x:
            self.visit(y, *args, **kwargs)

    def visit_tuple(self, x: Sequence[AST], *args: Sequence[Any], **kwargs):
        '''
        Visit a tuple of AST nodes.
        '''
        for y in x:
            self.visit(y, *args, **kwargs)

    def visit_none(self, x: None, *args: Sequence[Any], **kwargs):
        '''
        Called when an optional child in an AST is absent.
        '''

    def visit(self, x: Optional[AST], *args: Sequence[Any], **kwargs):
        '''
        Generic visit method dispatching to specific member functions to visit
        child nodes.
        '''
        if isinstance(x, AST):
            attr = "visit_" + str(x.type)
            if hasattr(self, attr):
                getattr(self, attr)(x, *args, **kwargs)
            else:
                self.visit_children(x, *args, **kwargs)
        elif isinstance(x, list):
            self.visit_list(x, *args, **kwargs)
        elif isinstance(x, tuple):
            self.visit_tuple(x, *args, **kwargs)
        elif x is None:
            self.visit_none(x, *args, **kwargs)
        else:
            raise TypeError("unexpected type: {}".format(x))

    def __call__(self, x: Optional[AST], *args: Sequence[Any], **kwargs):
        '''
        Alternative to call visit.
        '''
        return self.visit(x, *args, **kwargs)


class Transformer:
    '''
    This class is similar to the `Visitor` but allows for mutating the AST by
    returning modified AST nodes from the visit methods.
    '''
    def visit_children(self, x: AST, *args: Sequence[Any], **kwargs: Mapping[str, Any]):
        '''
        Visit the children of an AST node.
        '''
        copied = False
        for key in x.child_keys:
            y = getattr(x, key)
            z = self.visit(y, *args, **kwargs)
            if y is not z:
                if not copied:
                    copied = True
                    x = copy(x)
                setattr(x, key, z)
        return x

    def _seq(self, i: int, z: AST, x: Sequence[AST], args: Sequence[Any], kwargs: Mapping[str, Any]):
        for y in x[:i]:
            yield y
        yield z
        for y in x[i+1:]:
            yield self.visit(y, *args, **kwargs)

    def visit_list(self, x: List[AST], *args: Sequence[Any], **kwargs: Mapping[str, Any]):
        '''
        Visit a list of AST nodes.
        '''
        for i, y in enumerate(x):
            z = self.visit(y, *args, **kwargs)
            if y is not z:
                return list(self._seq(i, cast(AST, z), x, args, kwargs))
        return x

    def visit_tuple(self, x: Sequence[AST], *args: Sequence[Any], **kwargs: Mapping[str, Any]):
        '''
        Visit a tuple of AST nodes.
        '''
        for i, y in enumerate(x):
            z = self.visit(y, *args, **kwargs)
            if y is not z:
                return tuple(self._seq(i, cast(AST, z), x, args, kwargs))
        return x

    def visit_none(self, x: None, *args: Sequence[Any], **kwargs: Mapping[str, Any]):
        '''
        Called when an optional child in an AST is absent.
        '''
        # pylint: disable=no-self-use,unused-argument
        return x

    def visit(self, x: Optional[AST], *args: Sequence[Any], **kwargs) -> Optional[AST]:
        '''
        Generic visit method dispatching to specific member functions to visit
        child nodes.
        '''
        if isinstance(x, AST):
            attr = "visit_" + str(x.type)
            if hasattr(self, attr):
                return getattr(self, attr)(x, *args, **kwargs)
            return self.visit_children(x, *args, **kwargs)
        if isinstance(x, list):
            return self.visit_list(x, *args, **kwargs)
        if isinstance(x, tuple):
            return self.visit_tuple(x, *args, **kwargs)
        if x is None:
            return self.visit_none(x, *args, **kwargs)
        raise TypeError("unexpected type: {}".format(x))
