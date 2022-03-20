'''
Test cases for the ground program and observer.
'''
from unittest import TestCase

from typing import cast

from clingo import Control

from ..reify import Reifier


class TestReifier(TestCase):
    '''
    Tests for the Reifier.
    '''

    def test_reify(self):
        ctl = Control()
        ctl.add('base', [], '{a}. {b}. b :- a. a :- b.')
        x = []
        reifier = Reifier(x.append, True)
        ctl.register_observer(reifier)
        ctl.ground([('base', [])])
        ctl.solve()
        reifier.calculate_sccs()
        print('\n'.join(str(y) for y in x))
