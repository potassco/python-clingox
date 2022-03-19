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
        ctl.add('base', [], '{a}. b :- a.')
        x = []
        reifier = Reifier(x.append)
        ctl.register_observer(reifier)
        ctl.ground([('base', [])])
        ctl.solve()
        print(', '.join(str(y) for y in x))
