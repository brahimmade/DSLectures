from enum import Enum
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

class State:
    def __init__(self, dealer_card, player_card, is_terminal=False):
        """
        :type self.is_terminal: bool
        :type self.dealer: int
        :type self.player: int
        """
        self.dealer = dealer_card.value
        self.player = player_card.value
        self.terminal = is_terminal
        self.r = 0

    def dealer_idx(self):
        return self.dealer - 1

    def player_idx(self):
        return self.player - 1


class DealCard:
    def __init__(self, force_black=False):
        """
        :type self.value: int
        """

        self.value = random.randint(1, 10)

        if force_black or random.randint(1, 3) != 1:
            None
        else:
            self.value = 0 - self.value


class Actions(Enum):

    # Possible actions
    hit = 0
    stick = 1

    def to_action(n):
        return Actions.hit if n == 0 else Actions.stick

    def as_int(a):
        return 0 if a == Actions.hit else 1


class Environment:
    def __init__(self):
        self.player_values_count = 21
        self.dealer_values_count = 10
        self.actions_count = 2  # number of possible actions

    def get_start_state(self):
        s = State(DealCard(force_black = True), DealCard(force_black = True))
        return s

    def step(self, s, a):
        # type: (object, object) -> object
        """
        :type s: State
        """
        next_state = copy.copy(s)
        r = 0
        if a == Actions.hit:
            next_state.player = next_state.player + DealCard().value
            if next_state.player < 1 or next_state.player > 21:
                next_state.terminal = True
                r = -1
        else:
            while not next_state.terminal:
                next_state.dealer = next_state.dealer + DealCard().value
                if next_state.dealer < 1 or next_state.dealer > 21:
                    next_state.terminal = True
                    r = 1
                elif next_state.dealer >= 17:
                    next_state.terminal = True
                    if next_state.dealer > next_state.player:
                        r = -1
                    elif next_state.dealer < next_state.player:
                        r = 1

        next_state.r = r
        return next_state, r

