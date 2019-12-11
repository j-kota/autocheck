import numpy as np
import math
import collections
import checkers

# Exploration constant c is defined as C_input
C_input = 1

class ParentRootNode(object):
    def __init__(self):
        self.parent = None
        self.child_number_visits = collections.defaultdict(float)
        self.child_simulation_reward = collections.defaultdict(float)


class Node(object):
    def __init__(self, board, moves, player, parent=None):
        self.board = board
        self.player = player
        self.is_expanded = False
        self.moves = moves
        self.parent = parent
        self.child_prior_probability = np.zeros([moves.size()], dtype=np.float32)
        self.child_number_visits = np.zeros([moves.size()], dtype=np.float32)
        self.child_simulation_reward = np.zeros([moves.size()], dtype=np.float32)
        self.children = {}

    def N(self):
        return self.parent.child_number_vists[self.move]

    def N(self, value):
        self.parent.child_number_vists[self.move] = value

    def R(self):
        return self.parent.child_simulation_reward[self.move]

    def R(self, value):
        self.parent.child_simulation_reward[self.move] = value

    def Q(self):
        return self.R() / (1 + self.N())

    def child_Q(self):
        return self.child_simulation_reward/(1 + self.child_number_visits)

    def child_U(self):
        return C_input * math.sqrt(self.N())*self.child_prior_probability/(1 + self.child_number_visits)

    def child_score(self):
        return self.child_Q() + self.child_U()

    def best_child(self):
        return np.argmax(self.child_score())

    def select_leaf(self):
        current = self
        while current.is_expanded:
            current = current.maybe_add_child(current.best_child())
        return current

    def maybe_add_child(self, move):
        if move not in self.children:
            new_state = checkers.apply_move(self.board, self.moves[move][0], self.moves[move][1], self.player)
            self.children[move] = Node()


    def backpropagate(self):
















