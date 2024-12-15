# -*- coding: utf-8 -*-
"""
A pure MCTS (Monte Carlo Tree Search) implementation without any policy-value network.
This serves as a baseline or weaker AI opponent for comparison.

Classes:
- MCTS: a simple MCTS implementation with rollout for evaluation.
- MCTSPlayer: a player controlled by the pure MCTS algorithm.

Functions:
- rollout_policy_fn(board): used during rollouts to select moves randomly.
- policy_value_fn(board): returns uniform probabilities for all available moves and zero value.
"""

import numpy as np
import copy
from operator import itemgetter

def rollout_policy_fn(board):
    """
    A fast, coarse policy function used during the rollout phase.
    Here we simply choose a move randomly.
    """
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)

def policy_value_fn(board):
    """
    A policy-value function for the pure MCTS:
    - Returns uniform probabilities for all available moves.
    - Returns a zero value (no evaluation by a neural network).
    """
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0.0

class TreeNode(object):
    """
    A node in the MCTS tree.

    Attributes:
        _parent (TreeNode): the parent node
        _children (dict): map from action to TreeNode
        _n_visits (int): visit count
        _Q (float): average value of this node
        _u (float): exploration bonus
        _P (float): prior probability of selecting this action
    """
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0.0
        self._u = 0.0
        self._P = prior_p

    def expand(self, action_priors):
        """
        Expand the node by creating new children.
        action_priors: list of (action, probability) from policy_value_fn.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        Select the child that maximizes Q+U.
        Returns:
            (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        Update this node from a leaf evaluation.
        leaf_value: a scalar in [-1,1] from the current player's perspective.
        """
        self._n_visits += 1
        # Q_new = Q_old + (leaf_value - Q_old) / N
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        Recursively update all ancestors.
        The parent's perspective is opposite the child's.
        """
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        Calculate node's value for MCTS selection:
        Q + U = Q + c_puct * P * sqrt(parent_N)/(1+N)
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return len(self._children) == 0

    def is_root(self):
        return self._parent is None

class MCTS(object):
    """
    A pure MCTS implementation without neural network guidance.
    Uses random rollouts to estimate the value of leaf states.
    """
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        Args:
            policy_value_fn: function(board) -> (action_probs, value)
            c_puct (float): exploration parameter
            n_playout (int): number of simulations per move
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        Execute one simulation (playout) from root to a leaf, then use rollout
        to evaluate and update the nodes.
        """
        node = self._root
        # 1. Selection
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 2. Expansion
        action_probs, _ = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)

        # 3. Evaluation by random rollout
        leaf_value = self._evaluate_rollout(state)

        # 4. Backpropagation
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """
        Perform a rollout until the game ends or we reach the move limit.
        Returns:
            +1 if current player eventually wins,
            -1 if opponent wins,
            0 if tie.
        """
        player = state.get_current_player()
        for _ in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # If we didn't break, reached limit without ending
            print("WARNING: Rollout reached move limit")

        # Determine the result
        if winner == -1:  # tie
            return 0.0
        return 1.0 if winner == player else -1.0

    def get_move(self, state):
        """
        Perform all playouts and then return the most visited action.
        """
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        # select move with highest visit count
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """
        Step the tree forward: if last_move is a child of the root, use it;
        otherwise create a new root.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    """
    A player controlled by the pure MCTS algorithm.
    Used as a weaker baseline opponent or for evaluation.
    """
    def __init__(self, c_puct=5, n_playout=2000):
        """
        Args:
            c_puct (float): exploration parameter
            n_playout (int): number of playouts
        """
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.player = None

    def set_player_ind(self, p):
        """
        Set the player ID (1 or 2).
        """
        self.player = p

    def reset_player(self):
        """
        Reset the MCTS tree.
        """
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        """
        Get the next move from MCTS.
        """
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            # After choosing a move, reset MCTS tree for next turn
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: The board is full. No moves left.")
            return -1

    def __str__(self):
        return f"MCTS Player {self.player}"
