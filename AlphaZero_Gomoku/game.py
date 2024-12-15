# -*- coding: utf-8 -*-
"""
A simplified and cleaned-up version of the Gomoku game logic,
compatible with TensorFlow 2.x training pipeline and the human-vs-AI script.

This file defines:
- Board: the board state and operations (initialize, check moves, get current state)
- Game: a wrapper to run the game between two players

The `current_state` method returns a 4-channel representation of the board,
compatible with the input of Policy-Value networks used in AlphaZero-like frameworks.
"""

import numpy as np

class Board(object):
    """
    Board for the Gomoku game.

    Attributes:
        width (int): board width
        height (int): board height
        n_in_row (int): number of pieces in a row needed to win
        states (dict): a dictionary storing {move: player}, move is an int index
        players (list): player identifiers, default as [1, 2]
        current_player (int): the player who is about to move
        availables (list): a list of available moves (int)
        last_move (int): the last move made on the board
    """

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 6))
        self.height = int(kwargs.get('height', 6))
        self.n_in_row = int(kwargs.get('n_in_row', 4))
        self.players = [1, 2]
        self.states = {}
        self.availables = []
        self.current_player = self.players[0]
        self.last_move = -1

    def init_board(self, start_player=0):
        """
        Initialize the board and start the game with the given start player.

        Args:
            start_player (int): 0 or 1, the index in self.players for who plays first.
        """
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise ValueError(f"board width and height must be at least {self.n_in_row}")

        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        Convert a move (int) to a board location [row, col].
        
        Example for a 3x3 board:
          moves:    locations:
           6 7 8     (2,0)(2,1)(2,2)
           3 4 5 ->  (1,0)(1,1)(1,2)
           0 1 2     (0,0)(0,1)(0,2)

        Args:
            move (int): the move index
        
        Returns:
            [h, w] (list): row and column
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """
        Convert a board location [row, col] to the move index.

        Args:
            location (list[int]): [row, col]
        
        Returns:
            move (int): the move index or -1 if invalid
        """
        if len(location) != 2:
            return -1
        h, w = location
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        Return the board state from the perspective of the current player.

        The state shape is [4, width, height]:
        - state[0]: current player's stones
        - state[1]: opponent player's stones
        - state[2]: mark of the last move made (for policy/value net reference)
        - state[3]: a plane indicating whose turn it is (1.0 if current player's turn)

        Note:
            The returned state uses the current player's perspective.
            state is flipped upside down (::-1) as in original code for consistency.

        Returns:
            np.ndarray: A numpy array of shape (4, width, height).
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(self.states.keys())), np.array(list(self.states.values()))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0

            if self.last_move != -1:
                square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0

        # If it's the current player's turn, fill with 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]

    def do_move(self, move):
        """
        Execute a move on the board, and switch current player.

        Args:
            move (int): move index
        """
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move

    def has_a_winner(self):
        """
        Check if there's a winner on the current board.

        Returns:
            (bool, int): (win_flag, winner)
                win_flag: True if there's a winner
                winner: winner's player number if win_flag is True, otherwise -1
        """
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        # No need to check until we have enough moves
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            # Horizontal
            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # Vertical
            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            # Diagonal \
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            # Diagonal /
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """
        Check if the game has ended.

        Returns:
            (bool, int): (end, winner)
                end: True if the game ended (win or tie)
                winner: player number if there's a winner, or -1 for tie
        """
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not self.availables:
            # tie
            return True, -1
        return False, -1

    def get_current_player(self):
        """
        Get the current player.

        Returns:
            int: current player's identifier (1 or 2)
        """
        return self.current_player


class Game(object):
    """
    Game server to manage a match between two players.
    """

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """
        Print the board state for display.
        'X' for player1's move, 'O' for player2's move, '_' for empty.

        Args:
            board (Board): The board instance
            player1 (int): player1's id (should be 1 or 2)
            player2 (int): player2's id (should be 1 or 2)
        """
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        # print column indices
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\n')

        # print row indices and board state
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\n')
        print('\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        Start a game between two players (player1 and player2).

        Args:
            player1: an object with get_action and set_player_ind methods
            player2: same as player1
            start_player (int): 0 or 1, indicates who moves first
            is_shown (int): 1 to display the board and moves, 0 otherwise

        Returns:
            winner (int): the winner player id (1 or 2) or -1 if tie
        """
        if start_player not in (0, 1):
            raise ValueError('start_player should be either 0 (player1 first) or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """
        Start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data for training.

        Args:
            player: MCTS-based player
            is_shown (int): 1 if show the board, 0 otherwise
            temp (float): temperature parameter for MCTS action selection

        Returns:
            winner (int): winner player id or -1 if tie
            data (iterator): an iterator of (state, mcts_prob, winner_z)
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)

            end, winner = self.board.game_end()
            if end:
                # assign winner z values
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0

                # reset MCTS player for next game
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
