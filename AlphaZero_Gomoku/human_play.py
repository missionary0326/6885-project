# -*- coding: utf-8 -*-
"""
Human VS AI for Gomoku (AlphaZero-based model)

Instructions:
-------------
- When the program starts, you will be prompted to input your move in the format: "row,column"
  For example: Enter "2,3" to place a piece at row 2, column 3 (0-indexed).
- Press Enter to confirm your move, then the AI will make its move.
- The game continues until it ends with a win or a tie.
- `start_player=1` means AI plays first. Change to 0 if you want to play first.

Note:
-----
- Ensure TensorFlow 2.x is installed and GPU is enabled (if desired).
- Ensure the following files are in the same directory:
  - game.py: Defines the board and game logic
  - mcts_alphaZero.py: AlphaZero MCTS player implementation
  - policy_value_net_tensorflow.py: TensorFlow version of the policy-value network
- Ensure the best policy model file (e.g., 'best_policy.weights.h5') is present.
"""

import os
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet  # TensorFlow 2.x version

class HumanPlayer(object):
    """
    Human player class.
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        """
        Set the player index (1 or 2).
        """
        self.player = p

    def get_action(self, board):
        """
        Get human player's action based on input.
        """
        try:
            location = input("Your move (row,column): ")
            if isinstance(location, str):  # For Python 3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            print(f"Input error: {e}")
            move = -1

        if move == -1 or move not in board.availables:
            print("Invalid move. Please try again.")
            return self.get_action(board)
        return move

    def __str__(self):
        return f"Human {self.player}"

def run():
    # Game parameters
    n = 5          # Number of pieces in a row needed to win
    width, height = 10, 10  # Board size
    model_file = 'best_policy.weights.h5'  # Trained model file

    # Check if the model file exists
    if not os.path.exists(model_file):
        print(f"Model file '{model_file}' not found. Please train the model first.")
        return

    try:
        # Initialize the board and game
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # Load the trained policy-value network
        best_policy = PolicyValueNet(width, height, model_file=model_file)
        best_policy.build(input_shape=(None, height, width, 4))  # Build the model with input shape
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # Adjust n_playout as needed

        # Initialize human player
        human = HumanPlayer()

        # Start the game: start_player=1 (AI first), set to 0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    run()
