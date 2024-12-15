# -*- coding: utf-8 -*-
"""
Policy-Value Network implementation using TensorFlow 2.x (tf.keras)

Author: Your Name
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

class PolicyValueNet(tf.keras.Model):
    def __init__(self, board_width, board_height, model_file=None):
        super(PolicyValueNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # Define the network architecture [N, H, W, C]
        self.conv1 = layers.Conv2D(32, (3, 3), padding="same", activation='relu')
        self.conv2 = layers.Conv2D(64, (3, 3), padding="same", activation='relu')
        self.conv3 = layers.Conv2D(128, (3, 3), padding="same", activation='relu')

        # Policy head
        self.policy_conv = layers.Conv2D(4, (1, 1), padding="same", activation='relu')
        self.policy_flat = layers.Flatten()
        self.policy_fc = layers.Dense(board_height * board_width, activation='softmax')

        # Value head
        self.value_conv = layers.Conv2D(2, (1, 1), padding="same", activation='relu')
        self.value_flat = layers.Flatten()
        self.value_fc1 = layers.Dense(64, activation='relu')
        self.value_fc2 = layers.Dense(1, activation='tanh')

        if model_file:
            self.load_weights(model_file).expect_partial()

    def call(self, inputs, training=False):
        """
        Forward pass.
        Inputs: [N, H, W, C]
        Outputs: (action_probs [N, board_height*board_width], value [N, 1])
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        # Policy branch
        policy = self.policy_conv(x)
        policy = self.policy_flat(policy)
        act_probs = self.policy_fc(policy)

        # Value branch
        value = self.value_conv(x)
        value = self.value_flat(value)
        value = self.value_fc1(value)
        value = self.value_fc2(value)

        return act_probs, value

    def policy_value(self, state_batch):
        """
        Get policy and value for a batch of states.
        Args:
            state_batch: [N, H, W, C]
        Returns:
            act_probs: [N, board_height*board_width]
            value: [N, 1]
        """
        act_probs, value = self(state_batch, training=False)
        return act_probs.numpy(), value.numpy()

    def policy_value_fn(self, board):
        """
        Interface for MCTS to get action probabilities and value.
        Args:
            board: Board object
        Returns:
            action_probs: list of (action, probability)
            value: float
        """
        legal_positions = board.availables
        current_state = board.current_state().reshape(-1, self.board_height, self.board_width, 4)
        act_probs, value = self.policy_value(current_state)
        act_probs = act_probs[0]
        value = value[0][0]
        # Only return probabilities for legal moves
        act_probs = list(zip(legal_positions, act_probs[legal_positions]))
        return act_probs, value

    def train_step_custom(self, state_batch, mcts_probs, winner_batch, lr):
        """
        Custom training step.
        Args:
            state_batch: [N, H, W, C]
            mcts_probs: [N, board_height*board_width]
            winner_batch: [N, 1]
            lr: learning rate
        Returns:
            loss: float
            entropy: float
        """
        with tf.GradientTape() as tape:
            act_probs, value = self(state_batch, training=True)
            # Value loss
            value_loss = tf.reduce_mean(tf.square(winner_batch - value))
            # Policy loss
            policy_loss = -tf.reduce_mean(tf.reduce_sum(mcts_probs * tf.math.log(act_probs + 1e-10), axis=1))
            # Total loss
            loss = value_loss + policy_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # Calculate entropy for monitoring
        entropy = -tf.reduce_mean(tf.reduce_sum(act_probs * tf.math.log(act_probs + 1e-10), axis=1))
        return loss.numpy(), entropy.numpy()

    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer.
        Args:
            learning_rate (float): learning rate for the optimizer
        """
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        # Compile with dummy loss to utilize Keras functionalities if needed
        self.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def save_model(self, model_path):
        """
        Save model weights.
        Args:
            model_path (str): path to save the model, should end with .h5
        """
        if not model_path.endswith('.h5'):
            model_path += '.h5'
        self.save_weights(model_path)

    def restore_model(self, model_path):
        """
        Load model weights.
        Args:
            model_path (str): path to the saved model
        """
        self.load_weights(model_path).expect_partial()
