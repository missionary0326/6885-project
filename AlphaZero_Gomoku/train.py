# -*- coding: utf-8 -*-
"""
Training pipeline for AlphaZero Gomoku with plotting metrics

This example records and plots the following metrics:
- Loss
- Entropy
- KL Divergence (kl)
- Explained Variance (explained_var_old and explained_var_new)
- Win Ratio against pure MCTS opponent

After the training finishes, it saves line charts for these metrics.
"""

import os
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import datetime

# Reduce TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(filename='training.log',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

class TrainPipeline():
    def __init__(self, init_model=None):
        # Game and board parameters
        self.board_width = 10
        self.board_height = 10
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # Training parameters (unchanged)
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 50
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 500
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000

        # Initialize policy-value net
        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.policy_value_net.compile_model(learning_rate=self.learn_rate)

        # MCTS player
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

        # Record metrics
        self.batch_index_list = []
        self.loss_list = []
        self.entropy_list = []
        self.kl_list = []
        self.explained_var_old_list = []
        self.explained_var_new_list = []
        self.win_ratio_list = []

    def get_equi_data(self, play_data):
        """Data augmentation by rotation and flipping"""
        # 尝试减少Python循环层的一些重复操作
        # 这里保持原逻辑不变，但可考虑在将来进一步向量化处理
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate
                equi_state = np.rot90(state, i, axes=(1,2))
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.fliplr(equi_state)
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        # 返回扩增后的数据
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """Generate training data through self-play"""
        for _ in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            # data augmentation
            play_data = self.get_equi_data(play_data)
            # 在这里提前进行numpy转换和数据类型设置减少后续转换
            # 但不改变整体逻辑和参数
            for i, (state, prob, w) in enumerate(play_data):
                # 保持现有逻辑不变，仅确保类型转换快速
                play_data[i] = (state.astype(np.float32), 
                                prob.astype(np.float32), 
                                np.float32(w))
            self.data_buffer.extend(play_data)
            logging.info(f"Collected self-play data for game with winner: {winner}")

    def policy_update(self):
        """Update policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch], dtype=np.float32)
        mcts_probs_batch = np.array([data[1] for data in mini_batch], dtype=np.float32)
        winner_batch = np.array([data[2] for data in mini_batch], dtype=np.float32).reshape(-1, 1)

        # [N,4,H,W] -> [N,H,W,4]
        # 如果数据提前存储为H,W,4格式可省略这一步，此处保持逻辑不变
        state_batch = np.transpose(state_batch, (0, 2, 3, 1))

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step_custom(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(
                old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1
            ))
            if kl > self.kl_targ * 4:
                logging.warning(f"Early stopping at epoch {i+1} due to KL {kl}")
                break

        # Adaptive learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
            logging.info(f"Learning rate multiplier decreased to {self.lr_multiplier}")
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
            logging.info(f"Learning rate multiplier increased to {self.lr_multiplier}")

        explained_var_old = (1 - np.var(winner_batch - old_v.flatten()) / np.var(winner_batch))
        explained_var_new = (1 - np.var(winner_batch - new_v.flatten()) / np.var(winner_batch))
        logging.info(("kl:{:.5f}, "
                      "lr_multiplier:{:.3f}, "
                      "loss:{}, "
                      "entropy:{}, " 
                      "explained_var_old:{:.3f}, "
                      "explained_var_new:{:.3f}"
                      ).format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))

        print(("kl:{:.5f}, "
               "lr_multiplier:{:.3f}, "
               "loss:{:.6f}, "
               "entropy:{:.6f}, " 
               "explained_var_old:{:.3f}, "
               "explained_var_new:{:.3f}"
               ).format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))

        # Record metrics
        self.kl_list.append(kl)
        self.explained_var_old_list.append(explained_var_old)
        self.explained_var_new_list.append(explained_var_new)

        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """Evaluate current policy against pure MCTS"""
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=5,
                                      n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
            logging.info(f"Game {i+1}: Winner is {winner}")

        win_ratio = (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        logging.info(f"Evaluation after {n_games} games: win_ratio={win_ratio}")
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))

        self.win_ratio_list.append(win_ratio)
        return win_ratio

    def run(self):
        """Run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                logging.info(f"batch i:{i+1}, episode_len:{self.episode_len}")

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    self.batch_index_list.append(i+1)
                    self.loss_list.append(loss)
                    self.entropy_list.append(entropy)

                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    logging.info(f"current self-play batch: {i+1}")
                    win_ratio = self.policy_evaluate()
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    current_model_path = f'./current_policy_{timestamp}.weights.h5'
                    self.policy_value_net.save_model(current_model_path)
                    logging.info(f"Saved current policy model to {current_model_path}")

                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        logging.info("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        best_model_path = f'./best_policy_{timestamp}.weights.h5'
                        self.policy_value_net.save_model(best_model_path)
                        logging.info(f"Saved best policy model to {best_model_path}")

                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                            logging.info(f"Increased pure MCTS playout num to {self.pure_mcts_playout_num}")

            self.plot_metrics()
        except KeyboardInterrupt:
            print('\nquit')
            logging.info('Training interrupted by user.')

    def plot_metrics(self):
        """Plot and save training metrics"""
        plt.figure(figsize=(18, 10))

        # Loss
        plt.subplot(3, 2, 1)
        plt.plot(self.batch_index_list, self.loss_list, label='Loss', color='blue')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Loss')
        plt.title('Loss over Training')
        plt.grid(True)
        plt.legend()

        # Entropy
        plt.subplot(3, 2, 2)
        plt.plot(self.batch_index_list, self.entropy_list, label='Entropy', color='orange')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Entropy')
        plt.title('Entropy over Training')
        plt.grid(True)
        plt.legend()

        # KL Divergence
        plt.subplot(3, 2, 3)
        plt.plot(self.batch_index_list, self.kl_list, label='KL Divergence', color='green')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence over Training')
        plt.grid(True)
        plt.legend()

        # Explained Variance Old
        plt.subplot(3, 2, 4)
        plt.plot(self.batch_index_list, self.explained_var_old_list, label='Explained Var Old', color='red')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Explained Variance Old')
        plt.title('Explained Variance (Old) over Training')
        plt.grid(True)
        plt.legend()

        # Explained Variance New
        plt.subplot(3, 2, 5)
        plt.plot(self.batch_index_list, self.explained_var_new_list, label='Explained Var New', color='purple')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Explained Variance New')
        plt.title('Explained Variance (New) over Training')
        plt.grid(True)
        plt.legend()

        # Win Ratio
        plt.subplot(3, 2, 6)
        # Win Ratio在check_freq时才记录，因此x轴使用check_freq的倍数
        x_vals = [self.check_freq * (i+1) for i in range(len(self.win_ratio_list))]
        plt.plot(x_vals, self.win_ratio_list, label='Win Ratio', color='brown')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Win Ratio')
        plt.title('Win Ratio over Training')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
        print("Training metrics saved to 'training_metrics.png' and displayed.")


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
