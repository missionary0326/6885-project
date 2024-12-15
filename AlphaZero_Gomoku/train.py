# -*- coding: utf-8 -*-
"""
Training pipeline for AlphaZero Gomoku

Author: Your Name
"""

import os
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet  # TensorFlow 2.x version
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

# Configure TensorFlow to use GPU and manage GPU memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors and warnings

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 已启用")
    except RuntimeError as e:
        print(e)
else:
    print("未检测到 GPU，使用 CPU")


class TrainPipeline():
    def __init__(self, init_model=None):
        # Game and board parameters
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
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
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 50
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000

        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)

        self.policy_value_net.compile_model(learning_rate=self.learn_rate)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

        # 增加用于记录指标的列表
        self.batch_index_list = []            # 用于记录每次更新的批次编号
        self.loss_list = []                   # 记录loss随批次的变化
        self.entropy_list = []                # 记录entropy随批次的变化
        self.kl_list = []                     # 记录kl散度随批次的变化
        self.explained_var_old_list = []      # 记录explained_var_old随批次的变化
        self.explained_var_new_list = []      # 记录explained_var_new随批次的变化
        self.win_ratio_list = []              # 记录win_ratio随检查频率的变化

    def get_equi_data(self, play_data):
        """Augment data by rotating and flipping."""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            # state: [4, H, W]
            for i in [1, 2, 3, 4]:
                # rotate
                equi_state = np.rot90(state, i, axes=(1, 2))
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
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """Generate training data through self-play."""
        for _ in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """Update the policy-value network."""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = np.array([data[2] for data in mini_batch]).reshape(-1, 1)

        # Adjust state data format [N, 4, H, W] -> [N, H, W, 4]
        state_batch = np.transpose(state_batch, (0, 2, 3, 1))

        # Get old predictions for KL divergence
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step_custom(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        # Adaptive learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(winner_batch - old_v.flatten()) / np.var(winner_batch))
        explained_var_new = (1 - np.var(winner_batch - new_v.flatten()) / np.var(winner_batch))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{}," 
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))

        # 在这里记录指标
        self.batch_index_list.append(len(self.batch_index_list) + 1)
        self.loss_list.append(loss)
        self.entropy_list.append(entropy)
        self.kl_list.append(kl)
        self.explained_var_old_list.append(explained_var_old)
        self.explained_var_new_list.append(explained_var_new)

        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """Evaluate the current policy against pure MCTS."""
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
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))

        # 记录 win_ratio
        self.win_ratio_list.append(win_ratio)

        return win_ratio

    def run(self):
        """Run the training pipeline."""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # Periodic evaluation and model saving
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.weights.h5')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model('./best_policy.weights.h5')
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

            # 训练结束后绘图
            self.plot_metrics()
        except KeyboardInterrupt:
            print('\nquit')
            # 即使被中断，也可尝试绘图
            self.plot_metrics()

    def plot_metrics(self):
        """绘制并保存loss、entropy、kl、explained_var和win_ratio的变化曲线"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(18, 10))

        # Loss曲线
        plt.subplot(3, 2, 1)
        plt.plot(self.batch_index_list, self.loss_list, label='Loss', color='blue')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Loss')
        plt.title('Loss over Training')
        plt.grid(True)
        plt.legend()

        # Entropy曲线
        plt.subplot(3, 2, 2)
        plt.plot(self.batch_index_list, self.entropy_list, label='Entropy', color='orange')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Entropy')
        plt.title('Entropy over Training')
        plt.grid(True)
        plt.legend()

        # KL曲线
        plt.subplot(3, 2, 3)
        plt.plot(self.batch_index_list, self.kl_list, label='KL Divergence', color='green')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence over Training')
        plt.grid(True)
        plt.legend()

        # Explained Variance Old曲线
        plt.subplot(3, 2, 4)
        plt.plot(self.batch_index_list, self.explained_var_old_list, label='Explained Var Old', color='red')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Explained Var Old')
        plt.title('Explained Variance (Old) over Training')
        plt.grid(True)
        plt.legend()

        # Explained Variance New曲线
        plt.subplot(3, 2, 5)
        plt.plot(self.batch_index_list, self.explained_var_new_list, label='Explained Var New', color='purple')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Explained Var New')
        plt.title('Explained Variance (New) over Training')
        plt.grid(True)
        plt.legend()

        # Win Ratio曲线
        plt.subplot(3, 2, 6)
        # Win Ratio是每check_freq批次才记录一次，因此x轴使用check_freq的倍数
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
