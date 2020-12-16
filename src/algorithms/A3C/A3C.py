from threading import Lock
from matplotlib import pyplot as plt
import torch
import numpy as np
from algorithms.A3C.Actor import Actor
from algorithms.A3C.Critic import Critic
import gym
import os

from logger.logger import a3c_logger
from utils import get_default_save_filename, ensure_unique_path, ENV_NAME, t

SAVE_DIR = "saved_models"
PLOTS_DIR = "plots"


class A3C:
    # for PyCharm to resolve it correctly in A3CWorker
    MAX_EPISODES, lock, learning_rate, discount_rate = {}, {}, {}, {}
    step_max, Actor, Critic, workers = {}, {}, {}, []

    # Actor-Critic Main Optimization Algorithm
    def __init__(self, max_episodes=100000, discount_rate=0.99, step_max=5, actor_lr=0.001, critic_lr=0.001, n_threads=5):
        self.env = gym.make(ENV_NAME)
        self.action_space = self.env.action_space
        self.action_size = self.action_space.shape[0]
        self.MAX_EPISODES, self.episode = max_episodes, 0
        self.lock = Lock()
        self.actor_learning_rate = actor_lr
        self.critic_learning_rate = critic_lr
        self.n_threads = n_threads
        self.discount_rate = discount_rate
        self.step_max = step_max
        # Instantiate plot memory
        self.scores, self.episodes = [], []

        # Create Actor-Critic network models
        self.Actor = Actor(state_space=self.env.observation_space, learning_rate=self.actor_learning_rate,
                           action_space=self.action_space)
        self.Critic = Critic(state_space=self.env.observation_space, learning_rate=self.critic_learning_rate)

        # just pytorch stuff
        self.Actor.model.train()
        self.Critic.model.train()

    def set_workers(self, workers):
        self.workers = workers

    def load_models(self, file_name):
        checkpoint = torch.load(f"{SAVE_DIR}/{file_name}")
        self.Actor.model.load_state_dict(checkpoint['Actor_state_dict'])
        self.Critic.model.load_state_dict(checkpoint['Critic_state_dict'])
        self.Actor.optimizer.load_state_dict(checkpoint['Actor_opt_state_dict'])
        self.Critic.optimizer.load_state_dict(checkpoint['Critic_opt_state_dict'])
        self.Actor.model.eval()
        self.Critic.model.eval()

    def save_models(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = SAVE_DIR + "/" + get_default_save_filename(self.MAX_EPISODES, self.n_threads, self.discount_rate, self.step_max,
                                                          self.actor_learning_rate, self.critic_learning_rate)
        path = ensure_unique_path(path)

        torch.save({
            'Actor_state_dict': self.Actor.model.state_dict(),
            'Critic_state_dict': self.Critic.model.state_dict(),
            'Actor_opt_state_dict': self.Actor.optimizer.state_dict(),
            'Critic_opt_state_dict': self.Critic.optimizer.state_dict()
        }, path)

    def plot(self):
        plot_dir = PLOTS_DIR + "/" + get_default_save_filename(self.MAX_EPISODES, len(self.workers), self.discount_rate,
                                                               self.step_max, self.actor_learning_rate,
                                                               self.critic_learning_rate)
        plot_dir = ensure_unique_path(plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

        self.plot_workers(plot_dir)
        self.plot_averages(plot_dir)
        self.plot_test(plot_dir)

    """
        Plots accumulated rewards over an episode (from init state to terminal state).
        Creates one plot for each thread + one plot with all threads together
    """
    def plot_workers(self, plot_dir):
        fig, ax_gather = plt.subplots(figsize=(18, 9))
        ax_gather.set_title("Scores until terminal state for all threads.", fontsize=24)
        ax_gather.set_xlabel("Local episode", fontsize=22)
        ax_gather.set_ylabel("Score", fontsize=22)

        for worker in self.workers:
            ax_gather.plot(worker.episodes, worker.scores)

        ax_gather.set_ylim(ymin=0)
        ax_gather.set_xlim(xmin=0)
        fig.savefig(plot_dir + "/gathered.png")
        plt.close(fig)

        for i in range(len(self.workers)):
            fig, ax = plt.subplots(figsize=(18, 9))
            ax.set_title(f"Scores until terminal state for thread {i + 1}", fontsize=24)
            ax.set_xlabel("Local episode", fontsize=22)
            ax.set_ylabel("Score", fontsize=22)
            ax.plot(self.workers[i].episodes, self.workers[i].scores)
            ax.set_ylim(ymin=0)
            ax.set_xlim(xmin=0)
            fig.savefig(plot_dir + f"/thread{i + 1}.png")
            plt.close(fig)

    """
        Plots average of accumulated rewards over all threads for each episode (from init state to terminal state).
        In addition, creates one plot of moving average for each thread + moving average of average over threads
    """
    def plot_averages(self, plot_dir):
        scores_lengths = np.array([len(worker.scores) for worker in self.workers])
        min_len = np.min(scores_lengths)
        scores = [worker.scores for worker in self.workers]
        averages = [np.mean(list(zip(*scores))[i]) for i in range(min_len)]

        fig, ax_gather = plt.subplots(figsize=(18, 9))
        ax_gather.set_title("Average scores until terminal state over all threads.", fontsize=24)
        ax_gather.set_xlabel("Local episode", fontsize=22)
        ax_gather.set_ylabel("Score", fontsize=22)
        ax_gather.plot([i for i in range(min_len)], averages, 'r')
        ax_gather.set_ylim(ymin=0)
        ax_gather.set_xlim(xmin=0)
        fig.savefig(plot_dir + "/average.png")
        plt.close(fig)

        fig, ax_mov_av = plt.subplots(figsize=(18, 9))
        ax_mov_av.set_title("Moving average of average score until terminal state over all threads.", fontsize=24)
        ax_mov_av.set_xlabel("Local episode", fontsize=22)
        ax_mov_av.set_ylabel("Score", fontsize=22)
        y = np.convolve(averages, np.ones(10) / 10, mode='valid')
        ax_mov_av.plot([i for i in range(len(y))], y, 'r')
        ax_mov_av.set_ylim(ymin=0)
        ax_mov_av.set_xlim(xmin=0)
        fig.savefig(plot_dir + "/moving_av_gathered.png")
        plt.close(fig)

        for i in range(len(self.workers)):
            fig, ax = plt.subplots(figsize=(18, 9))
            ax.set_title(f"Moving average of scores until terminal state for thread {i + 1}", fontsize=24)
            ax.set_xlabel("Local episode", fontsize=22)
            ax.set_ylabel("Score", fontsize=22)
            y = np.convolve(self.workers[i].scores, np.ones(10) / 10, mode='valid')
            ax.plot(self.workers[i].episodes[:len(y)], y)
            ax.set_ylim(ymin=0)
            ax.set_xlim(xmin=0)
            fig.savefig(plot_dir + f"/moving_av_thread{i + 1}.png")
            plt.close(fig)

    def plot_test(self, plot_dir):
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.set_title(f"Scores until terminal state for test run", fontsize=24)
        ax.set_xlabel("Episode", fontsize=22)
        ax.set_ylabel("Score", fontsize=22)
        ax.plot(self.episodes, self.scores, 'o')
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        fig.savefig(plot_dir + f"/test.png")
        plt.close(fig)

    def test(self):
        self.scores, self.episodes = [], []
        a3c_logger.info(f"Starting test of A3C after {self.MAX_EPISODES} episodes of training.")
        for e in range(100):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.Actor.get_best_action(t(state))
                state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    self.scores.append(score)
                    self.episodes.append(e)
                    print("Episode: {}/100, score: {}".format(e + 1, score))
                    break

    """
        Render environment and run trained Actor on it.
    """
    def render(self):
        for e in range(10):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                self.env.render()
                action = self.Actor.get_best_action(t(state))
                state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    print("episode: {}, score: {}".format(e, score))
                    break
        self.env.close()
