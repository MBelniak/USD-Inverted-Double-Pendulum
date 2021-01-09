import threading
import time
from threading import Lock
from matplotlib import pyplot as plt
import torch
import numpy as np

from algorithms.A3C.A3CWorker import A3CWorker
from algorithms.A3C.Actor import Actor
from algorithms.A3C.Critic import Critic
import gym
import os

from algorithms.Model import Model
from logger.logger import a3c_logger
from utils import get_default_save_filename, ensure_unique_path, ENV_NAME, t

SAVE_DIR = "saved_models"
PLOTS_DIR = "plots"


class A3C(Model):
    # for PyCharm to resolve it correctly in A3CWorker
    lock, discount_rate = {}, {}
    step_max, Actor, Critic, workers = {}, {}, {}, []

    # Actor-Critic Main Optimization Algorithm
    def __init__(self, max_episodes=100000, discount_rate=0.99, step_max=5, actor_lr=0.001, critic_lr=0.001,
                 n_threads=5, measure_step=100, eval_repeats=20, no_log=True):
        super().__init__()
        self.env = gym.make(ENV_NAME)
        self.action_space = self.env.action_space
        self.action_size = self.action_space.shape[0]
        self.max_episodes, self.episode = max_episodes, 0
        self.lock = Lock()
        self.actor_learning_rate = actor_lr
        self.critic_learning_rate = critic_lr
        self.n_threads = n_threads
        self.discount_rate = discount_rate
        self.step_max = step_max
        self.measure_step = measure_step
        self.eval_repeats = eval_repeats
        self.no_log = no_log

        # Create Actor-Critic network models
        self.Actor = Actor(state_space=self.env.observation_space, learning_rate=self.actor_learning_rate,
                           action_space=self.action_space)
        self.Critic = Critic(state_space=self.env.observation_space, learning_rate=self.critic_learning_rate)

        self.Actor.model.train()
        self.Critic.model.train()

    def set_workers(self, workers):
        self.workers = workers
        for worker in workers:
            worker.set_global_model(self)

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
        path = SAVE_DIR + "/" + get_default_save_filename(self.max_episodes, self.n_threads, self.discount_rate,
                                                          self.step_max,
                                                          self.actor_learning_rate, self.critic_learning_rate)
        path = ensure_unique_path(path)

        torch.save({
            'Actor_state_dict': self.Actor.model.state_dict(),
            'Critic_state_dict': self.Critic.model.state_dict(),
            'Actor_opt_state_dict': self.Actor.optimizer.state_dict(),
            'Critic_opt_state_dict': self.Critic.optimizer.state_dict()
        }, path)

    def run(self):
        # Instantiate one worker per thread
        workers = [A3CWorker(self.lock, max_episodes=self.max_episodes, discount_rate=self.discount_rate,
                             step_max=self.step_max, measure_step=self.measure_step, log_info=not self.no_log,
                             eval_repeats=self.eval_repeats)
                   for _ in range(self.n_threads)]

        self.set_workers(workers)
        self.env.close()

        # Create threads
        threads = [threading.Thread(
            target=workers[i].run,
            daemon=True) for i in range(self.n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()

        while self.episode < self.max_episodes:
            time.sleep(1)

        for t in threads:
            t.join()

        return np.array(self.performance)

    def evaluate(self, eval_repeats=20):
        self.Actor.model.eval()
        self.Critic.model.eval()
        scores = []
        for ep in range(eval_repeats):
            state = self.env.reset()
            done = False
            performance = 0
            while not done:
                action = self.Actor.get_best_action(t(state))
                state, reward, done, _ = self.env.step(action)
                performance += reward

            scores.append([ep, performance])

        scores = np.array(scores)
        self.Actor.model.train()
        self.Critic.model.train()
        return scores[:, 1].mean(), scores

    def test(self):
        a3c_logger.info(f"Starting test of A3C after {self.max_episodes} episodes of training.")
        mean, performance = self.evaluate(50)
        a3c_logger.info(f"Mean accumulated score: {mean}")
        return np.array(performance)

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
                    a3c_logger.info("episode: {}, score: {}".format(e, score))
                    break
        self.env.close()

    def get_plot_dir(self, subdir):
        plot_dir = PLOTS_DIR + "/" + subdir + "/" + get_default_save_filename(self.max_episodes, len(self.workers),
                                                                              self.discount_rate,
                                                                              self.step_max, self.actor_learning_rate,
                                                                              self.critic_learning_rate)
        plot_dir = ensure_unique_path(plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

        return plot_dir

    """
            Plots average of accumulated rewards over certain amount of repetitions in a training phase.
    """
    def plot_training(self, performance):
        plot_dir = self.get_plot_dir("training")
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.set_title(f"Performance in a training phase - accumulated rewards until a terminal state", fontsize=24)
        ax.set_xlabel("Episode", fontsize=22)
        ax.set_ylabel("Mean accumulated rewards ", fontsize=22)
        ax.plot(performance[:, 0], performance[:, 1], 'o')
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        fig.savefig(plot_dir + f"/training.png")
        plt.close(fig)

    """
            Plots accumulated rewards from start until a terminal state in a test phase.
    """
    def plot_test(self, performance):
        plot_dir = self.get_plot_dir("test")
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.set_title(f"Performance in a test phase - accumulated rewards until a terminal state", fontsize=24)
        ax.set_xlabel("Run", fontsize=22)
        ax.set_ylabel("Accumulated rewards", fontsize=22)
        ax.plot(performance[:, 0], performance[:, 1], 'o')
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        fig.savefig(plot_dir + f"/test.png")
        plt.close(fig)

    """
        Plots accumulated rewards over an episode (from init state to terminal state).
        Creates one plot for each thread + one plot with all threads together.
        In addition, creates one plot of moving average for each thread + moving average of average over threads
    """
    def plot_workers(self):
        plot_dir = self.get_plot_dir("workers")

        fig, ax_gather = plt.subplots(figsize=(18, 9))
        ax_gather.set_title("Scores until terminal state during training for all threads.", fontsize=24)
        ax_gather.set_xlabel("Local episode", fontsize=22)
        ax_gather.set_ylabel("Score", fontsize=22)

        for worker in self.workers:
            ax_gather.plot(np.array(worker.performance)[:, 0], np.array(worker.performance)[:, 1])

        ax_gather.set_ylim(ymin=0)
        ax_gather.set_xlim(xmin=0)
        fig.savefig(plot_dir + "/gathered.png")
        plt.close(fig)

        scores_lengths = np.array([len(np.array(worker.performance)[:, 1]) for worker in self.workers])
        min_len = np.min(scores_lengths)
        scores = [np.array(worker.performance)[:, 1] for worker in self.workers]
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

        for i, worker in enumerate(self.workers):
            fig, ax = plt.subplots(figsize=(18, 9))
            ax.set_title(f"Scores until terminal state during training for thread {i + 1}", fontsize=24)
            ax.set_xlabel("Local episode", fontsize=22)
            ax.set_ylabel("Score", fontsize=22)
            ax.plot(np.array(worker.performance)[:, 0], np.array(worker.performance)[:, 1])
            ax.set_ylim(ymin=0)
            ax.set_xlim(xmin=0)
            fig.savefig(plot_dir + f"/thread{i + 1}.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(18, 9))
            ax.set_title(f"Moving average of scores until terminal state for thread {i + 1}", fontsize=24)
            ax.set_xlabel("Local episode", fontsize=22)
            ax.set_ylabel("Score", fontsize=22)
            y = np.convolve(np.array(worker.performance)[:, 1], np.ones(10) / 10, mode='valid')
            ax.plot(np.array(worker.performance)[:len(y), 0], y)
            ax.set_ylim(ymin=0)
            ax.set_xlim(xmin=0)
            fig.savefig(plot_dir + f"/moving_av_thread{i + 1}.png")
            plt.close(fig)
