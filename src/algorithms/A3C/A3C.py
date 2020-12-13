from threading import Lock
from matplotlib import pyplot as plt
import torch
import numpy as np
from utils import ENV_NAME, t
from algorithms.A3C.Actor import Actor
from algorithms.A3C.Critic import Critic
import gym
import os

SAVE_DIR = "saved_models"
PLOTS_DIR = "plots"


def get_default_save_filename(episodes, threads, discount, step_max, actor_lr, critic_lr):
    return f"A3C--{episodes}-{threads}-{str(discount).replace('.', '_')}-{str(step_max).replace('.', '_')}-" \
           f"{str(actor_lr).replace('.', '_')}-{str(critic_lr).replace('.', '_')}"


def ensure_unique_path(path):
    if os.path.exists(path):
        counter = 1
        while os.path.exists(path + f"({str(counter)})"):
            counter += 1
        return path + f"({str(counter)})"
    return path


class A3C:
    # for PyCharm to resolve it correctly in A3CWorker
    MAX_EPISODES, lock, learning_rate, discount_rate = {}, {}, {}, {}
    step_max, Actor, Critic = {}, {}, {}

    # Actor-Critic Main Optimization Algorithm
    def __init__(self, max_episodes=100000, discount_rate=0.99, step_max=5, actor_lr=0.001, critic_lr=0.001):
        self.env = gym.make(ENV_NAME)
        self.action_space = self.env.action_space
        self.action_size = self.action_space.shape[0]
        self.MAX_EPISODES, self.episode = max_episodes, 0
        self.lock = Lock()
        self.actor_learning_rate = actor_lr
        self.critic_learning_rate = critic_lr
        self.discount_rate = discount_rate
        self.step_max = step_max
        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        # Create Actor-Critic network model
        self.Actor = Actor(state_space=self.env.observation_space, learning_rate=self.actor_learning_rate,
                           action_space=self.action_space)
        self.Critic = Critic(state_space=self.env.observation_space, learning_rate=self.critic_learning_rate)

        # just pytorch stuff
        self.Actor.model.train()
        self.Critic.model.train()

    def load_models(self, file_name):
        checkpoint = torch.load(f"{SAVE_DIR}/{file_name}")
        self.Actor.model.load_state_dict(checkpoint['Actor_state_dict'])
        self.Critic.model.load_state_dict(checkpoint['Critic_state_dict'])
        self.Actor.optimizer.load_state_dict(checkpoint['Actor_opt_state_dict'])
        self.Critic.optimizer.load_state_dict(checkpoint['Critic_opt_state_dict'])
        self.Actor.model.eval()
        self.Critic.model.eval()

    def save_models(self, n_threads):
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = SAVE_DIR + "/" + get_default_save_filename(self.MAX_EPISODES, n_threads, self.discount_rate, self.step_max,
                                                          self.actor_learning_rate, self.critic_learning_rate)
        path = ensure_unique_path(path)

        torch.save({
            'Actor_state_dict': self.Actor.model.state_dict(),
            'Critic_state_dict': self.Critic.model.state_dict(),
            'Actor_opt_state_dict': self.Actor.optimizer.state_dict(),
            'Critic_opt_state_dict': self.Critic.optimizer.state_dict()
        }, path)

    def plot(self, workers):
        plot_dir = PLOTS_DIR + "/" + get_default_save_filename(self.MAX_EPISODES, len(workers), self.discount_rate,
                                                               self.step_max, self.actor_learning_rate,
                                                               self.critic_learning_rate)
        plot_dir = ensure_unique_path(plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

        self.plot_workers(plot_dir, workers)
        self.plot_averages(plot_dir, workers)

    # TODO
    # def test(self, actor_name, critic_name):
    #     #     self.load(actor_name, critic_name)
    #     #     for e in range(100):
    #     #         state = self.env.reset()
    #     #         done = False
    #     #         score = 0
    #     #         while not done:
    #     #             action = np.argmax(self.Actor.get_action(state))
    #     #             state, reward, done, _ = self.env.step(action)
    #     #             score += reward
    #     #             if done:
    #     #                 print("episode: {}/{}, score: {}".format(e, self.MAX_EPISODES, score))
    #     #                 break
    #     #     self.env.close()

    """
        Plots accumulated rewards over an episode (from init state to terminal state).
        Creates one plot for each thread + one plot with all threads together
    """
    def plot_workers(self, plot_dir, workers):
        fig_gather, ax_gather = plt.subplots(figsize=(18, 9))
        ax_gather.set_title("Scores until terminal state for all threads.", fontsize=24)
        ax_gather.set_xlabel("Local episode", fontsize=22)
        ax_gather.set_ylabel("Score", fontsize=22)

        for worker in workers:
            ax_gather.plot(worker.episodes, worker.scores)

        ax_gather.set_ylim(ymin=0)
        fig_gather.savefig(plot_dir + "/gathered.png")

        for i in range(len(workers)):
            fig, ax = plt.subplots(figsize=(18, 9))
            ax.set_title(f"Scores until terminal state for thread {i + 1}", fontsize=24)
            ax.set_xlabel("Local episode", fontsize=22)
            ax.set_ylabel("Score", fontsize=22)
            ax.plot(workers[i].episodes, workers[i].scores)
            ax.set_ylim(ymin=0)
            fig.savefig(plot_dir + f"/thread{i + 1}.png")

    """
        Plots average of accumulated rewards over all threads for each episode (from init state to terminal state).
        In addition, creates one plot of moving average for each thread + moving average of average over threads
    """
    def plot_averages(self, plot_dir, workers):
        scores_lengths = np.array([len(worker.scores) for worker in workers])
        min_len = np.min(scores_lengths)
        scores = [worker.scores for worker in workers]
        averages = [np.mean(list(zip(*scores))[i]) for i in range(min_len)]

        fig_gather, ax_gather = plt.subplots(figsize=(18, 9))
        ax_gather.set_title("Average scores until terminal state over all threads.", fontsize=24)
        ax_gather.set_xlabel("Local episode", fontsize=22)
        ax_gather.set_ylabel("Score", fontsize=22)
        ax_gather.plot([i for i in range(min_len)], averages, 'r')
        ax_gather.set_ylim(ymin=0)
        fig_gather.savefig(plot_dir + "/average.png")

        fig_mov_av, ax_mov_av = plt.subplots(figsize=(18, 9))
        ax_mov_av.set_title("Moving average of average score until terminal state for all threads.", fontsize=24)
        ax_mov_av.set_xlabel("Local episode", fontsize=22)
        ax_mov_av.set_ylabel("Score", fontsize=22)
        y = np.convolve(averages, np.ones(10) / 10, mode='valid')
        ax_mov_av.plot([i for i in range(len(y))], y, 'r')
        ax_mov_av.set_ylim(ymin=0)
        fig_mov_av.savefig(plot_dir + "/moving_av_gathered.png")

        for i in range(len(workers)):
            fig, ax = plt.subplots(figsize=(18, 9))
            ax.set_title(f"Moving average of scores until terminal state for thread {i + 1}", fontsize=24)
            ax.set_xlabel("Local episode", fontsize=22)
            ax.set_ylabel("Score", fontsize=22)
            y = np.convolve(workers[i].scores, np.ones(10) / 10, mode='valid')
            ax.plot(workers[i].episodes[:len(y)], y)
            ax.set_ylim(ymin=0)
            fig.savefig(plot_dir + f"/moving_av_thread{i + 1}.png")

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
                action = self.Actor.get_action(t(state))
                state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    print("episode: {}, score: {}".format(e, score))
                    break
        self.env.close()
