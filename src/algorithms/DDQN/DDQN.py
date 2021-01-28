import os

import gym
import torch
import numpy as np
import random

from algorithms.DDQN.Memory import Memory
from algorithms.DDQN.QNetwork import QNetwork
from algorithms.Model import Model
from logger.logger import dqn_logger
from utils import ensure_unique_path, ENV_NAME
from matplotlib import pyplot as plt

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "saved_models"
PLOTS_DIR = "plots"


class DDQN(Model):
    """
    :return: the trained Q-Network and the measured performances
    """
    def __init__(self, kwargs):
        super().__init__()
        self.discount_rate = kwargs['discount_rate']
        self.learning_rate = kwargs['lr']
        self.min_episodes = kwargs['min_episodes']
        self.eps = kwargs['eps']
        self.eps_decay = kwargs['eps_decay']
        self.eps_min = kwargs['eps_min']
        self.update_step = kwargs['update_step']
        self.batch_size = kwargs['batch_size']
        self.update_repeats = kwargs['update_repeats']
        self.max_episodes = kwargs['episodes']
        self.seed = kwargs['seed']
        self.max_memory_size = kwargs['max_memory_size']
        self.measure_step = kwargs['measure_step']
        self.measure_repeats = kwargs['measure_repeats']
        self.hidden_dim = kwargs['hidden_dim']
        self.horizon = kwargs['horizon']
        self.do_render = kwargs['render']
        self.render_step = kwargs['render_step']
        self.num_actions = kwargs['num_actions']
        self.load_file = kwargs['load_file']
        self.env = gym.make(ENV_NAME)
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        self.discretized_actions = [
            (((self.env.action_space.high[0] - self.env.action_space.low[0]) * i / (self.num_actions - 1)) - 1)
            for i in range(self.num_actions)]

        self.primary_q = QNetwork(action_dim=self.num_actions, state_dim=self.env.observation_space.shape[0],
                                  hidden_dim=self.hidden_dim).to(device)
        self.target_q = QNetwork(action_dim=self.num_actions, state_dim=self.env.observation_space.shape[0],
                                 hidden_dim=self.hidden_dim).to(device)

        if(self.load_file != None):
            self.load_models(self.load_file)

    def load_models(self, file_name=None):
        if file_name == None:
            file_name = f"DDQN-{self.max_episodes}"

        checkpoint = torch.load(f"{SAVE_DIR}/{file_name}")
        self.primary_q.load_state_dict(checkpoint['primary_q'])
        self.target_q.load_state_dict(checkpoint['target_q'])
        self.primary_q.eval()
        self.target_q.eval()

    def save_models(self, file_name=None):
        if file_name is None:
            file_name = f"DDQN-{self.max_episodes}"

        os.makedirs(SAVE_DIR, exist_ok=True)
        path = SAVE_DIR + "/" + file_name
        path = ensure_unique_path(path)

        torch.save({
            'primary_q': self.primary_q.state_dict(),
            'target_q': self.target_q.state_dict(),
        }, path)

    def select_action(self, model, state, eps):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = model(state)

        # select a random action wih probability eps
        if random.random() <= eps:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(values.cpu().numpy())

        return action

    def train(self, batch_size, primary, target, optim, memory, discount_rate):

        states, actions, next_states, rewards, is_done = memory.sample(batch_size)

        q_values = primary(states)

        next_q_values = primary(next_states)
        next_q_state_values = target(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + discount_rate * next_q_value * (1 - is_done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

    def evaluate(self, eval_repeats):
        """
        Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
        episode reward.
        """
        self.primary_q.eval()
        scores = []
        for e in range(eval_repeats):
            state = self.env.reset()
            done = False
            perform = 0
            while not done:
                state = torch.Tensor(state).to(device)
                with torch.no_grad():
                    values = self.primary_q(state)
                action = np.argmax(values.cpu().numpy())
                state, reward, done, _ = self.env.step(self.discretized_actions[action])
                perform += reward

            scores.append([e, perform])

        scores = np.array(scores)
        self.primary_q.train()
        return scores[:,1].mean(), scores

    def update_parameters(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def run(self):
        # transfer parameters from Q_1 to Q_2
        self.update_parameters(self.primary_q, self.target_q)

        # we only train Q_1
        for param in self.target_q.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(self.primary_q.parameters(), lr=self.learning_rate)

        memory = Memory(self.max_memory_size)
        self.performance = []

        for episode in range(self.max_episodes):
            # display the performance
            if episode % self.measure_step == 0:
                mean, _ = self.evaluate(self.measure_repeats)
                self.performance.append([episode, mean])
                dqn_logger.info(f"\nEpisode: {episode}\nMean accumulated reward: {mean}\neps: {self.eps}")

            state = self.env.reset()
            memory.state.append(state)

            done = False
            i = 0
            while not done:
                i += 1
                action = self.select_action(self.target_q, state, self.eps)
                state, reward, done, _ = self.env.step(self.discretized_actions[action])
                if i > self.horizon:
                    done = True

                # render the environment if render == True
                if self.do_render and episode % self.render_step == 0:
                    self.env.render()

                # save state, action, reward sequence
                memory.update(state, action, reward, done)


            if episode >= self.min_episodes and episode % self.update_step == 0:
                for _ in range(self.update_repeats):
                    self.train(self.batch_size, self.primary_q, self.target_q, optimizer, memory, self.discount_rate)

                # transfer new parameter from Q_1 to Q_2
                self.update_parameters(self.primary_q, self.target_q)

            self.eps = max(self.eps * self.eps_decay, self.eps_min)

        return np.array(self.performance)

    def render(self):
        scores = []
        for e in range(10):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                self.env.render()
                action = self.select_action(self.primary_q, state, 0, )
                state, reward, done, _ = self.env.step(self.discretized_actions[action])
                score += reward
                if done:
                    dqn_logger.info("episode: {}, score: {}".format(e, score))
                    break
            scores.append(score)
            self.env.close()
        return scores

    def get_plot_dir(self, subdir):
        plot_dir = PLOTS_DIR + "/" + subdir
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

    def test(self):
        return np.array(self.evaluate(50)[1])