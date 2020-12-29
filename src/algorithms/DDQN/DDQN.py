import os

import gym
import torch
import numpy as np
from torch import device
import random
from torch.optim.lr_scheduler import StepLR

from algorithms.DDQN.Memory import Memory
from algorithms.DDQN.QNetwork import QNetwork
from utils import ensure_unique_path, get_default_save_filename

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "saved_models"
PLOTS_DIR = "plots"


class DDQN():
    """
    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """

    def __init__(self, gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.999, eps_min=0.01, update_step=10,
                 batch_size=64, update_repeats=50,
                 num_episodes=10000, seed=42, max_memory_size=50000, lr_gamma=0.9, lr_step=100, measure_step=500,
                 measure_repeats=100, hidden_dim=64, env_name='InvertedDoublePendulum-v2', horizon=np.inf, render=False,
                 render_step=50, num_actions=100):

        self.gamma = gamma
        self.lr = lr
        self.min_episodes = min_episodes
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.update_step = update_step
        self.batch_size = batch_size
        self.update_repeats = update_repeats
        self.num_episodes = num_episodes
        self.seed = seed
        self.max_memory_size = max_memory_size
        self.lr_gamma = lr_gamma
        self.lr_step = lr_step
        self.measure_step = measure_step
        self.measure_repeats = measure_repeats
        self.hidden_dim = hidden_dim
        self.env_name = env_name
        self.horizon = horizon
        self.do_render = render
        self.render_step = render_step
        self.num_actions = num_actions

        self.env = gym.make(env_name)
        torch.manual_seed(seed)
        self.env.seed(seed)

        self.discretized_actions = [
            (((self.env.action_space.high[0] - self.env.action_space.low[0]) * i / (num_actions - 1)) - 1)
            for i in range(num_actions)]

        self.primary_q = QNetwork(action_dim=num_actions, state_dim=self.env.observation_space.shape[0],
                                  hidden_dim=hidden_dim).to(device)
        self.target_q = QNetwork(action_dim=num_actions, state_dim=self.env.observation_space.shape[0],
                                 hidden_dim=hidden_dim).to(device)

    def load_models(self, file_name=None):
        if file_name == None:
            file_name = f"DDQN-{self.num_episodes}"

        checkpoint = torch.load(f"{SAVE_DIR}/{file_name}")
        self.primary_q.model.load_state_dict(checkpoint['primary_q'])
        self.target_q.model.load_state_dict(checkpoint['target_q'])
        self.primary_q.model.eval()
        self.target_q.model.eval()

    def save_models(self, file_name=None):
        if file_name == None:
            file_name = f"DDQN-{self.num_episodes}"

        os.makedirs(SAVE_DIR, exist_ok=True)
        path = SAVE_DIR + "/" + file_name
        path = ensure_unique_path(path)

        torch.save({
            'primary_q': self.primary_q.model.state_dict(),
            'target_q': self.target_q.model.state_dict(),
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

    def train(self, batch_size, primary, target, optim, memory, gamma):

        states, actions, next_states, rewards, is_done = memory.sample(batch_size)

        q_values = primary(states)

        next_q_values = primary(next_states)
        next_q_state_values = target(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

    def evaluate(self, Qmodel, env, repeats):
        """
        Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
        episode reward.
        """
        Qmodel.eval()
        perform = 0
        for _ in range(repeats):
            state = env.reset()
            done = False
            while not done:
                state = torch.Tensor(state).to(device)
                with torch.no_grad():
                    values = Qmodel(state)
                action = np.argmax(values.cpu().numpy())
                state, reward, done, _ = env.step(self.discretized_actions[action])
                perform += reward
        Qmodel.train()
        return perform / repeats

    def update_parameters(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def run(self):

        # transfer parameters from Q_1 to Q_2
        self.update_parameters(self.primary_q, self.target_q)

        # we only train Q_1
        for param in self.target_q.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(self.primary_q.parameters(), lr=self.lr)

        memory = Memory(self.max_memory_size)
        performance = []

        for episode in range(self.num_episodes):
            # display the performance
            if episode % self.measure_step == 0:
                performance.append([episode, self.evaluate(self.primary_q, self.env, self.measure_repeats)])
                print("Episode: ", episode)
                print("rewards: ", performance[-1][1])
                print("eps: ", self.eps)

            state = self.env.reset()
            memory.state.append(state)

            done = False
            i = 0
            while not done:
                i += 1
                action = self.select_action(self.target_q,  state, self.eps)
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
                    self.train(self.batch_size, self.primary_q, self.target_q, optimizer, memory, self.gamma)

                # transfer new parameter from Q_1 to Q_2
                self.update_parameters(self.primary_q, self.target_q)

            self.eps = max(self.eps * self.eps_decay, self.eps_min)

        return self.primary_q, performance

    def render(self):
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
                    print("episode: {}, score: {}".format(e, score))
                    break
        self.env.close()


if __name__ == '__main__':
    ddqn = DDQN()
    ddqn.run()
