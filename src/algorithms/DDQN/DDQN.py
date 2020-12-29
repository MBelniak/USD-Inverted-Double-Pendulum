import gym
import torch
import numpy as np
from torch import device
import random
from torch.optim.lr_scheduler import StepLR

from algorithms.DDQN.Memory import Memory
from algorithms.DDQN.QNetwork import QNetwork

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDQN():
    def select_action(self, model, env, state, eps, num_actions):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = model(state)

        # select a random action wih probability eps
        if random.random() <= eps:
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(values.cpu().numpy())

        return action

    def train(self, batch_size, current, target, optim, memory, gamma):

        states, actions, next_states, rewards, is_done = memory.sample(batch_size)

        q_values = current(states)

        next_q_values = current(next_states)
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

    def run(self, gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.999, eps_min=0.01, update_step=10,
            batch_size=64, update_repeats=50,
            num_episodes=100000, seed=42, max_memory_size=50000, lr_gamma=0.9, lr_step=100, measure_step=500,
            measure_repeats=100, hidden_dim=64, env_name='InvertedDoublePendulum-v2', horizon=np.inf, render=False,
            render_step=50, num_actions=100):
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
        env = gym.make(env_name)
        torch.manual_seed(seed)
        env.seed(seed)

        self.discretized_actions = [(((env.action_space.high[0] - env.action_space.low[0]) * i / (num_actions - 1)) - 1)
                                    for i in range(num_actions)]

        Q_1 = QNetwork(action_dim=num_actions, state_dim=env.observation_space.shape[0],
                       hidden_dim=hidden_dim).to(device)
        Q_2 = QNetwork(action_dim=num_actions, state_dim=env.observation_space.shape[0],
                       hidden_dim=hidden_dim).to(device)

        # transfer parameters from Q_1 to Q_2
        self.update_parameters(Q_1, Q_2)

        # we only train Q_1
        for param in Q_2.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)

        memory = Memory(max_memory_size)
        performance = []

        for episode in range(num_episodes):
            # display the performance
            if episode % measure_step == 0:
                performance.append([episode, self.evaluate(Q_1, env, measure_repeats)])
                print("Episode: ", episode)
                print("rewards: ", performance[-1][1])
                print("eps: ", eps)

            state = env.reset()
            memory.state.append(state)

            done = False
            i = 0
            while not done:
                i += 1
                action = self.select_action(Q_2, env, state, eps, num_actions)
                state, reward, done, _ = env.step(self.discretized_actions[action])

                if i > horizon:
                    done = True

                # render the environment if render == True
                if render and episode % render_step == 0:
                    env.render()

                # save state, action, reward sequence
                memory.update(state, action, reward, done)

            if episode >= min_episodes and episode % update_step == 0:
                for _ in range(update_repeats):
                    self.train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

                # transfer new parameter from Q_1 to Q_2
                self.update_parameters(Q_1, Q_2)

            eps = max(eps * eps_decay, eps_min)

        return Q_1, performance


if __name__ == '__main__':
    ddqn = DDQN()
    ddqn.run()
