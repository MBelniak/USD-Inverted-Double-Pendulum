import gym
import torch
import numpy as np
from torch import nn
import random
import collections
# from torch.optim.lr_scheduler import StepLR

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""


class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.model = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_dim, action_dim))

    def forward(self, inp):
        return self.model(inp)


# """
# If the observations are images we use CNNs.
# """
# class QNetworkCNN(nn.Module):
#     def __init__(self, action_dim):
#         super(QNetworkCNN, self).__init__()
#
#         self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
#         self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
#         self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.fc_1 = nn.Linear(8960, 512)
#         self.fc_2 = nn.Linear(512, action_dim)
#
#     def forward(self, inp):
#         inp = inp.view((1, 3, 210, 160))
#         x1 = F.relu(self.conv_1(inp))
#         x1 = F.relu(self.conv_2(x1))
#         x1 = F.relu(self.conv_3(x1))
#         x1 = torch.flatten(x1, 1)
#         x1 = F.leaky_relu(self.fc_1(x1))
#         x1 = self.fc_2(x1)
#
#         return x1


"""
memory to save the state, action, reward sequence from the current episode. 
"""


class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.states = collections.deque(maxlen=len)
        self.actions = collections.deque(maxlen=len)
        self.is_done_arr = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_done_arr.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.actions)
        idx = random.sample(range(0, n - 1), batch_size)

        return torch.Tensor(self.states)[idx].to(device), torch.LongTensor(self.actions)[idx].to(device), \
               torch.Tensor(self.states)[1 + np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done_arr)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.is_done_arr.clear()


def select_action(model, state, eps, num_actions):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(values.cpu().numpy())

    return action


def train(batch_size, current_model, target_model, optim, memory, gamma):
    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current_model(states)

    next_q_values = current_model(next_states)
    next_q_state_values = target_model(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


def evaluate(Qmodel, env, repeats):
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
            state, reward, done, _ = env.step(discretized_actions[action])
            perform += reward
    Qmodel.train()
    return perform / repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def main(gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.995, eps_min=0.01, update_step=10, batch_size=64,
         update_repeats=50,
         num_episodes=3000, seed=42, max_memory_size=50000, lr_gamma=0.9, lr_step=100, measure_step=100,
         measure_repeats=10, hidden_dim=64, env_name='InvertedDoublePendulum-v2', horizon=np.inf, render=True,
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
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """
    env = gym.make(env_name)
    torch.manual_seed(seed)
    env.seed(seed)

    global discretized_actions
    discretized_actions = [(((env.action_space.high[0] - env.action_space.low[0]) * i / (num_actions - 1)) - 1) for i in
                           range(num_actions)]

    current_Q = QNetwork(action_dim=num_actions, state_dim=env.observation_space.shape[0],
                   hidden_dim=hidden_dim).to(device)
    target_Q = QNetwork(action_dim=num_actions, state_dim=env.observation_space.shape[0],
                   hidden_dim=hidden_dim).to(device)
    # transfer parameters from current_Q to target_Q
    update_parameters(current_Q, target_Q)

    # we only train current_Q
    for param in target_Q.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(current_Q.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []

    for episode in range(num_episodes):
        # display the performance
        if episode % 100 == 0:
            performance.append([episode, evaluate(current_Q, env, measure_repeats)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            # print("lr: ", scheduler.get_lr()[0])
            print("eps: ", eps)

        state = env.reset()
        memory.states.append(state)

        done = False
        i = 0
        while not done:
            i += 1
            action = select_action(target_Q, state, eps, num_actions)
            state, reward, done, _ = env.step(discretized_actions[action])

            if i > horizon:
                done = True

            # render the environment if render == True
            if render and episode % render_step == 0:
                env.render()

            # save state, action, reward sequence
            memory.update(state, action, reward, done)

        if episode >= min_episodes and episode % update_step == 0:
            for _ in range(update_repeats):
                train(batch_size, current_Q, target_Q, optimizer, memory, gamma)

            # transfer new parameter from current_Q to target_Q
            update_parameters(current_Q, target_Q)

        # update learning rate and eps
        # scheduler.step()
        eps = max(eps * eps_decay, eps_min)

    return current_Q, performance


if __name__ == '__main__':
    main()
