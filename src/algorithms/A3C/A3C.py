from threading import Lock
import pylab
from utils import ENV_NAME
from algorithms.A3C.actor import Actor
from algorithms.A3C.critic import Critic
import numpy as np
import gym
import os


class A3C:
    MAX_EPISODES, lock, learning_rate, discount_rate = {}, {}, {}, {}  # for PyCharm to resolve it correctly in A3CWorker
    t_max, Actor, Critic = {}, {}, {}

    # Actor-Critic Main Optimization Algorithm
    def __init__(self, max_episodes=100000, discount_rate=0.99, t_max=5):
        self.env = gym.make(ENV_NAME)
        self.action_space = self.env.action_space
        self.action_size = self.action_space.shape[0]
        self.MAX_EPISODES, self.episode, self.max_average = max_episodes, 0, -21.0
        self.lock = Lock()
        self.learning_rate = 0.0001
        self.discount_rate = discount_rate
        self.t_max = t_max
        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'

        if not os.path.exists(self.Save_Path):
            os.makedirs(self.Save_Path)
        self.path = 'A3C_{}'.format(self.learning_rate)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.Actor = Actor(state_space=self.env.observation_space, learning_rate=self.learning_rate,
                           action_space=self.action_space)
        self.Critic = Critic(state_space=self.env.observation_space, learning_rate=self.learning_rate)

    # def load(self, actor_name, critic_name):
    #     self.Actor.model = load_model(actor_name, compile=False)
    #     self.Critic.model = load_model(critic_name, compile=False)
    #
    # def save(self):
    #     self.Actor.model.save(self.Model_name + '_Actor.h5')
    #     self.Critic.model.save(self.Model_name + '_Critic.h5')

    pylab.figure(figsize=(18, 9))

    # TODO
    # def PlotModel(self, score, episode):
    #     self.scores.append(score)
    #     self.episodes.append(episode)
    #     self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
    #     if str(episode)[-2:] == "00":  # much faster than episode % 100
    #         pylab.plot(self.episodes, self.scores, 'b')
    #         pylab.plot(self.episodes, self.average, 'r')
    #         pylab.ylabel('Score', fontsize=18)
    #         pylab.xlabel('Steps', fontsize=18)
    #         try:
    #             pylab.savefig(self.path + ".png")
    #         except OSError:
    #             pass
    #
    #     return self.average[-1]

    # def test(self, actor_name, critic_name):
    #     self.load(actor_name, critic_name)
    #     for e in range(100):
    #         state = self.env.reset()
    #         done = False
    #         score = 0
    #         while not done:
    #             action = np.argmax(self.Actor.get_action(state))
    #             state, reward, done, _ = self.env.step(action)
    #             score += reward
    #             if done:
    #                 print("episode: {}/{}, score: {}".format(e, self.MAX_EPISODES, score))
    #                 break
    #     self.env.close()
