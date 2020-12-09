import threading
from threading import Lock
import time
import pylab

from algorithms.A3C.A3CWorker import A3CWorker
from main import ENV_NAME
from algorithms.A3C.actor import Actor
from algorithms.A3C.critic import Critic
import numpy as np
import gym
from keras.models import load_model
import os


class A3C:
    MAX_EPISODES, lock, learning_rate, discount_rate = {}, {}, {}, {}  # for PyCharm to resolve it correctly in A3CWorker
    t_max, Actor, Critic = {}, {}, {}

    # Actor-Critic Main Optimization Algorithm
    def __init__(self, max_episodes = 100000, discount_rate = 0.99, t_max = 5):
        self.env = gym.make(ENV_NAME)
        self.action_size = self.env.action_space.n
        self.MAX_EPISODES, self.episode, self.max_average = max_episodes, 0, -21.0
        self.lock = Lock()
        self.learning_rate = 0.0001
        self.discount_rate = discount_rate
        self.t_max = t_max
        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_shape = self.env.observation_space.shape

        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = 'A3C_{}'.format(self.learning_rate)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.Actor = Actor(input_shape=self.state_shape, action_space=self.action_size,
                           learning_rate=self.learning_rate)
        self.Critic = Critic(input_shape=self.state_shape, action_space=self.action_size,
                             learning_rate=self.learning_rate)

    @staticmethod
    def get_action(state, actor):
        # Use the actor's network to predict the next action to take, using its model
        # TODO find out how to map normal distribution to range [-1; 1]
        prediction = actor.predict(state)[0]
        action = np.clip(np.random.normal(loc=np.clip(prediction, -1, 1), scale=1), -1, 1)
        return action

    def load(self, actor_name, critic_name):
        self.Actor.model = load_model(actor_name, compile=False)
        self.Critic.model = load_model(critic_name, compile=False)

    def save(self):
        self.Actor.model.save(self.Model_name + '_Actor.h5')
        self.Critic.model.save(self.Model_name + '_Critic.h5')

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

    def train(self, n_threads):
        self.env.close()
        # Instantiate one worker per thread
        workers = [A3CWorker(global_a3c=self) for i in range(n_threads)]

        # Create threads
        threads = [threading.Thread(
            target=workers[i].run,
            daemon=True) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()

    def test(self, actor_name, critic_name):
        self.load(actor_name, critic_name)
        for e in range(100):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                action = np.argmax(self.get_action(state, self.Actor))
                state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.MAX_EPISODES, score))
                    break
        self.env.close()