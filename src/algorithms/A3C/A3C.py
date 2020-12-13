from threading import Lock
import pylab
import torch

from utils import ENV_NAME, t
from algorithms.A3C.Actor import Actor
from algorithms.A3C.Critic import Critic
import numpy as np
import gym
import os

SAVE_DIR = "saved_models"


def get_default_save_filename(episodes, discount, tmax, actor_lr, critic_lr):
    return f"A3C_{episodes}--{str(discount).replace('.', '_')}-{str(tmax).replace('.', '_')}-" \
           f"{str(actor_lr).replace('.', '_')}-{str(critic_lr).replace('.', '_')}"


def ensure_unique_path(path):
    if os.path.exists(path):
        counter = 1
        while os.path.exists(path + str(counter)):
            counter += 1
        return path + str(counter)
    return path


class A3C:
    # for PyCharm to resolve it correctly in A3CWorker
    MAX_EPISODES, lock, learning_rate, discount_rate = {}, {}, {}, {}
    t_max, Actor, Critic = {}, {}, {}

    # Actor-Critic Main Optimization Algorithm
    def __init__(self, max_episodes=100000, discount_rate=0.99, t_max=5, actor_lr=0.001, critic_lr=0.001):
        self.env = gym.make(ENV_NAME)
        self.action_space = self.env.action_space
        self.action_size = self.action_space.shape[0]
        self.MAX_EPISODES, self.episode = max_episodes, 0
        self.lock = Lock()
        self.actor_learning_rate = actor_lr
        self.critic_learning_rate = critic_lr
        self.discount_rate = discount_rate
        self.t_max = t_max
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

    def save_models(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = SAVE_DIR + "/" + get_default_save_filename(self.MAX_EPISODES, self.discount_rate, self.t_max,
                                                          self.actor_learning_rate, self.critic_learning_rate)
        path = ensure_unique_path(path)

        torch.save({
            'Actor_state_dict': self.Actor.model.state_dict(),
            'Critic_state_dict': self.Critic.model.state_dict(),
            'Actor_opt_state_dict': self.Actor.optimizer.state_dict(),
            'Critic_opt_state_dict': self.Critic.optimizer.state_dict()
        }, path)

    # pylab.figure(figsize=(18, 9))

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
