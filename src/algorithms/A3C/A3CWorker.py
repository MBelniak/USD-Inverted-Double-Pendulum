from main import ENV_NAME
from algorithms.A3C.actor import Actor
from algorithms.A3C.critic import Critic
from algorithms.A3C.A3C import A3C
import numpy as np
import gym
from keras import backend as k
from keras.models import load_model


class A3CWorker:
    # A3C worker (thread)
    def __init__(self, global_a3c: A3C):
        # Environment and PPO parameters
        self.env = gym.make(ENV_NAME)
        self.action_size = self.env.action_space.n
        self.globalA3C = global_a3c
        self.MAX_EPISODES, self.max_average = global_a3c.MAX_EPISODES, -21.0
        self.lock = global_a3c.lock
        self.learning_rate = global_a3c.learning_rate
        self.discount_rate = global_a3c.discount_rate
        self.t_max = global_a3c.t_max
        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []
        self.state_shape = self.env.observation_space.shape

        # Create Actor-Critic network model
        self.Actor = Actor(input_shape=self.state_shape, action_space=self.action_size,
                           learning_rate=self.learning_rate)
        self.Critic = Critic(input_shape=self.state_shape, action_space=self.action_size,
                             learning_rate=self.learning_rate)

    def accumulate_gradients(self, t_start, states, actions, rewards, last_state, last_terminal: bool):
        R = 0 if last_terminal else self.Critic.predict(last_state)
        actor_gradients, critic_gradients = 0, 0
        for i in range(len(states) - 1, t_start - 1, -1):
            R = rewards[i] + self.discount_rate * R
            error = R - self.Critic.predict(states[i])
            actor_gradients = actor_gradients + action_grad * error  #TODO
            critic_gradients = critic_gradients + value_grad * error  #TODO

        return actor_gradients, critic_gradients

    def update_global_model(self, actor_gradients, critic_gradients):
        #TODO update models
        pass

    def reset_model(self):
        self.Actor.model.set_weights(self.globalA3C.Actor.model.get_weights())  # copy weights from global model
        self.Critic.model.set_weights(self.globalA3C.Critic.model.get_weights())

    def run(self):
        global graph
        iteration = 1
        with graph.as_default():
            while self.globalA3C.episode < self.MAX_EPISODES:
                is_terminal, saving = False, ''  # Reset gradients etc
                states, actions, rewards = [], [], []  # reset thread memory
                self.reset_model()
                t_start = iteration
                state = self.env.reset()  # reset env and get initial state

                while not is_terminal and iteration - t_start < self.t_max:
                    states.append(state)  # register current state
                    action = self.globalA3C.get_action(state, self.Actor)  # draw action
                    next_state, reward, is_terminal, _ = self.env.step(action)  # perform action
                    actions.append(action)  # register action
                    rewards.append(reward)  # register reward
                    state = next_state


                actor_gradients, critic_gradients = \
                    self.accumulate_gradients(t_start, states, actions,
                                              rewards, last_state=state, last_terminal=is_terminal)
                self.lock.acquire()
                self.update_global_model(actor_gradients, critic_gradients)
                self.globalA3C.episode += 1
                self.lock.release()

            self.env.close()
