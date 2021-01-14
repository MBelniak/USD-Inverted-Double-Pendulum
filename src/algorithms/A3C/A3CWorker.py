from copy import deepcopy

import numpy as np
import torch
from logger.logger import a3c_logger
from utils import ENV_NAME, t
from algorithms.A3C.Actor import Actor
from algorithms.A3C.Critic import Critic
import gym


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class A3CWorker:
    globalA3C = None

    # A3C worker (thread)
    def __init__(self, global_lock, max_episodes=100000, discount_rate=0.99, step_max=5, actor_lr=0.001, critic_lr=0.001,
                 measure_step=100, log_info=False, eval_repeats=0):
        # Environment and PPO parameters
        self.env = gym.make(ENV_NAME)
        self.action_space = self.env.action_space
        self.max_episodes = max_episodes
        self.lock = global_lock
        self.discount_rate = discount_rate
        self.eval_repeats = eval_repeats
        self.step = 1
        # just for logging and plotting rewards. Incremented after achieving terminal state,
        self.local_episode = 0
        self.step_max = step_max
        self.log_info = log_info
        self.measure_step = measure_step
        # Instantiate plot memory
        self.performance, self.eval_episodes, self.accum_rewards = [], [], 0

        # Create Actor-Critic network models
        self.Actor = Actor(state_space=self.env.observation_space, learning_rate=actor_lr, action_space=self.action_space)
        self.Critic = Critic(state_space=self.env.observation_space, learning_rate=critic_lr)
        self.Actor.model.train()
        self.Critic.model.train()

    def set_global_model(self, global_model):
        self.globalA3C = global_model
        self.lock.acquire()
        self.sync_models()
        self.lock.release()

    def update_local_results(self):
        self.performance.append([self.local_episode + 1, self.accum_rewards])
        self.accum_rewards = 0

    def update_global_models(self):
        # assign gradients of workers' models to global models
        ensure_shared_grads(self.Actor.model, self.globalA3C.Actor.model)
        ensure_shared_grads(self.Critic.model, self.globalA3C.Critic.model)
        # update weights using these gradients
        self.globalA3C.Critic.optimizer.step()
        self.globalA3C.Actor.optimizer.step()

    def replay_steps(self, states, actions, rewards, last_state, last_terminal: bool):
        # get predicted reward for the last state - we didn't do action in that state
        R = 0 if last_terminal and rewards[-1] < 9 else self.Critic.predict(t(last_state))
        # reset gradients for optimizers
        self.Actor.optimizer.zero_grad()
        self.Critic.optimizer.zero_grad()
        critic_loss, actor_loss = 0, 0
        # go backwards through states, actions and rewards taken in this episode
        for i in reversed(range(len(rewards))):
            self.accum_rewards += rewards[i]
            R = rewards[i] + self.discount_rate * R
            advantage = (R - self.Critic.predict(t(states[i])))
            # get Beta distribution parameters with which the action was drawn
            alpha, beta = self.Actor.predict(t(states[i]))

            torch.distributions.Beta.set_default_validate_args(True)
            dist = torch.distributions.Beta(alpha, beta)

            # accumulate critic loss
            critic_loss = critic_loss + advantage.pow(2).mean()
            # accumulate actor loss - we maximize the rewards, thus we take negation of gradient.
            # Adam opt. then negates it again, so weights are updated in a way which makes advantages higher
            actor_loss = actor_loss - dist.log_prob(self.Actor.action_to_beta(t(actions[i]))) * advantage.detach()

        # compute gradients wrt. weights
        actor_loss.backward()
        critic_loss.backward()

    def sync_models(self):
        # take weights from global models and assign them to workers models
        self.Actor.set_model(self.globalA3C.Actor.model)
        self.Critic.set_model(self.globalA3C.Critic.model)

    def run(self):
        if self.globalA3C is None:
            raise Exception("Global model is not set! Please call set_global_model(global_model) to set the parent model.")

        state = self.env.reset()  # reset env and get initial state
        episode = 0
        while episode < self.max_episodes:
            # reset stuff
            is_terminal = False
            states, actions, rewards = [], [], []
            step_start = self.step

            while not is_terminal and self.step - step_start < self.step_max:
                states.append(state)  # register current state
                action = self.Actor.draw_action(t(state))  # draw action
                next_state, reward, is_terminal, info = self.env.step(action)  # perform action
                actions.append(action)  # register action
                rewards.append(reward)  # register reward
                state = next_state
                self.step += 1

            # replay experience backwards and compute gradients
            self.replay_steps(states, actions, rewards, last_state=state, last_terminal=is_terminal)
            self.lock.acquire()
            self.update_global_models()
            self.sync_models()
            self.globalA3C.episode += 1
            episode = self.globalA3C.episode
            self.lock.release()

            if episode % self.measure_step == 0 and self.eval_repeats != 0:
                self.lock.acquire()
                mean, _ = self.evaluate(self.eval_repeats)
                self.globalA3C.performance.append([episode, mean])
                self.lock.release()
                if self.log_info:
                    a3c_logger.info(f"\nEpisode: {episode}\nMean accumulated rewards: {mean}")

            if is_terminal:
                self.update_local_results()
                state = self.env.reset()  # reset env and get initial state
                self.local_episode += 1

        self.env.close()

    def evaluate(self, eval_repeats=20):
        self.Actor.model.eval()
        self.Critic.model.eval()
        scores = []
        env = gym.make(ENV_NAME)
        for ep in range(eval_repeats):
            state = env.reset()
            done = False
            performance = 0
            while not done:
                with torch.no_grad():
                    action = self.Actor.get_best_action(t(state))
                state, reward, done, _ = env.step(action)
                performance += reward

            scores.append([ep + 1, performance])

        scores = np.array(scores)
        self.Actor.model.train()
        self.Critic.model.train()
        return scores[:, 1].mean(), scores
