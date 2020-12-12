import torch
from algorithms.A3C.A3C import A3C
from utils import ENV_NAME, t
from algorithms.A3C.actor import Actor
from algorithms.A3C.critic import Critic
import gym


class A3CWorker:
    # A3C worker (thread)
    def __init__(self, globalA3C: A3C):
        # Environment and PPO parameters
        self.env = gym.make(ENV_NAME)
        self.globalA3C = globalA3C
        self.action_space = self.env.action_space
        self.MAX_EPISODES, self.max_average = globalA3C.MAX_EPISODES, -21.0
        self.lock = globalA3C.lock
        self.learning_rate = globalA3C.learning_rate
        self.discount_rate = globalA3C.discount_rate
        self.t_max = globalA3C.t_max
        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        # Create Actor-Critic network model
        self.Actor = Actor(state_space=self.env.observation_space, learning_rate=self.learning_rate,
                           action_space=self.action_space)
        self.Critic = Critic(state_space=self.env.observation_space, learning_rate=self.learning_rate)

    def update_global_models(self, t_start, states, actions, rewards, last_state, last_terminal: bool):
        R = 0 if last_terminal else self.Critic.predict(last_state)
        accumulated_reward = 0
        for i in range(len(states) - 1, t_start - 1, -1):
            accumulated_reward += rewards[i]
            R = rewards[i] + self.discount_rate * R
            advantage = (R - self.Critic.predict(t(states[i])))
            alpha, beta = self.Actor.predict(t(states[i]))

            dist = torch.distributions.Beta(alpha + 1, beta + 1)

            critic_loss = advantage.pow(2).mean()
            self.Critic.optimizer.zero_grad()
            critic_loss.backward()
            self.Critic.optimizer.step()

            actor_loss = -dist.log_prob(actions[i]) * advantage.detach()
            self.Actor.optimizer.zero_grad()
            actor_loss.backward()
            self.Actor.optimizer.step()

        self.scores.append(R)
        if self.globalA3C.episode % 200 == 0:
            print(f"Episode: {self.globalA3C.episode}, accumulated reward for 5 steps: {accumulated_reward}")

    def reset_model(self):
        self.Actor.set_model_from_global(self.globalA3C.Actor.model)
        self.Critic.set_model_from_global(self.globalA3C.Critic.model)

    def run(self):
        # global graph
        iteration = 1
        # with graph.as_default():
        while self.globalA3C.episode < self.MAX_EPISODES:
            is_terminal, saving = False, ''  # Reset gradients etc
            states, actions, rewards = [], [], []  # reset thread memory
            self.reset_model()
            t_start = iteration
            state = self.env.reset()  # reset env and get initial state

            while not is_terminal and iteration - t_start < self.t_max:
                states.append(state)  # register current state
                action = self.Actor.get_action(t(state))  # draw action
                next_state, reward, is_terminal, info = self.env.step(action.detach().data.numpy())  # perform action
                actions.append(action)  # register action
                rewards.append(reward)  # register reward
                state = next_state

            self.lock.acquire()
            self.update_global_models(t_start, states, actions, rewards, last_state=state, last_terminal=is_terminal)
            self.globalA3C.episode += 1
            self.lock.release()

        self.env.close()
