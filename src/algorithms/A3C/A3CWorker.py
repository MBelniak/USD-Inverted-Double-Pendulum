import torch
from algorithms.A3C.A3C import A3C
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
    # A3C worker (thread)
    def __init__(self, globalA3C: A3C):
        # Environment and PPO parameters
        self.env = gym.make(ENV_NAME)
        self.globalA3C = globalA3C
        self.action_space = self.env.action_space
        self.MAX_EPISODES = globalA3C.MAX_EPISODES
        self.lock = globalA3C.lock
        self.discount_rate = globalA3C.discount_rate
        self.t_max = globalA3C.t_max
        # Instantiate plot memory
        self.scores, self.episodes, self.accum_rewards = [], [], 0

        # Create Actor-Critic network model
        self.Actor = Actor(global_model_params=globalA3C.Actor.model.parameters(),
                           state_space=self.env.observation_space, learning_rate=self.globalA3C.actor_learning_rate,
                           action_space=self.action_space)
        self.Critic = Critic(global_model_params=globalA3C.Actor.model.parameters(),
                             state_space=self.env.observation_space, learning_rate=self.globalA3C.critic_learning_rate)
        self.Actor.model.train()
        self.Critic.model.train()

    def update_global_models(self, states, actions, rewards, last_state, last_terminal: bool):
        R = 0 if last_terminal else self.Critic.predict(last_state)
        self.Actor.optimizer.zero_grad()
        self.Critic.optimizer.zero_grad()
        critic_loss = 0
        actor_loss = 0
        for i in reversed(range(len(rewards))):
            self.accum_rewards += rewards[i]
            R = rewards[i] + self.discount_rate * R
            advantage = (R - self.Critic.predict(t(states[i])))
            alpha, beta = self.Actor.predict(t(states[i]))

            torch.distributions.Beta.set_default_validate_args(True)
            dist = torch.distributions.Beta(alpha + 1, beta + 1)

            critic_loss = critic_loss + advantage.pow(2).mean()
            actor_loss = actor_loss - dist.log_prob(self.Actor.action_to_beta(actions[i])) * advantage.detach()

        actor_loss.backward()
        critic_loss.backward()
        self.globalA3C.Critic.optimizer.step()
        self.globalA3C.Actor.optimizer.step()
        self.scores.append(R)
        if self.globalA3C.episode > 0 and self.globalA3C.episode % 100 == 0:
            print(f"Episode: {self.globalA3C.episode}, accumulated rewards over 100 episodes: {self.accum_rewards}")
            self.accum_rewards = 0

    def sync_models(self):
        self.Actor.set_model_from_global(self.globalA3C.Actor.model)
        self.Critic.set_model_from_global(self.globalA3C.Critic.model)

    def run(self):
        iteration = 1
        ensure_shared_grads(self.Actor.model, self.globalA3C.Actor.model)
        ensure_shared_grads(self.Critic.model, self.globalA3C.Critic.model)
        while self.globalA3C.episode < self.MAX_EPISODES:
            is_terminal, saving = False, ''  # Reset gradients etc
            states, actions, rewards = [], [], []  # reset thread memory
            self.sync_models()
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
            self.update_global_models(states, actions, rewards, last_state=state, last_terminal=is_terminal)
            self.globalA3C.episode += 1
            self.lock.release()

        self.env.close()
