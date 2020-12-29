import torch
from algorithms.A3C.A3C import A3C
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
    # A3C worker (thread)
    # globalA3C is a parent object holding global actor and critic models
    def __init__(self, globalA3C: A3C, log_info=False, eval_repeats=0):
        # Environment and PPO parameters
        self.env = gym.make(ENV_NAME)
        self.globalA3C = globalA3C
        self.action_space = self.env.action_space
        self.MAX_EPISODES = globalA3C.MAX_EPISODES
        self.lock = globalA3C.lock
        self.discount_rate = globalA3C.discount_rate
        self.eval_repeats = eval_repeats
        self.step = 1
        # just for logging and plotting rewards. Incremented after achieving terminal state,
        self.local_episode = 0
        self.step_max = globalA3C.step_max
        self.log_info = log_info
        # Instantiate plot memory
        # not used too much for now
        self.scores, self.eval_av_scores, self.eval_episodes, self.episodes, self.accum_rewards = [], [], [], [], 0

        # Create Actor-Critic network models
        self.Actor = Actor(state_space=self.env.observation_space, action_space=self.action_space)
        self.Critic = Critic(state_space=self.env.observation_space)
        self.Actor.model.train()
        self.Critic.model.train()

    def update_results(self):
        self.scores.append(self.accum_rewards)
        self.episodes.append(self.local_episode)
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
        R = 0 if last_terminal else self.Critic.predict(t(last_state))
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

        if self.log_info != 0:
            if last_terminal:
                a3c_logger.info(f"Local episode: {self.local_episode}, global episode: {self.globalA3C.episode}, "
                                f"accumulated training score: {self.accum_rewards}")

    def sync_models(self):
        self.lock.acquire()
        # take weights from global models and assign them to workers models
        self.Actor.set_model_from_global(self.globalA3C.Actor.model)
        self.Critic.set_model_from_global(self.globalA3C.Critic.model)
        self.lock.release()

    def evaluate(self):
        self.Actor.model.eval()
        self.Critic.model.eval()
        performance = 0
        for _ in range(self.eval_repeats):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.Actor.get_best_action(t(state))
                state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    performance += score
                    break
        self.Actor.model.train()
        self.Critic.model.train()
        average_score = performance / self.eval_repeats
        self.eval_av_scores.append(average_score)
        self.eval_episodes.append(self.local_episode)
        print(f"Local episode, {self.local_episode}, global episode: {self.globalA3C.episode}, average eval score: {average_score}")

    def run(self):
        state = self.env.reset()  # reset env and get initial state
        while self.globalA3C.episode < self.MAX_EPISODES:
            # reset stuff
            is_terminal, saving = False, ''
            states, actions, rewards = [], [], []
            self.sync_models()
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
            self.globalA3C.episode += 1
            self.lock.release()

            if is_terminal:
                self.update_results()
                state = self.env.reset()  # reset env and get initial state
                self.local_episode += 1
                if self.local_episode % 100 == 0 and self.eval_repeats != 0:
                    self.evaluate()

        self.env.close()
