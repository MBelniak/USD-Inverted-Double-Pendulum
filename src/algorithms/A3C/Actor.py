import torch
from torch import nn
from logger.logger import a3c_logger


class Actor:
    train_fn = {}

    def __init__(self, **kwargs):
        self.action_space = kwargs['action_space']
        self.state_space = kwargs['state_space']
        self.learning_rate = kwargs['learning_rate']
        self.model_output_dim = 2  # alpha and beta for beta distribution
        self.model = self.create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_space.shape[0], 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, self.model_output_dim),
            nn.Softplus()
        )
        a3c_logger.info(model)

        return model

    def predict(self, state):
        return self.model(state)

    def get_action(self, state):
        # Use the actor's network to predict the next action to take, using its model
        # beta is better than Gaussian for a bounded action space: http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        alpha, beta = self.predict(state)
        dist = torch.distributions.Beta(alpha + 1, beta + 1)  # Add 1 to alpha and beta to ensure alpha, beta >=1
        action = self.beta_to_action(dist.sample())
        return action

    # two functions below: scale and move beta distribution as action space is non-symmetrical.
    def action_to_beta(self, action):
        low_boundary, high_boundary = self.action_space.low, self.action_space.high
        return (action - low_boundary) / (high_boundary - low_boundary)

    def beta_to_action(self, beta_sample):
        low_boundary, high_boundary = self.action_space.low, self.action_space.high
        return beta_sample * (high_boundary - low_boundary) + low_boundary

    def set_model_from_global(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
