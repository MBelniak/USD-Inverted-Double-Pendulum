import torch
from torch import nn


class Critic:
    def __init__(self, **kwargs):
        self.state_space = kwargs['state_space']
        self.learning_rate = kwargs['learning_rate']
        self.model = self.create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        print(model)

        return model

    def predict(self, state):
        return self.model(state)

    def set_model_from_global(self, global_model):
        self.model.load_state_dict(global_model.state_dict())

