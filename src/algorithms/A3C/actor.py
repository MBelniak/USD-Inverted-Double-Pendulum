from algorithms.model import get_dense_network


class Actor:
    def __init__(self, **kwargs):
        self.model = get_dense_network(kwargs['input_shape'], kwargs['action_space'], kwargs['learning_rate'])
        # make predict function to work while multithreading
        self.model._make_predict_function()

    def predict(self, state):
        return self.model.predict(state)[0]

