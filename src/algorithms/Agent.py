
class Agent(object):
    performance, learning_rate, env, max_episodes = [], {}, {}, {}

    def __init__(self):
        if type(self) is Agent:
            raise Exception('Model is an abstract class and cannot be instantiated directly')

    def test(self):
        raise NotImplementedError('subclasses must override test()!')

    def run(self):
        raise NotImplementedError('subclasses must override run()!')

    def plot_training(self, performance):
        raise NotImplementedError('subclasses must override plot_training(performance)!')

    def plot_test(self, performance):
        raise NotImplementedError('subclasses must override plot_test(performance)!')

    def save_models(self, file_name):
        raise NotImplementedError('subclasses must override save_models(file_name)!')
