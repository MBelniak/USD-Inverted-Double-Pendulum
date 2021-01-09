
class Model(object):
    performance, learning_rate, env, max_episodes = [], {}, {}, {}

    def __init__(self):
        if type(self) is Model:
            raise Exception('Model is an abstract class and cannot be instantiated directly')

    def test(self):
        raise NotImplementedError('subclasses must override test()!')

    def run(self):
        raise NotImplementedError('subclasses must override run()!')

    def plot(self, subdir, performance):
        raise NotImplementedError('subclasses must override plot(subdir, performance)!')