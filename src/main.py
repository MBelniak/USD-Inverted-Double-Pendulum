import argparse
import tensorflow as tf
from algorithms.A3C.A3C import A3C

ENV_NAME = 'InvertedDoublePendulum-v2'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', help='Algorithm to use.', default='A3C')
    parser.add_argument('--threads', help='Number of threads for A3C.', type=int, default=5)
    parser.add_argument('--episodes', help='Number of episodes.', type=int, default=10000)
    parser.add_argument('--discount', help='Discount rate.', type=float, default=0.99)
    parser.add_argument('--tmax', help='Max stapes before update.', type=int, default=5)
    args = parser.parse_args()

    if args.algorithm is 'A3C':
        agent = A3C(max_episodes=args.episodes, discount_rate=args.discount_rate, t_max=5)
        agent.train(n_threads=5)


# configure Keras and TensorFlow sessions and graph
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


if __name__ == "__main__":
    main()
