import argparse
from algorithms.A3C.A3C import A3C, get_default_save_filename
from algorithms.DDQN.DDQN import DDQN
import numpy as np

def renderA3C(**kwargs):
    if kwargs['load_file'] is None:
        parameters_file = get_default_save_filename(kwargs['episodes'], kwargs['threads'], kwargs['discount'],
                                                    kwargs['step_max'], kwargs['actor_lr'], kwargs['critic_lr'])
    else:
        parameters_file = kwargs['load_file']
    agent = A3C()
    agent.load_models(parameters_file)
    agent.render()


def renderDDQN(**kwargs):
    agent = DDQN(kwargs)

    if kwargs['load_file'] is None:
        parameters_file = f"DDQN-{agent.max_episodes}"
    else:
        parameters_file = kwargs['load_file']

    agent.load_models(parameters_file)
    agent.render()


def main():
    parser = argparse.ArgumentParser()
#general params
    parser.add_argument('-render', help='Render environment.', action='store_true')
    parser.add_argument('--algorithm', help='Algorithm to use.', default='DDQN', choices=['A3C', 'DDQN'])
    parser.add_argument('--load_file', help='Custom filename from which to load weights before rendering.', default=None)
    parser.add_argument('--episodes', help='Number of episodes for DDQN(run from beginning to terminal state).', type=int, default=10000)
#A3C pararms
    parser.add_argument('--threads', help='Number of threads for A3C.', type=int, default=5)
    parser.add_argument('--discount', help='Discount rate.', type=float, default=0.99)
    parser.add_argument('--step_max', help='Max steps before update.', type=int, default=5)
    parser.add_argument('--actor_lr', help='Actor learning rate.', type=float, default=0.001)
    parser.add_argument('--critic_lr', help='Critic learning rate.', type=float, default=0.001)
    parser.add_argument('--eval_repeats', help='Number of evaluation runs in one performance evaluation.'
                                               ' Set to 0 to disable.', type=int, default=10)
    parser.add_argument('-no_log', help='Disable logging during training.', action='store_true')
#DDQN paramrs
    parser.add_argument('--gamma', help='reward discount factor.', type=float, default=0.99)
    parser.add_argument('--lr', help='learning rate.', type=float, default=0.001)
    parser.add_argument('--min_episodes', help='We wait "min_episodes" many episodes in order to aggregate enough data before starting to train', type=int, default=20)
    parser.add_argument('--eps', help='probability to take a random action during training', type=float, default=1)
    parser.add_argument('--eps_decay', help='after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time', type=float, default=0.99)
    parser.add_argument('--eps_min', help='minimal value of "eps"', type=float, default=0.01)
    parser.add_argument('--update_step', help='after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a batch of size "batch_size" from the memory.', type=int, default=10)
    parser.add_argument('--batch_size', help='see above', type=int, default=64)
    parser.add_argument('--update_repeats', help='see above', type=int, default=50)
    parser.add_argument('--seed', help='random seed for reproducibility', type=int, default=42)
    parser.add_argument('--max_memory_size', help='size of the replay memory', type=int, default=50000)
    parser.add_argument('--measure_step', help='every "measure_step" episode the performance is measured', type=int, default=100)
    parser.add_argument('--measure_repeats', help='the amount of episodes played in to asses performance', type=int, default=20)
    parser.add_argument('--hidden_dim', help='hidden dimensions for the Q_network', type=int, default=128)
    parser.add_argument('--horizon', help='number of steps taken in the environment before terminating the episode (prevents very long episodes)', type=int, default=np.inf)
    parser.add_argument('--render_step', help='see above', type=int, default=50)
    parser.add_argument('--num_actions', help='Number of action space to discretize to', type=int, default=100)

    args = parser.parse_args()

    if args.render:
        if args.algorithm == "A3C":
            renderA3C(**vars(args))
        else:
            renderDDQN(**vars(args))

    else:
        if args.algorithm == 'A3C':
            # Create global actor-critic holding main models
            agent = A3C(max_episodes=args.episodes, discount_rate=args.discount, step_max=args.step_max,
                        actor_lr=args.actor_lr, critic_lr=args.critic_lr, n_threads=args.threads,
                        eval_repeats=args.eval_repeats, no_log=args.no_log)

        else:
            agent = DDQN(vars(args))

        performance = agent.run()
        agent.plot_training(performance)
        if type(agent) is A3C:
            agent.plot_workers()
        performance = agent.test()
        agent.plot_test(performance)
        agent.save_models()


if __name__ == "__main__":
    main()
