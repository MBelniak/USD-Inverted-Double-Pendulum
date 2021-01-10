import argparse
from algorithms.A3C.A3C import A3C, get_default_save_filename
from algorithms.DDQN.DDQN import DDQN

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
    agent = DDQN()

    if kwargs['load_file'] is None:
        parameters_file = f"DDQN-{agent.max_episodes}"
    else:
        parameters_file = kwargs['load_file']

    agent.load_models(parameters_file)
    agent.render()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-render', help='Render environment.', action='store_true')
    parser.add_argument('--algorithm', help='Algorithm to use.', default='DDQN', choices=['A3C', 'DDQN'])
    parser.add_argument('--load_file', help='Custom filename from which to load weights before rendering.', default=None)
    parser.add_argument('--threads', help='Number of threads for A3C.', type=int, default=5)
    parser.add_argument('--episodes', help='Number of episodes.', type=int, default=100000)
    parser.add_argument('--discount', help='Discount rate.', type=float, default=0.99)
    parser.add_argument('--step_max', help='Max steps before update.', type=int, default=5)
    parser.add_argument('--actor_lr', help='Actor learning rate.', type=float, default=0.001)
    parser.add_argument('--critic_lr', help='Critic learning rate.', type=float, default=0.001)
    parser.add_argument('--eval_repeats', help='Number of evaluation runs in one performance evaluation.'
                                               ' Set to 0 to disable.', type=int, default=10)
    parser.add_argument('-no_log', help='Disable logging during training.', action='store_true')
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
            agent = DDQN(max_episodes=args.episodes, max_memory_size=50000)

        performance = agent.run()
        agent.plot_training(performance)
        if type(agent) is A3C:
            agent.plot_workers()
        performance = agent.test()
        agent.plot_test(performance)
        agent.save_models()


if __name__ == "__main__":
    main()
