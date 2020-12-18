import argparse
import threading
import time

from algorithms.A3C.A3C import A3C, get_default_save_filename
from algorithms.A3C.A3CWorker import A3CWorker


def trainA3C(globalA3C: A3C, n_threads, no_log):
    # Instantiate one worker per thread
    if not no_log:
        workers = [A3CWorker(globalA3C) for _ in range(n_threads - 1)]
        workers.append(A3CWorker(globalA3C, log_info=True))
    else:
        workers = [A3CWorker(globalA3C) for _ in range(n_threads)]

    globalA3C.set_workers(workers)
    globalA3C.env.close()

    # Create threads
    threads = [threading.Thread(
        target=workers[i].run,
        daemon=True) for i in range(n_threads)]

    for t in threads:
        time.sleep(2)
        t.start()

    while globalA3C.episode < globalA3C.MAX_EPISODES:
        time.sleep(1)

    for t in threads:
        t.join()

    globalA3C.save_models()
    globalA3C.test()
    globalA3C.plot()


def renderA3C(**kwargs):
    if kwargs['load_file'] is None:
        parameters_file = get_default_save_filename(kwargs['episodes'], kwargs['threads'], kwargs['discount'],
                                                    kwargs['step_max'], kwargs['actor_lr'], kwargs['critic_lr'])
    else:
        parameters_file = kwargs['load_file']
    agent = A3C()
    agent.load_models(parameters_file)
    agent.render()


def renderQ(**kwargs):
    pass  # TODO


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-render', help='Render environment.', action='store_true')
    parser.add_argument('--algorithm', help='Algorithm to use.', default='A3C', choices=['A3C', 'Q'])
    parser.add_argument('--load_file', help='Custom filename from which to load weights before rendering.', default=None)
    parser.add_argument('--threads', help='Number of threads for A3C.', type=int, default=5)
    parser.add_argument('--episodes', help='Number of episodes.', type=int, default=100000)
    parser.add_argument('--discount', help='Discount rate.', type=float, default=0.99)
    parser.add_argument('--step_max', help='Max steps before update.', type=int, default=5)
    parser.add_argument('--actor_lr', help='Actor learning rate.', type=float, default=0.001)
    parser.add_argument('--critic_lr', help='Critic learning rate.', type=float, default=0.001)
    parser.add_argument('-no_log', help='Disable logging during training.', action='store_true')
    args = parser.parse_args()

    if args.render:
        if args.algorithm is "A3C":
            renderA3C(**vars(args))
        else:
            renderQ(**vars(args))

    elif args.algorithm is 'A3C':
        # Create global actor-critic holding main models
        agent = A3C(max_episodes=args.episodes, discount_rate=args.discount, step_max=args.step_max,
                    actor_lr=args.actor_lr, critic_lr=args.critic_lr, n_threads=args.threads)
        trainA3C(agent, args.threads, args.no_log)


if __name__ == "__main__":
    main()
