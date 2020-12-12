import argparse
import threading
import time

from algorithms.A3C.A3C import A3C
from algorithms.A3C.A3CWorker import A3CWorker


def train(globalA3C: A3C, n_threads):
    # Instantiate one worker per thread
    workers = [A3CWorker(globalA3C) for _ in range(n_threads - 1)]
    workers.append(A3CWorker(globalA3C, log_info=True))
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', help='Algorithm to use.', default='A3C')
    parser.add_argument('--threads', help='Number of threads for A3C.', type=int, default=5)
    parser.add_argument('--episodes', help='Number of episodes.', type=int, default=50000)
    parser.add_argument('--discount', help='Discount rate.', type=float, default=0.99)
    parser.add_argument('--tmax', help='Max stapes before update.', type=int, default=5)
    parser.add_argument('--actor_lr', help='Actor learning rate.', type=float, default=0.001)
    parser.add_argument('--critic_lr', help='Critic learning rate.', type=float, default=0.001)
    parser.add_argument('--render', help='Render environment after training.', type=bool, default=False)
    args = parser.parse_args()

    if args.algorithm is 'A3C':
        # Create global actor-critic holding main model
        agent = A3C(max_episodes=args.episodes, discount_rate=args.discount, t_max=args.tmax,
                    actor_lr=args.actor_lr, critic_lr=args.critic_lr)
        train(agent, args.threads)
        if args.render:
            agent.render()


if __name__ == "__main__":
    main()
