from agent_dqn import Agent_DQN
import argparse
from test import test
import time

def parse():
    parser = argparse.ArgumentParser(description="SnakeRL/snake_env")
    parser.add_argument('--train', action='store_true', help='whether train DQN')
    parser.add_argument('--test', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    start_time = time.time()
    if args.train:
        agent = Agent_DQN(args)
        print("Training")
        agent.train()

    if args.test:
        agent = Agent_DQN(args)
        test(agent, total_episodes=100, lives = 1)
    print('running time:',time.time()-start_time)

if __name__ == '__main__':
    args = parse()
    run(args)
