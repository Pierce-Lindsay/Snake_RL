
import argparse
import numpy as np
import time
from tqdm import tqdm


def test(agent, total_episodes=100, lives = 1):
    rewards = []
    for _ in tqdm(range(total_episodes)):
        agent.reset()
        episode_reward = 0.0
        for _ in range(lives):
            state = agent.make_board()
            agent.init_game_setting()
            terminated = False
            while True:
                action = agent.make_action(state, test=True)
                agent._apply_action(action)
                agent.process()
                reward = agent._calculate_reward()
                episode_reward += reward
                terminated = agent.game_over
                if terminated:
                    break
                state = agent.make_board()
        rewards.append(episode_reward)
    print(f'Run {total_episodes} episodes for {lives} life/lives each.')
    print('Mean:', np.mean(rewards))
    print('rewards', rewards)