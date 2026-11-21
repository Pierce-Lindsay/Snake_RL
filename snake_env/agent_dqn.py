#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import copy

from train import SnakeAI
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(SnakeAI):
    def __init__(self, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super().__init__()
        self.num_episodes = 10000
        self.gamma = 0.99
        self.lr = 1.5e-4
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1.0
        self.epsilon_min = 0.075
        self.decay_length = 0.85 * self.num_episodes
        self.decay_rate = (self.epsilon - self.epsilon_min)/self.decay_length
        self.r_buffer = deque(maxlen=10000)
        self.model = DQN().to(self.device)
        self.target = copy.deepcopy(self.model).to(self.device)
        self.target.eval()
        self.learn_steps = 0
        self.update_freq = 5000
        self.truncated_step = 1000
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.writer = SummaryWriter("runs/10k/")
        
        if args.test:
            #you can load your model here
            print('loading trained model')
            self.model.load_state_dict(torch.load("model-5000.pth",map_location = self.device))
            self.model.eval()

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                shape: (7, 30, 40)
        Return:
            action: int
                the predicted action from trained model
        """

        if np.random.rand() < self.epsilon and test == False:
            action = np.random.randint(3)
        else:
            x = torch.as_tensor(observation, dtype = torch.float32,device = self.device)
            x = x.unsqueeze(0)
            with torch.no_grad():
                actions = self.model(x)
            action = actions.argmax(dim=1).item()
        return action
    
    def push(self,state, action, reward, next_state, terminated, truncated):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        self.r_buffer.append((state.copy(), action, reward, next_state.copy(), terminated, truncated))
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        return random.sample(self.r_buffer, self.batch_size)
    
    def make_plots(self,total_rewards, epsilon_values):
        episodes = np.arange(1, len(total_rewards)+1)
        fig, ax1 = plt.subplots(figsize=(12,6))

        ax1.plot(episodes,total_rewards, color="b",label="Rolling Average Reward(last 30)")
        ax1.set_xlabel("Training Episode")
        ax1.set_ylabel("Rolling Average Reward(last 30)")

        ax2 = ax1.twinx()
        ax2.plot(episodes, epsilon_values, color="r", label = "Epsilon")
        ax2.set_ylabel("Epsilon")
        ax2.set_ylim(0,1.05)

        plt.title("Training")
        fig.tight_layout()
        fig.savefig("Reward-10000.png")


    def train(self):
        """
        Implement your training algorithm here
        """
        last_30 = deque(maxlen=30)
        total_rewards=[]
        epsilon_values=[]
        for episode in range(self.num_episodes):
            self.reset()
            state = self.make_board()
            total_reward = 0.0
            steps = 0
            total_loss = 0
            loss_step = 0

            while True:
                action = self.make_action(state,False)
                self._apply_action(action)
                self.process()
                reward = self._calculate_reward()
                terminated = self.game_over
                truncated = (steps >= self.truncated_step)
                if terminated or truncated:
                    next_state = np.zeros_like(state)
                else:
                    next_state=self.make_board()
                self.push(state, action, reward, next_state, terminated, truncated)
                total_reward += reward
                if len(self.r_buffer) >= self.batch_size:
                    batch = self.replay_buffer()


                    states, actions, rewards, next_states, terminateds, truncateds = zip(*batch)
                    states = torch.as_tensor(np.stack(states),dtype=torch.float32, device = self.device)
                    next_states = torch.as_tensor(np.stack(next_states),dtype=torch.float32, device = self.device)
                    actions = torch.as_tensor(actions,dtype=torch.long, device = self.device)
                    rewards = torch.as_tensor(rewards,dtype=torch.float32, device = self.device)
                    terminateds = torch.as_tensor(terminateds,dtype=torch.float32, device = self.device)
                    # truncateds = torch.as_tensor(truncateds,dtype=torch.float32, device = self.device)

                    q_values = self.model(states)
                    predictions = q_values.gather(1,actions.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_q = self.target(next_states)
                        max_next = next_q.max(dim=1).values

                        target_q = rewards + self.gamma * (1-terminateds) *max_next
                    loss = F.smooth_l1_loss(predictions, target_q)
                    total_loss += loss.item()
                    loss_step += 1
                    self.optimizer.zero_grad()
                    loss.backward()
                    # print(
                    #     f"Loss: {loss.item():.12f} | "
                    #     f"Predictions: [{predictions.min().item():.6f}, " f"{predictions.max().item():.6f} | "
                    #     f"Targets: [{target_q.min().item():.6f}," f"{target_q.max().item():.6f} | "
                    #     f"Mean abs Error: {(predictions - target_q).abs().mean().item():.8f}"
                    #     )
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value = 1.0)
                    self.optimizer.step()
                    self.learn_steps += 1
                    if self.learn_steps >= self.update_freq:
                        self.target.load_state_dict(self.model.state_dict())
                        self.learn_steps = 0
                steps += 1
                state = next_state
                if terminated or truncated:
                    break
            avg_loss = total_loss/loss_step if loss_step != 0 else 0
            self.epsilon = max(self.epsilon - self.decay_rate, self.epsilon_min)
            # self.writer.add_scalar("Reward/episode", total_reward, episode)
            # self.writer.add_scalar("Loss/episode", avg_loss, episode)
            last_30.append(total_reward)
            total_rewards.append(float(np.mean(last_30)))
            epsilon_values.append(self.epsilon)
            print(
                f"Episode: {episode + 1} | "
                f"Episode reward: {total_reward:.2f} | "
                f"Loss: {avg_loss:.2f} | "
                f"Steps: {steps} | "
                f"Epsilon: {self.epsilon:.3f}"
                )
        torch.save(self.model.state_dict(), "model.pth")
        print("Model saved to model-10000.pth")
        # self.writer.close()
        self.make_plots(total_rewards, epsilon_values)
        print("Plot saved to Reward-10000.png")