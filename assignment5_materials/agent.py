import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN, DQN_LSTM
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            ### CODE ####
            # Choose a random action
            action = random.randrange(self.action_size)
        else:
            ### CODE ####
            # Choose the best action
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action = self.policy_net(state).max(1)[1].item()

        return action

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        # [(state: (5, 84, 84), actioon, reward, done)]
        mini_batch = self.memory.sample_mini_batch(frame) # (batch_size, 4)
        mini_batch = np.array(mini_batch).transpose() # (4, batch_size)

        history = np.stack(mini_batch[0], axis=0) # (batch_size, 5, 84, 84)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda() # (batch_size, 4, 84, 84)
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda() # (batch_size)
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda() # (batch_size)
        next_states = np.float32(history[:, 1:, :, :]) / 255. # (batch_size, 4, 84, 84)
        next_states = torch.from_numpy(next_states).cuda() # (batch_size, 4, 84, 84)

        # checks if the game is over
        dones = mini_batch[3] # (batch_size)

        # non-terminal mask [False, True] -> [1, 0]
        mask = torch.tensor(list(map(int, dones==False)), dtype=torch.uint8).cuda() # (batch_size)

        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        curr_state_actions: Tensor = self.policy_net(states) # (batch_size, action_size)
        curr_state_values = curr_state_actions.gather(1, actions.unsqueeze(1)).squeeze(1) # (batch_size)

        # Compute Q function of next state
        ### CODE ####
        next_state_actions: Tensor = self.policy_net(next_states) # (batch_size, action_size)

        # Find maximum Q-value of action at next state from policy net
        ### CODE ####
        next_state_values = next_state_actions.max(1)[0] * mask.float() # (batch_size)
        next_state_values = next_state_values.detach()

        # Compute the Huber Loss
        ### CODE ####
        criterion = nn.SmoothL1Loss()
        expected_state_values = next_state_values * self.discount_factor + rewards
        loss = criterion(expected_state_values, curr_state_values)

        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        self.scheduler.step()


class LSTM_Agent(Agent):
    def __init__(self, action_size):
        super().__init__(action_size)

        # Generate the memory
        self.memory = ReplayMemoryLSTM()

        # Create the policy net
        self.policy_net = DQN_LSTM(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)


    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state: np.ndarray, hidden = None) -> int:
        if np.random.rand() <= self.epsilon:
            ### CODE ####
            # Choose a random action
            action = random.randrange(self.action_size)
        else:
            ### CODE ####
            # Choose the best action
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action, hidden = self.policy_net(state, hidden)
            action = action.argmax().item()

        return action, hidden

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame) # (batch_size, 4)
        mini_batch = np.array(mini_batch).transpose() # (4, batch_size)

        history = np.stack(mini_batch[0], axis=0) # (batch_size, 21, 84, 84)
        states = np.float32(history[:, :lstm_seq_length, :, :]) / 255.
        states = torch.from_numpy(states).cuda() # (batch_size, 20, 84, 84)
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda() # (batch_size)

        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda() # (batch_size)
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        next_states = torch.from_numpy(next_states).cuda() # (batch_size, 20, 84, 84)

        # checks if the game is over
        dones = mini_batch[3] # (batch_size)
        mask = torch.tensor(list(map(int, dones==False)), dtype=torch.uint8).cuda() # (batch_size)

        ### All the following code is nearly same as that for Agent

        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        curr_state_actions, hidden = self.policy_net(states) # (batch_size, action_size)
        curr_state_values = curr_state_actions.gather(1, actions.unsqueeze(1)).squeeze(1) # (batch_size)

        # Compute Q function of next state
        ### CODE ####
        next_state_actions, _ = self.policy_net(next_states, hidden) # (batch_size, action_size)

        # Find maximum Q-value of action at next state from policy net
        ### CODE ####
        next_state_values = next_state_actions.max(1)[0] * mask.float() # (batch_size)
        next_state_values = next_state_values.detach()

        # Compute the Huber Loss
        ### CODE ####
        criterion = nn.SmoothL1Loss()
        expected_state_values = next_state_values * self.discount_factor + rewards
        loss = criterion(expected_state_values, curr_state_values)

        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

