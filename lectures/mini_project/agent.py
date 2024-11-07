'''
Key improvements:

Optimized Hyperparameters:

alpha = 0.4 (higher learning rate for faster learning)
gamma = 0.9 (balanced future reward consideration)
Initial epsilon = 1.0 (start with full exploration)


Epsilon Decay:

Added epsilon decay for better exploration-exploitation balance
Minimum epsilon to ensure some exploration always occurs


Q-Learning Implementation:

Uses Q-Learning (off-policy TD control)
Better TD target calculation
Handles terminal states properly

This improved version should achieve better performance (>9.1 average reward) by:

Starting with more exploration
Gradually transitioning to exploitation
Learning faster with higher learning rate
Better balancing immediate vs future rewards

'''

from collections import defaultdict

import numpy as np


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        # Initialize hyperparameters
        self.alpha = 0.4  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_decay = 0.99999  # decay rate for epsilon
        self.epsilon_min = 0.01  # minimum exploration rate

        # Initialize Q-table
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # Epsilon-greedy action selection
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Q-Learning update
        best_next_action = np.argmax(self.Q[next_state])

        # TD Update
        target = reward + (self.gamma * self.Q[next_state][best_next_action] * (not done))
        current = self.Q[state][action]
        self.Q[state][action] = current + self.alpha * (target - current)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)