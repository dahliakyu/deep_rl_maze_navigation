from maze_env.agent import Agent
from maze_env.utils import Utils
import random
import numpy as np

class QLearningAgent(Agent):
    """
    Q-learning agent for solving the maze environment.
    """

    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        """
        Initializes the Q-learning agent.

        Args:
            env (MazeEnv): The maze environment.
            learning_rate (float): The learning rate (alpha).
            discount_factor (float): The discount factor (gamma).
            exploration_rate (float): The exploration rate (epsilon).
        """
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = Utils.initialize_q_table(env)

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy policy.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The action to take.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action
            return random.choice(self.env.action_space)
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-value for the given state-action pair using the Q-learning update rule.

        Args:
            state (tuple): The state the agent was in.
            action (int): The action the agent took.
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state the agent transitioned to.
            done (bool): Whether the episode is finished.
        """
        if done:
            self.q_table[state][action] = reward
        else:
            best_next_action = int(np.argmax(self.q_table[next_state]))
            td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.learning_rate * td_error