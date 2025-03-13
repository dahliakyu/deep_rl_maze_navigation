import random
from collections import deque

class ReplayBuffer:
    """A simple replay buffer for storing and sampling experiences."""

    def __init__(self, capacity):
        """
        Initializes the replay buffer.

        Args:
            capacity (int): The maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Adds an experience to the replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Samples a batch of experiences from the replay buffer."""
        if len(self.buffer) < batch_size:
            return None  # Not enough experiences to sample a batch
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Returns the current size of the replay buffer."""
        return len(self.buffer)