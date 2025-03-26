import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time
import matplotlib.patches as patches

class RandomShapeEnvironment2D:
    def __init__(self, grid_size=(5, 5), num_shapes=5, shape_complexity=5, max_steps=1000, max_shapes=None):
        self.grid_size = grid_size
        self.num_shapes = num_shapes
        self.shape_complexity = shape_complexity
        self.max_steps = max_steps
        self.goal_position = np.array([grid_size[0] - 1, grid_size[1] - 1])
        self.max_shapes = max_shapes if max_shapes is not None else num_shapes
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        self.agent_position = np.array([0, 0])  # Start at top-left
        self.shapes = self._generate_random_shapes()
        self.steps = 0
        return self._get_state()

    def _generate_random_shapes(self):
        """Generate random polygon shapes."""
        shapes = []
        for _ in range(self.num_shapes):
            num_vertices = random.randint(3, self.shape_complexity)
            # Generate shape center away from start and goal positions
            center_x = random.uniform(0.2 * self.grid_size[0], 0.8 * self.grid_size[0])
            center_y = random.uniform(0.2 * self.grid_size[1], 0.8 * self.grid_size[1])
            
            # Base radius scaled with grid size
            radius = random.uniform(0.05 * min(self.grid_size), 0.2 * min(self.grid_size))
            
            vertices = []
            for i in range(num_vertices):
                angle = 2 * np.pi * i / num_vertices
                # Add random variation to vertices
                offset_x = radius * np.cos(angle) + random.uniform(-radius/3, radius/3)
                offset_y = radius * np.sin(angle) + random.uniform(-radius/3, radius/3)
                
                # Ensure vertices are within grid bounds
                vertex_x = int(np.clip(center_x + offset_x, 0, self.grid_size[0] - 1))
                vertex_y = int(np.clip(center_y + offset_y, 0, self.grid_size[1] - 1))
                vertices.append([vertex_x, vertex_y])
            
            shapes.append(np.array(vertices))
        return shapes

    def _get_state(self):
        """Get the current state representation."""
        agent_pos_norm = self.agent_position / np.array(self.grid_size)
        goal_pos_norm = self.goal_position / np.array(self.grid_size)
        
        # Initialize shape centers with padding
        shape_centers = np.zeros((self.max_shapes, 2))
        
        # Fill available shape centers, ensuring we don't exceed max_shapes
        for i, shape in enumerate(self.shapes[:self.max_shapes]):
            center = np.mean(shape, axis=0) / np.array(self.grid_size)
            shape_centers[i] = center
            
        # Flatten and concatenate
        state = np.concatenate([
            agent_pos_norm,
            goal_pos_norm,
            shape_centers.flatten()
        ]).astype(np.float32)
        
        return state

    def step(self, action):
        """Take a step in the environment."""
        self.steps += 1
        
        # Store previous position for collision checking
        previous_position = self.agent_position.copy()
        
        # Process action
        if action == 0:  # Left
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:  # Right
            self.agent_position[0] = min(self.grid_size[0] - 1, self.agent_position[0] + 1)
        elif action == 2:  # Up
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 3:  # Down
            self.agent_position[1] = min(self.grid_size[1] - 1, self.agent_position[1] + 1)

        # Initialize reward
        reward = -0.01  # Small step penalty to encourage efficiency
        done = False

        # Check if goal reached
        if np.array_equal(self.agent_position, self.goal_position):
            reward = 1.0
            done = True
        
        # Check if max steps reached
        elif self.steps >= self.max_steps:
            done = True
            reward = -0.5  # Penalty for not reaching goal in time

        # Optional: Check for shape collisions (simplified point-in-polygon check)
        if self._check_collision():
            self.agent_position = previous_position  # Revert move
            reward = -0.1  # Collision penalty

        return self._get_state(), reward, done

    def _check_collision(self):
        """Check if agent collides with any shape."""
        # Simplified collision check - just checking if agent position matches any shape vertex
        for shape in self.shapes:
            for vertex in shape:
                if np.array_equal(self.agent_position, vertex):
                    return True
        return False

    def render(self, ax=None):
        """Render the environment."""
        plt.clf()
        
        # Create grid
        grid = np.zeros(self.grid_size)
        
        # Draw shapes
        for shape in self.shapes:
            for vertex in shape:
                grid[vertex[1], vertex[0]] = 0.5  # Gray for shapes
        
        # Draw agent
        grid[self.agent_position[1], self.agent_position[0]] = 1.0  # White for agent
        
        # Draw goal
        grid[self.goal_position[1], self.goal_position[0]] = 0.75  # Light gray for goal
        
        # Display
        plt.imshow(grid, origin='lower', cmap='gray')
        plt.title(f'Step: {self.steps}')
        plt.pause(0.01)

# Simple 2D neural network
class SimpleNetwork2D(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size))

    def forward(self, x):
        return self.network(x)


class BaseTrainer2D:
    def __init__(self, env, learning_rate=0.001):
        self.env = env
        self.learning_rate = learning_rate

    def _train_step(self, net: nn.Module, batch: List[Tuple], optimizer: optim.Optimizer, device: torch.device):
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        current_q = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q = net(next_states_t).max(1)[0]
            next_q[done_mask] = 0.0
            expected_q = rewards_t + 0.99 * next_q

        loss = nn.MSELoss()(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, episodes: int = 2000) -> Tuple[List[float], List[float], nn.Module]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Input size: agent pos (2) + goal pos (2) + shape centers (num_shapes * 2)
        input_size = 4 + self.env.max_shapes * 2 # Use max_shapes here for input size
        print(f"BaseTrainer2D Input size for network: {input_size}")
        net = SimpleNetwork2D(input_size, 4).to(device) # Output size is 4 actions (L, R, U, D)
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)

        buffer = deque(maxlen=10000)
        epsilon = 1.0

        success_rates = []
        success_window = deque(maxlen=50)
        losses = []

        for episode in range(episodes):
            state = self.env.reset()
            print(f"BaseTrainer2D State shape: {state.shape}")
            total_reward = 0
            episode_losses = []
            self.env.render() # Render at start of episode

            while True:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_vals = net(state_t)
                        action = torch.argmax(q_vals).item()

                next_state, reward, done = self.env.step(action)
                buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(buffer) >= 64: # Batch size
                    loss = self._train_step(net, random.sample(buffer, 64), optimizer, device)
                    episode_losses.append(loss)

                self.env.render() # Render at each step

                if done:
                    break

            success_window.append(1 if total_reward > 0 else 0) # Simple success if reward > 0 (reached goal)
            success_rate = sum(success_window) / len(success_window) if success_window else 0
            success_rates.append(success_rate)
            losses.append(np.mean(episode_losses) if episode_losses else 0)

            epsilon = max(0.01, epsilon * 0.995) # Epsilon decay

            if episode % 100 == 0:
                print(f"Episode {episode}, Success Rate: {success_rate:.2f}, Avg Loss: {np.mean(losses[-100:]):.3f}, Epsilon: {epsilon:.2f}")

        return success_rates, losses, net

class EnhancedNetwork2D(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super().__init__()
        # Replace BatchNorm with Layer Norm which works with any batch size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # Ensure input is at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

class SimpleCurriculumTrainer2D:
    def __init__(self, initial_num_shapes=0, max_num_shapes=5, initial_grid_size=(10, 10), max_grid_size=(20, 20), shape_complexity=5):
        self.initial_num_shapes = initial_num_shapes
        self.max_num_shapes = max_num_shapes
        self.current_num_shapes = initial_num_shapes

        self.initial_grid_size = initial_grid_size
        self.max_grid_size = max_grid_size
        self.current_grid_size = initial_grid_size

        self.shape_complexity = shape_complexity

        self.success_threshold = 0.8
        self.success_window = deque(maxlen=50)

    def train(self, episodes=2000, visualize=True) -> nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Input size: agent pos (2) + goal pos (2) + shape centers (num_shapes * 2) - max_num_shapes for consistent input size
        input_size = 4 + self.max_num_shapes * 2
        print(f"CurriculumTrainer2D Input size for network: {input_size}")
        net = SimpleNetwork2D(input_size, 4).to(device) # Output size is 4 actions (L, R, U, D)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        buffer = deque(maxlen=10000)
        epsilon = 1.0

        # Visualization setup
        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # create one subplot
            fig.canvas.manager.set_window_title("Shape Exploration")  # Title to the window
            plt.ion()  # Turn on interactive mode for live updates
            fig.show()

        for episode in range(episodes):
            print(f"--- Episode {episode} ---") # Episode start marker
            print(f"Current Trainer num_shapes: {self.current_num_shapes}, Grid Size: {self.current_grid_size}") # ADDED: Print trainer's current num_shapes and grid size
            env = RandomShapeEnvironment2D(grid_size=self.current_grid_size, num_shapes=self.current_num_shapes, shape_complexity=self.shape_complexity, max_shapes=self.max_num_shapes) # pass max_num_shapes here
            print(f"Current num_shapes in env (after creation): {env.num_shapes}") # ADDED: Print env's num_shapes right after creation
            state = env.reset()
            print(f"CurriculumTrainer2D State shape: {state.shape}")
            print(f"Current num_shapes in env (after reset): {env.num_shapes}") # ADDED: Print env's num_shapes after reset
            total_reward = 0
            done = False

            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_vals = net(state_t)
                        action = torch.argmax(q_vals).item()

                next_state, reward, done = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(buffer) >= 64:
                    batch = random.sample(buffer, 64)
                    self._train_step(net, batch, optimizer, device)

                # Render the environment
                if visualize:
                    env.render(ax=ax)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

            self.success_window.append(1 if total_reward > 0 else 0)
            success_rate = sum(self.success_window) / len(self.success_window) if self.success_window else 0.0

            # Curriculum progression (adjust num_shapes first, then grid_size)
            if (len(self.success_window) == self.success_window.maxlen and
                success_rate >= self.success_threshold):
                if self.current_num_shapes < self.max_num_shapes:
                    self.current_num_shapes += 1
                    print(f"Progressing to {self.current_num_shapes} shapes")
                elif self.current_grid_size < self.max_grid_size:
                    next_grid_size = (min(self.current_grid_size[0] + 2, self.max_grid_size[0]), min(self.current_grid_size[1] + 2, self.max_grid_size[1])) # Increment grid size by 2 in each dimension
                    if next_grid_size != self.current_grid_size: # Avoid infinite loop if already at max_grid_size in one dimension
                        self.current_grid_size = next_grid_size
                        print(f"Progressing to grid size {self.current_grid_size}")
                    else:
                        print("Reached max complexity.")

                self.success_window.clear()
                epsilon = 1.0  # Reset exploration rate

            epsilon = max(0.01, epsilon * 0.995)

            if episode % 100 == 0:
                print(f"Episode {episode}, Success Rate: {success_rate:.2f}, Shapes: {self.current_num_shapes}, Grid Size: {self.current_grid_size}")

        return net

    def _train_step(self, net, batch, optimizer, device): # <----- RE-ADDED _train_step METHOD
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(device)
        print(f"Shape of states_t in _train_step: {states_t.shape}") # ADDED: Print states_t shape
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        current_q = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q = net(next_states_t).max(1)[0]
            next_q[done_mask] = 0.0
            expected_q = rewards_t + 0.99 * next_q

        loss = nn.MSELoss()(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



class AdaptiveCurriculumTrainer2D(SimpleCurriculumTrainer2D):
    def __init__(self, initial_num_shapes=0, max_num_shapes=5, 
                 initial_grid_size=(10, 10), max_grid_size=(20, 20), 
                 shape_complexity=5):
        super().__init__(initial_num_shapes, max_num_shapes, 
                        initial_grid_size, max_grid_size, shape_complexity)
        self.success_threshold = 0.7
        self.min_success_threshold = 0.6
        self.success_decay = 0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.batch_size = 64
        self.min_buffer_size = 1000
        self.target_update_frequency = 10

    def _train_step(self, net, target_net, batch, optimizer, device):
        """Performs a single training step with safeguards for batch size."""
        if len(batch) < 2:  # Skip training if batch is too small
            return 0.0

        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        current_q = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q = target_net(next_states_t).max(1)[0]
            next_q[done_mask] = 0.0
            target_q = rewards_t + self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def adjust_difficulty(self, success_rate):
      """Adjusts the environment difficulty based on success rate."""
      if success_rate > self.success_threshold:
        # Increase difficulty
        if self.current_num_shapes < self.max_num_shapes:
          self.current_num_shapes += 1
          print("Increasing number of shapes.")
        elif self.current_grid_size < self.max_grid_size:
          next_grid_size = (min(self.current_grid_size[0] + 2, self.max_grid_size[0]),
                            min(self.current_grid_size[1] + 2, self.max_grid_size[1]))
          if next_grid_size != self.current_grid_size:
            self.current_grid_size = next_grid_size
            print("Increasing grid size.")
        else:
          print("Maximum difficulty reached.")
      elif success_rate < self.min_success_threshold:
          # Decrease difficulty
          if self.current_num_shapes > 0:
              self.current_num_shapes = max(0, self.current_num_shapes - 1)
              print("Decreasing number of shapes.")
          elif self.current_grid_size > self.initial_grid_size:
              next_grid_size = (max(self.current_grid_size[0] - 2, self.initial_grid_size[0]),
                                  max(self.current_grid_size[1] - 2, self.initial_grid_size[1]))

              if next_grid_size != self.current_grid_size:  # prevent infinite loop if we hit min size on one dimension
                  self.current_grid_size = next_grid_size
                  print("Decreasing grid size.")
              else:
                  print("Minimum difficulty reached.")

      # Adapt success threshold
      self.success_threshold = max(self.min_success_threshold, self.success_threshold * self.success_decay)

    def train(self, episodes=2000, visualize=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = 4 + self.max_num_shapes * 2
        net = EnhancedNetwork2D(input_size, 4).to(device)
        target_net = EnhancedNetwork2D(input_size, 4).to(device)
        target_net.load_state_dict(net.state_dict())
        
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        buffer = deque(maxlen=10000)
        epsilon = 1.0
        
        success_window = deque(maxlen=50)
        losses = []

        # Visualization setup
        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # create one subplot
            fig.canvas.manager.set_window_title("Shape Exploration")  # Title to the window
            plt.ion()  # Turn on interactive mode for live updates
            fig.show()

        for episode in range(episodes):
            env = RandomShapeEnvironment2D(
                grid_size=self.current_grid_size,
                num_shapes=self.current_num_shapes,
                shape_complexity=self.shape_complexity,
                max_shapes=self.max_num_shapes
            )
            
            state = env.reset()
            total_reward = 0
            episode_loss = []
            
            if visualize:
                env.render(ax=ax)  # Render initial state
                fig.canvas.draw()
                fig.canvas.flush_events()

            while True:
                # Ensure state is properly batched
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_vals = net(state_t)
                        action = torch.argmax(q_vals).item()
                
                next_state, reward, done = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                
                # Only train if we have enough samples
                if len(buffer) >= max(self.batch_size, self.min_buffer_size):
                    batch = random.sample(buffer, self.batch_size)
                    loss = self._train_step(net, target_net, batch, optimizer, device)
                    if loss > 0:  # Only append non-zero losses
                        episode_loss.append(loss)
                
                if done:
                    break
                
                if visualize:
                    env.render(ax=ax)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            
            # Update metrics
            success = total_reward > 0
            success_window.append(1 if success else 0)
            success_rate = sum(success_window) / len(success_window)
            
            if episode_loss:
                losses.append(np.mean(episode_loss))
            
            # Update target network
            if episode % self.target_update_frequency == 0:
                target_net.load_state_dict(net.state_dict())
            
            if success_rate >= self.success_threshold:
                self.adjust_difficulty(success_rate)
                success_window.clear()
                epsilon = 1.0
            else:
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            
            if episode % 100 == 0:
                print(f"Episode {episode}")
                print(f"Success Rate: {success_rate:.2f}")
                print(f"Shapes: {self.current_num_shapes}")
                print(f"Grid Size: {self.current_grid_size}")
                print(f"Epsilon: {epsilon:.3f}")
                if losses:
                    print(f"Average Loss: {np.mean(losses[-100:]):.3f}")
                print("-------------------")
        
        return net, losses

# Example usage:
if __name__ == "__main__":
    trainer = AdaptiveCurriculumTrainer2D(initial_num_shapes=2, max_num_shapes=5, initial_grid_size=(10, 10), max_grid_size=(20, 20))
    trained_net, losses = trainer.train(episodes=2000, visualize=True)



    # Plotting the losses
    plt.ioff()  # Turn off interactive mode
    plt.plot(losses)
    plt.xlabel("Episode (Loss computed only when batch size is met)")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()