## Project Structure

├── maze_env/
│   ├── __init__.py
│   ├── environment.py  # MazeEnv class
│   ├── agent.py      # Agent ABC
│   └── utils.py      # Utility functions
├── rl_algorithms/
│   ├── __init__.py
│   ├── simple_q_table.py   # Simple Q-learning agent implementation
│   ├── dqn.py          # DQN agent implementation
│   └── sarsa.py       # SARSA agent implementation
├── replay_buffer.py # ReplayBuffer Class
├── visualization.py  # Visualization Class
├── experiments/
│   ├── __init__.py
│   ├── q_learning_experiment.py # Script to run Q-learning on the maze
│   └── dqn_experiment.py        # Script to run DQN on the maze
├── notebooks/             # Jupyter notebooks for exploration
│   ├── q_learning_demo.ipynb
│   └── dqn_demo.ipynb
├── README.md
├── setup.py
└── requirements.txt

## AI tools

The project has utilised AI tools including copilot and Gemini 2.0 Flash to add comments and docstrings to the code base, code refractoring and redundant code cleaning.