## Project Structure
```bash
├── maze_env/
│   ├── __init__.py
│   └── environment.py  # MazeEnv class
├── rl_algorithms/
│   ├── __init__.py
│   ├── simple_q_table.py   # Simple Q-learning agent implementation
│   ├── dqn.py          # DQN agent implementation
│   └── sarsa.py       # SARSA agent implementation
├── experiments/
│   ├── __init__.py
│   ├── q_learning_experiment.py # Script to run Q-learning on the maze
│   └── dqn_experiment.py        # Script to run DQN on the maze
├── README.md
├── setup.py
└── requirements.txt
```
## AI tools

The project has utilised AI tools including copilot and Gemini 2.0 Flash for the following tasks:
- add comments and docstrings to the code base
- code optimisation and refractoring
- maze generation
- enhanced visuals in visualisations such as customised colour and heat map settings
- debugging process:
    - hyperparameter adjustment suggestions for undesirable tranining results
    - improving extremely slow training speed with keras