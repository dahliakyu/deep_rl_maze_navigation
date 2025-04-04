## Project Structure
```bash
├── experiments
│   ├── ddqn_hyperparameter_search.py # Script to run hyperparameter search for ddqn agent
│   ├── ddqn_torch_experiment.py # Script to run single experiment with fixed hyperparameters and single maze config with ddqn agent
│   ├── dqn_experiment.py # Script to run single experiment with fixed hyperparameters and single maze config with dqn agent
│   ├── dqn_hyperparameter_search.py # Script to run hyperparameter search for dqn agent
│   ├── dqn_incremental.py # Attempt to train dqn agent incrementally, needs update and tuning
│   ├── manually_drawn_mazes # Directory for manually designed mazes for testing
│   ├── ql_hyperparameter_search.py # Script to run hyperparameter search for q-learning agent
│   ├── results # Directory for results from experiments, some data are attached as zip files for references
│   ├── sarsa_experiment.py # Script to run sarsa experiment with fixed hyperparameters and single maze config with sarsa agent
│   ├── sarsa_hyperparameter_search.py # Script to run hyperparameter search for sarsa agent
│   └── simple_q_table_experiment.py  # Script to run sarsa experiment with fixed hyperparameters and single maze config with q-learning agent
├── maze_env
│   ├── environment.py # MazeEnv class
│   └── maze_generation.py # Script for auto generated mazes with Kruskal's algorithm
├── requirements.txt
├── result_analysis
│   ├── TD_hyper_dependency.py # Script to show hyperparamter dependencies for Q-learning and SARSA agants
│   ├── TD_stat_test.py # Scirpt to run statistical testings for SARSA and Q-learning agents
│   ├── compare_TD_history.py # Scirpt to plot the reward or step history for a particualr complexity for SARSA and Q-learning agents
│   ├── hyperparameter_plot.py # Script to plot the hyperparameter search results
├── rl_algorithms
│   ├── simple_q_table.py   # Simple Q-learning agent implementation
│   ├── ddqn_torch.py   # Double DQN agent implementation
│   ├── dqn.py          # DQN agent implementation
│   └── sarsa.py       # SARSA agent implementation
├── README.md
# project config files
├── setup.py 
└── requirements.txt
```
## Installation

Before using any of the scripts please follow: 
```bash
pip install -e .
```
in the main directory
## Script Running Instruction

For methodologies of the project, please refer to the written paper named as deep_rl_maze_navigation.pdf. The research on the Q-learning and SARSA agents have been completed and further improvement for DQN and DDQN agents is planned for the future. 

For now the directory variables only work in windows environments and need to be updated according to your own operating system. For the result analysis scripts, run the desired experiments first and adjust the directory accordingly. The results used in current research were partially added as zip files for references. Run the experiments by calling them in desired terminals, currently no extra parameter is needed for any of the scirpts. sarsa_hyperparameter_search.py and ql_hyperparameter_search.py uses the maze generation functions in maze_generation.py directly, the rest of the experiments read from a saved maze file in json format that could be either generated by the script or encoded manually. To test the maze generation, simply enter the maze_env directory and run the maze_generation.py script. N.B. The actual maze size in numpy format is in the size of (2\*rows+1, 2\*cols+1).

## AI tools

The project has utilised AI tools including copilot and Gemini 2.0 Flash for the following tasks:
- add comments and docstrings to the code base
- code optimisation and refractoring
- maze generation
- enhanced visuals in visualisations such as customised colour and heat map settings
- debugging process:
    - hyperparameter adjustment suggestions for undesirable tranining results
    - improving extremely slow training speed with keras
