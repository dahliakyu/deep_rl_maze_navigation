class Utils:

    @staticmethod
    def initialize_q_table(env):
        """
        Initializes the Q-table with zeros for all state-action pairs.

        Args:
            env (MazeEnv): The maze environment.

        Returns:
            dict: The initialized Q-table.
        """
        q_table = {}
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                q_table[(i, j)] = [0.0] * len(env.action_space)  # Initialize Q-values to 0
        return q_table