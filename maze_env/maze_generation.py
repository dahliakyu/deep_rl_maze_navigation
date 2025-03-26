import random
import numpy as np
import json

def generate_maze(rows, cols, extra_passages=0):
    """
    Generates a maze using a randomized version of Kruskal's algorithm.

    Args:
        rows (int): Number of rows in the maze.
        cols (int): Number of columns in the maze.
        extra_passages (int, optional): Number of extra passages to add to create loops. Defaults to 0.

    Returns:
        list: A 2D list representing the maze, where each cell is a dictionary with keys 'N', 'E', 'S', 'W' indicating the presence of walls.
    """
    # Initialize each cell with all walls
    maze = [[{'N': True, 'E': True, 'S': True, 'W': True} for _ in range(cols)] for _ in range(rows)]
    
    # Generate all possible walls between adjacent cells
    walls = []
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                walls.append(((i, j), (i, j + 1)))  # East walls
            if i < rows - 1:
                walls.append(((i, j), (i + 1, j)))  # South walls
    
    random.shuffle(walls)
    
    parent = {}
    
    def find(cell):
        """
        Finds the root of the set containing the cell.
        
        Args:
            cell (tuple): The cell to find.
        
        Returns:
            tuple: The root of the set containing the cell.
        """
        if parent[cell] != cell:
            parent[cell] = find(parent[cell])
        return parent[cell]
    
    def union(cell1, cell2):
        """
        Unites the sets containing cell1 and cell2.
        
        Args:
            cell1 (tuple): The first cell.
            cell2 (tuple): The second cell.
        """
        root1 = find(cell1)
        root2 = find(cell2)
        if root1 != root2:
            parent[root2] = root1
    
    # Initialize each cell as its own parent
    for i in range(rows):
        for j in range(cols):
            parent[(i, j)] = (i, j)
    
    # Process each wall in random order
    for wall in walls:
        cell1, cell2 = wall
        if find(cell1) != find(cell2):
            dx = cell2[0] - cell1[0]
            dy = cell2[1] - cell1[1]
            
            if dx == 0 and dy == 1:
                maze[cell1[0]][cell1[1]]['E'] = False
                maze[cell2[0]][cell2[1]]['W'] = False
            elif dx == 1 and dy == 0:
                maze[cell1[0]][cell1[1]]['S'] = False
                maze[cell2[0]][cell2[1]]['N'] = False
            
            union(cell1, cell2)
    
    # Add extra passages to create loops
    additional_walls = []
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1 and maze[i][j]['E']:
                additional_walls.append(((i, j), (i, j + 1)))
            if i < rows - 1 and maze[i][j]['S']:
                additional_walls.append(((i, j), (i + 1, j)))
    
    random.shuffle(additional_walls)
    for k in range(min(extra_passages, len(additional_walls))):
        cell1, cell2 = additional_walls[k]
        i1, j1 = cell1
        i2, j2 = cell2
        if j2 == j1 + 1:
            maze[i1][j1]['E'] = False
            maze[i2][j2]['W'] = False
        elif i2 == i1 + 1:
            maze[i1][j1]['S'] = False
            maze[i2][j2]['N'] = False
    
    return maze

def maze_to_numpy(maze):
    """
    Converts the maze to a numpy array representation.

    Args:
        maze (list): A 2D list representing the maze, where each cell is a dictionary with keys 'N', 'E', 'S', 'W' indicating the presence of walls.

    Returns:
        np.ndarray: A numpy array representing the maze, where 1 indicates a wall and 0 indicates a passage.
    """
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    size = 2 * rows + 1
    grid = np.ones((size, size), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            x, y = 2*i+1, 2*j+1
            grid[x][y] = 0
            if not maze[i][j]['N']: grid[x-1][y] = 0
            if not maze[i][j]['S']: grid[x+1][y] = 0
            if not maze[i][j]['W']: grid[x][y-1] = 0
            if not maze[i][j]['E']: grid[x][y+1] = 0
    
    return grid[1:-1, 1:-1]

if __name__ == "__main__":
    rows = 5
    cols = 5
    extra_passages = 2  # Adjust this value to increase complexity
    maze = generate_maze(rows, cols, extra_passages)
    numpy_maze = maze_to_numpy(maze)
    print(numpy_maze)
    
    with open("maze.json", "w") as f:
        json.dump(numpy_maze.tolist(), f)