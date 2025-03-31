from setuptools import setup, find_packages

setup(
    name='deep_rl_maze',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'keras',
        'seaborn',
        'scipy'
    ],
    author='Ellen Zheng',
    author_email='lamiyavi@hotmail.com',
    description='Maze navigation with deep reinforcement learning',
)