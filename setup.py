from setuptools import setup, find_packages

setup(
    name='gomoku_madrona',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch',
        'numpy',
        'tensorboard',
        'gym',
    ],
    description='Gomoku (Five in a Row) RL environment built on the Madrona game engine',
)
