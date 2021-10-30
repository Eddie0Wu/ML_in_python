# Machine Learning in Python

This repository contains machine learning models written in Python. Although there are existing packages that can readily implement these models, the models in this repository show how the algorithms work under the hood. The models can be conveniently adapted to various applications.

## File structure

[`decision_tree.py`](decision_tree.py): contains codes for implementing a decision tree. It also has functions for pruning which helps reduce overfitting to training data, and for evaluation of performance.

The `numpy_neural_network` folder contains codes for writing a neural network from scratch in Numpy. It has 4 files:
- [`nn_lib.py`](numpy_neural_network/nn_lib.py) contains the classes required for building a feed forward neural network.
- [`house_value_regression.py`](numpy_neural_network/house_value_regression.py) contains the codes for building a neural network to predict housing prices.
- [`housing.csv`](numpy_neural_network/housing.csv) is the dataset for the housing prices regression.
- [`iris.dat`](numpy_neural_network/iris.dat) is the Iris dataset for testing the self-written neural network classes and functions.


The `reinforcement_learning` folder contains two sub-folders: `deep_RL` and `value_iteration`.

`deep_RL` contains the codes for deep reinforcement learning. It has 3 files:
- [`agent.py`](reinforcement_learning/deep_RL/agent.py): it contains the agent class that defines how the agent's behaviour and learning process. Edit this file to alter the agent's behaviour.
- [`random_environment.py`](reinforcement_learning/deep_RL/random_environment.py): contains the environment class that defines the random environment in which the agent will learn. It also contains codes for visualising the environment and the learning process.
- [`train_and_test.py`](reinforcement_learning/deep_RL/train_and_test.py): this is the main script for training and testing the agent.

`value_iteration` contains the codes for reinforcement learning with discrete states and actions. It has 1 file:
- [`RL_discrete_state.ipynb`](reinforcement_learning/deep_RL/RL_discrete_state.ipynb): the notebook builds the Gridworld environment and contains the codes for implementing policy evaluation, policy iteration and value iteration.


The `regression` folder contains codes for gradient descent and polynomial regression. It is mainly for learning the optimisation processes behind regressions. The codes are not readily adaptable for actual applications. It has 2 files:
- [`numpy_gradient_descent.py`](regression/numpy_gradient_descent.py): contains the derivatives of the desired functions for running gradient descent.
- [`polynomial_regression.py`](regression/polynomial_regression.py): contains the codes for running polynomial regression, and visualising the outcome.


