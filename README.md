<h3 align="center">
  <img src="assets/cartpole_example.gif" width="300">
</h3>

# DQN implementation - environment generalized
I've found it frustrating that the algorithm implementations I've come across can't be trained on different environments straight out of the box. Here I modify an implementation of DQN such that it can run on any environment (that has a discrete action space), simply by changing the environment name. 

Environments with non-discrete action spaces were not accounted for because DQN cannot be trained on such environments.

This is a fork of gsurma's repo, which was a DQN solution to the cartpole environment. Thanks gsurma.

### Changes made to create ability to generalize between environments
* Added a layer to the neural network called **Flatten()**. This layer transforms the dimensionality of the data in our neural network from whatever it may be in in the input layer, to one dimension in the next hidden layer.
* Created the function **reshape_dims** which finds the dimensions of the environment's observation space, which is used to reshape the state vector after every step.
* Created the function **find_input_shape** which finds the required shape of input layer of the neural network, based on the dimensionality of the observation space.
* Created an exception which is called when the dimensionality of the action space is not discrete.

### Other modifications from the original repo
* Added batch normalization to the keras model in order to increase training speed (so some results could be found on my puny laptop)
* Added a testing function that lets the user watch the agent play the game once with its trained policy
* Created a method for the model to make a prediction for a given state called **predict**
* Created the parameters NUM_EPISODES and WATCH_TRAINING. Former sets the number of episodes undertaken during training. Latter toggles whether the user views the agent plaing the game during training

### Hyperparameters:

* GAMMA = 0.95
* LEARNING_RATE = 0.001
* MEMORY_SIZE = 1000000
* BATCH_SIZE = 20
* EXPLORATION_MAX = 1.0
* EXPLORATION_MIN = 0.01
* EXPLORATION_DECAY = 0.995

### Other parameters
* NUM_EPISODES=100
* WATCH_TRAINING=False

### Model structure:
DQN with experience relay, and batch normalization.

1. Dense layer - input: **Observation space shape**, output: **24**, activation: **relu**
2. Dense layer - input **24**, output: **24**, activation: **relu**
3. Dense layer - input **24**, output: **Action space shape**, activation: **linear**

* **MSE** loss function
* **Adam** optimizer

##### Example trial gif

<img src="assets/cartpole_example.gif" width="200">



