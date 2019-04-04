import random
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization,Activation
from tensorflow.keras.optimizers import Adam
from scores.score_logger import ScoreLogger

ENV_NAME = "TimePilot-ram-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
NUM_EPISODES=2
WATCH_TRAINING=False

class DQNSolver:

    def __init__(self, observation_input,action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.observation_input=observation_input
        self.action_space = action_space
        self.memory = []

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=observation_input, activation="relu"))
        self.model.add(Flatten())
        self.model.add(Dense(24, use_bias=False))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dense(action_space, use_bias=False))
        self.model.add(BatchNormalization())
        self.model.add(Activation("linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        
    def predict(self,state):
        return self.model.predict(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.predict(state)[0]
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                #new q value for this state/action pair is equal to the reward gained by taking this action at this state, plus the expected reward to be gained for the rest of the game.
                #GAMMA is a parameter relating to the short/long term planning tendencies of the model. High GAMMA means were planning ahead, low means were looking most at short term rewards.
                q_update = (reward + GAMMA * np.amax(self.predict(state_next)[0]))
            q_values = self.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        
def reshape_dims(obs_space):
    dims=[1]
    for i in range(len(obs_space.shape)):
        dims.append(obs_space.shape[i])
    return dims

def find_input_shape(env):
    input_shape=None
    #Box
    if(type(env.observation_space)==gym.spaces.box.Box):
        input_shape=env.observation_space.shape
    #Discrete
    elif(type(env.observation_space)==gym.spaces.discrete.Discrete):
        input_shape=[env.observation_space.n]
    return input_shape

class ActionSpaceError(Exception):
    pass
    
def training():
    env = gym.make(ENV_NAME)
    # If the user chooses an environment with a non-discrete action space, return an error because DQN only works with discrete action spaces
    if(type(env.action_space)!=gym.spaces.discrete.Discrete):
        raise ActionSpaceError('This environment uses an action space that is not discrete. DQN can only be trained using discrete action spaces. Please select an envionment with a discrete action space.')

    act_space=env.action_space.n
        
    score_logger = ScoreLogger(ENV_NAME)
    observation_input=find_input_shape(env)
    
    dims=reshape_dims(env.observation_space)

    dqn_solver = DQNSolver(observation_input,act_space)
    for i in range(NUM_EPISODES):
        state = env.reset()
        #reshape state array if it has more than one dimension
        if(len(dims)>1):
            state = state.reshape(dims)
        step = 0
        while True:
            step += 1
            if(WATCH_TRAINING):
                env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            #reshape state array if it has more than one dimension
            if(len(dims)>1):
                state_next = state_next.reshape(dims)
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(i+1) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, i+1)
                break
            dqn_solver.experience_replay()
    return dqn_solver

def testing(dqn_solver):
    env=gym.make(ENV_NAME)
    dims=reshape_dims(env.observation_space)
    step=0
    #set exploration rate to be 0 so that we just follow the q function's policy
    dqn_solver.exploration_rate=0
    
    state=env.reset()
    #reshape state array if it has more than one dimension
    if(len(dims)>1):
        state = state.reshape(dims)
    while True:
        step+=1
        
        env.render()
        action=dqn_solver.act(state)
        next_state,reward,terminal,info=env.step(action)
        if(terminal):
            break
        #reshape state array if it has more than one dimension
        if(len(dims)>1):
            state = next_state.reshape(dims)

if __name__ == "__main__":
    solution=training()
    testing(solution)
