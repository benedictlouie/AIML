import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from collections import deque
import random
import multiprocessing

N = 3

# Build the DQN model
def build_model():
    state_input = Input(shape=(N*N,))
    action_input = Input(shape=(1,))
    x = concatenate([state_input, action_input])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='linear')(x)
    model = tf.keras.models.Model(inputs=[state_input, action_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# DQN agent
class DQNAgent:
    def __init__(self):
        self.state_size = N*N
        self.action_size = 4
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.5
        self.epsilon_decay = 0.99
        self.model = build_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = [self.model.predict([state, np.array([[a]])], verbose=0)[0][0] for a in range(self.action_size)]
        return np.argmax(q_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([item[0] for item in minibatch])
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.array([item[3] for item in minibatch])
        dones = np.array([item[4] for item in minibatch])

        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                next_q_values = [self.model.predict([next_states[i], np.array([[a]])], verbose=0)[0][0] for a in range(self.action_size)]
                target += self.gamma * np.amax(next_q_values)
            target_f = self.model.predict([states[i], np.array([[actions[i]]])], verbose=0)
            target_f[0][0] = target
            self.model.fit([states[i], np.array([[actions[i]]])], target_f, epochs=1, verbose=0)

def shift(s2: np.array, a: int):

    for _ in range(N-1):
        for i2 in range(N):
            for j2 in range(N-1):
                i2 -= 1
                i = i2
                j = j2
                ki = 0
                kj = 1
                if a == 1:
                    j = N-1-j2
                    kj = -1
                elif a == 2:
                    i = j2
                    j = i2
                    ki = 1
                    kj = 0
                elif a == 3:
                    i = N-1-j2
                    j = i2
                    ki = -1
                    kj = 0
                if (s2[i][j] == 0 and s2[i+ki][j+kj] != 0):
                    s2[i][j] = s2[i+ki][j+kj]
                    s2[i+ki][j+kj] = 0

def combine(s2: np.array, a: int):
    for i2 in range(N):
        for j2 in range(N-1):
            i = i2
            j = j2
            ki = 0
            kj = 1
            if a == 1:
                j = N-1-j2
                kj = -1
            elif a == 2:
                i = j2
                j = i2
                ki = 1
                kj = 0
            elif a == 3:
                i = N-1-j2
                j = i2
                ki = -1
                kj = 0
            if (s2[i][j] != 0 and s2[i][j] == s2[i+ki][j+kj]):
                s2[i][j] += 1
                s2[i+ki][j+kj] = 0

def move(s1: np.array, a: int):
    
    # a: 0=left, 1=right, 2=up, 3=down

    s2 = np.copy(s1) # state 2, copy of state 1
    shift(s2, a)
    combine(s2, a)
    shift(s2, a)
    empty = []
    if not (s1 == s2).all():
      # new tile
      for i in range(N*N):
        if s2.flatten()[i] == 0: empty.append(i)
      z = np.random.choice(empty, 1)
      z = z[0] # get the only element in the array
      s2[z//N][z%N] = 1
    return s2

def newGame() -> np.array:
    new = np.zeros([N,N])
    x = np.random.choice(N*N, 2, replace=False)
    new[x[0]//N][x[0]%N] = 1
    new[x[1]//N][x[1]%N] = 1
    return new

def gameEnds(state: np.array) -> bool:
    if state[state != 0].size == N*N:
        if (move(state, 0) == state).all() and (move(state, 1) == state).all() and (move(state, 2) == state).all() and (move(state, 3) == state).all():
            return True
    return False

# Simulate the environment (functions need to be defined)
def nextState(state, action):
    # Define how the next state is obtained given the current state and action
    s1 = np.reshape(state, (N,N))
    s2 = move(s1, action)
    return np.reshape(s2, (1,N*N))

def reward(state, action):
    # Define the reward function for a given state and action
    s1 = np.reshape(state, (N,N))
    s2 = move(s1, action)
    if gameEnds(s2): return -2**(10-s1.max())
    return 2**s2.max() - 2**s1.max() - 0.1
    
def run_episode(agent, episode):
    state = newGame()
    state = np.reshape(state, (1, N*N))
    accumulated_rewards = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state = nextState(state, action)
        reward_val = reward(state, action)
        done = gameEnds(np.reshape(next_state, (N, N)))
        agent.remember(state, action, reward_val, next_state, done)

        if not (state == next_state).all():
            pass

        state = next_state

        accumulated_rewards += reward_val

        # print(['<','>','^','v'][action], reward_val)
        # print(np.reshape(state, (N, N)))

        if done:
            print(np.reshape(state, (N, N)))
            print(f"Episode: {episode+1}, Score: {accumulated_rewards:.2f}, Epsilon: {agent.epsilon:.2f}")
            break

    return accumulated_rewards


# Training the DQN
if __name__ == "__main__":
    
    agent = DQNAgent()
    agent.model.load_weights('dqn.weights.h5')
    agent.epsilon = 0.1
    episodes = 1000
    batch_size = 10

    for e in range(episodes):
        
        run_episode(agent, e)

        # if len(agent.memory) > batch_size:
        #     agent.replay(batch_size)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    agent.model.save_weights('dqn.weights.h5')