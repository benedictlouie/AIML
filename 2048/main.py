import numpy as np
import math
import random


N = int(input("Size? "))

def move(s1: np.array, a: int):
    
    s2 = np.copy(s1)

    # shift
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
    
    # combine
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
    
    # shift
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
    
    empty = []
    if not (s1 == s2).all():
      # new tile
      for i in range(N*N):
        if s2.flatten()[i] == 0:
          empty.append(i)
      z = np.random.choice(empty, 1)
      s2[math.floor(z/N)][z%N] = 1
    return s2


def hash(s: np.array):
    arr = []
    for i in range(N): arr.append(tuple(s[i]))
    return tuple(arr)

Q = {}

def R(s1: np.array, a: int, s2: np.array):
    return 2 ** s2.max() - 2 ** s1.max()

def VPolicy(s: np.array):
    m, ma = 0, random.randrange(0, 4)
    arr = [0, 1, 2, 3]
    random.shuffle(arr)
    for a in arr:
        h = hash(s)
        if h not in Q: Q[h] = np.zeros(4)
        if m < Q[h][a]:
            m = Q[h][a]
            ma = a
    return m, ma

gamma = 0.9
alpha = 0.5
epsilon = 0.7

# number of trials
trials = int(input("Number of trials? "))

for i in range(trials):

    # new game
    new = np.zeros([N,N])
    x = np.random.choice(N*N, 2, replace=False)
    new[math.floor(x[0]/N)][x[0]%N] = 1
    new[math.floor(x[1]/N)][x[1]%N] = 1

    # start game
    state = new
    while True:

        if state[state != 0].size == N*N:
            h = hash(state)
            if hash(move(state, 0)) == hash(move(state, 1)) == hash(move(state, 2)) == hash(move(state, 3)) == h:
                for action in [0, 1, 2, 3]: Q[h][action] -= (2**N - 1)
                break

        action = random.randrange(0, 4)
        if random.random() >= epsilon: _, action = VPolicy(state)

        h = hash(state)
        nextState = move(state, action)
        if h == hash(nextState): continue
        if h not in Q: Q[h] = np.zeros(4)
        V, policy = VPolicy(nextState)
        Q[h][action] = (1-alpha) * Q[h][action] + alpha * (R(state, action, nextState) + gamma * V)
        state = nextState


# use policy
state = new
while True:

    print(state)
    if state[state != 0].size == N*N:
        h = hash(state)
        if h not in Q: Q[h] = np.zeros(4)
        if hash(move(state, 0)) == hash(move(state, 1)) == hash(move(state, 2)) == hash(move(state, 3)) == h:
            for action in [0, 1, 2, 3]: Q[h][action] -= (2**N - 1)
            break
    _, action = VPolicy(state)
    print(Q[hash(state)])
    print(action)

    nextState = move(state, action)
    state = nextState
