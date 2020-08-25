import numpy as np


def evaluate(env, q_table):
    total_epochs, total_penalties = 0, 0
    episodes = 100

    for i in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        done = False
        if i % 10 == 0:
            print("Episode : " + str(i))

        while not done:
            if i % 10 == 0:
                env.render()

            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1
        if i % 10 == 0:
            env.render()

        total_epochs += epochs
        total_penalties += penalties

    return episodes, total_penalties, total_epochs
