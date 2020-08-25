import gym
import numpy as np
import pandas as pd
import evaluate

env = gym.make('Taxi-v3').env

env.render()

state = env.encode(3, 1, 2, 0)
print("State:", state)

env.s = state
env.render()

df = pd.read_json('data.json', orient='split')
q_table = df.rename_axis('ID').values

episodes, total_penalties, total_epochs = evaluate.evaluate(env, q_table)

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")