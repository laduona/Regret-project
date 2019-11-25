import gym
import gym_gridworld
from standard_Q_learning import Elig_Trace
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import pickle


# The reverse trace of (state,act) from start position to target
Trace = [(25, 0), (34, 0), (43, 0), (42, 3), (41, 3), (40, 3), (49, 0), (58, 0), (67, 0)]

target_reward_after = -30  # large target reward change
with open("QTable_converge_target_30_-30.pkl", 'rb') as f:
    Q_table = pickle.load(f)

# target_reward_after = 29  # small target reward change
# with open("QTable_converge_target_30_29.pkl", 'rb') as f:
#     Q_table = pickle.load(f)


# target_reward_after = 10  # medium target reward change
# with open("QTable_converge_target_30_10.pkl", 'rb') as f:
#     Q_table = pickle.load(f)

env = gym.make('Gridworld-v0')

height = env.height
width = env.width

params = {
    'gamma': 0.9,
    'alpha': 0.9, # learning rate
    'lambda':0.9,
    'episodes': 50,
    'runs': 20,
    'max_steps': 150
}


print('--------------------------------------------------')
print('Evaluate by Sarsa')
print('trace back:',Trace)
state_regret = []
Q = copy.deepcopy(Q_table)
env.change_target_reward(target_reward_after)

for tra in Trace:
    state = tra[0]
    action = tra[1]
    env._reset()
    env._set_agent_at(state)
    next_state, r = env.simulate_step(state, action)
    if next_state == 10 or next_state == 16:
        delta = r - Q[state,action]
        Q[state, action] = Q[state, action] + params['alpha'] * delta
    else:
        index = Trace.index(tra)
        next_action = Trace[index-1][1]
        delta = r + params['gamma']*Q[next_state,next_action] - Q[state,action]
        Q[state, action] = Q[state, action] + params['alpha'] * delta

for tra in Trace:
    state = tra[0]
    action = tra[1]
    regret = np.max(Q[state,:]) - Q[state,action]
    state_regret.append(('state:'+str(state),regret))

print('Regret after sarsa:', state_regret)




print('--------------------------------------------------')
print('Evaluate by QTrace')
print('trace back:',Trace)
state_regret_QL = []
Q_QL = copy.deepcopy(Q_table)
env.change_target_reward(target_reward_after)

for tra in Trace:
    state = tra[0]
    action = tra[1]
    env._reset()
    env._set_agent_at(state)
    next_state, r = env.simulate_step(state, action)
    delta = r + params['gamma']* np.max(Q_QL[next_state,:])  - Q_QL[state,action]
    Q_QL[state,action] = Q_QL[state,action] + params['alpha'] * delta


for tra in Trace:
    state = tra[0]
    action = tra[1]
    regret = np.max(Q_QL[state,:]) - Q_QL[state,action]
    state_regret_QL.append(('state:'+str(state),regret))

print('Regret after QTrace:', state_regret_QL)




print('------------------------------------------------')
print('Evaluate regret by MC average sampling')

nS = np.prod([env.height,env.width]) # number of states
nA = len(env.actions) # number of actions
sampling_times = 500
max_steps_sampling = 80
movements = [0, 1, 2, 3]
Q_for_sampling = np.zeros((nS, nA))
regret_in_sampling = []
eTrace = Elig_Trace(env,params,nS,nA)

for s_a in Trace:
    state = s_a[0]
    actual_act = s_a[1]
    for move in movements:
        R = 0
        for _ in range(sampling_times):
            env.reset()
            env.change_target_reward(target_reward_after)
            reward, trace,step = eTrace.sampling(env, state, move, max_steps_sampling)
            R += reward
        average_R = R / sampling_times
        Q_for_sampling[state,move] = average_R

    regret = np.max(Q_for_sampling[state,:]) - Q_for_sampling[state,actual_act]
    regret_in_sampling.append(('state:'+ str(state),regret))

print('Regret after sampling:', regret_in_sampling)

