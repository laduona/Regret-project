import gym
import gym_gridworld
from standard_Q_learning import Elig_Trace
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import pickle

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
target_reward_after = -30 # 29 for small change, 10 for medium change and -30 for large change


initial_epsilon = 0.9
render_animation = True

nS = np.prod([env.height,env.width]) # number of states
nA = len(env.actions) # number of actions

eTrace = Elig_Trace(env,params,nS,nA)
eTrace.epsilon= initial_epsilon

episode_regret = []
reach_before = {'target': 0, 'candy': 0}
reach_after = {'target': 0, 'candy': 0}
target_episode=[]
candy_episode=[]
Trace =[]

for episode in range(params['episodes']):
    print('***********************************************  episode',episode)
    if episode < params['episodes']-2:
        eTrace.epsilon = initial_epsilon**(episode)
        obs = env.reset()
        initial_state = eTrace.map_position_to_number(env.agent_current_state, env.width)
        steps, regret, info,trace_back = eTrace.run(initial_state, env, render=render_animation)
        if info['Reach'] == 'target':
            reach_before['target']+=1
            target_episode.append(episode)
        elif info['Reach'] == 'candy':
            reach_before['candy']+=1
            candy_episode.append(episode)
        print('Q at 25 before change:', eTrace.Q[25])

    elif episode == params['episodes']-2:
        eTrace.epsilon = 0
        obs = env.reset()
        initial_state = eTrace.map_position_to_number(env.agent_current_state, env.width)
        steps, regret, info,trace_back = eTrace.run(initial_state, env, render=render_animation)
        if info['Reach'] == 'target':
            reach_before['target'] += 1
            target_episode.append(episode)
        elif info['Reach'] == 'candy':
            reach_before['candy'] += 1
            candy_episode.append(episode)
        print('Q at 25 before change:', eTrace.Q[25])
    else:
        eTrace.epsilon=0
        obs = env.reset()
        env.change_target_reward(target_reward_after)
        initial_state = eTrace.map_position_to_number(env.agent_current_state, env.width)
        steps, regret, info, trace_back = eTrace.run(initial_state, env, render=render_animation)
        Trace = trace_back
        if info['Reach'] == 'target':
            reach_after['target']+=1
            target_episode.append(episode)
        elif info['Reach'] == 'candy':
            reach_after['candy']+=1
            candy_episode.append(episode)
        print('Q at 25 after change:', eTrace.Q[25])
        print('Q at 34 after change:', eTrace.Q[34])
        print('Q at 19 after change:', eTrace.Q[19])

    print("*** Episode: ", episode)
    print('epsilon:', eTrace.epsilon)
    print('steps:', steps)
    print('Reach:',info['Reach'])


print('****Before change:',reach_before)
print('****After change:',reach_after)


# # Q table for large target change, converge to target and change reward of target 30 -> -30
# with open("QTable_converge_target_30_-30.pkl", 'wb') as f:
#     pickle.dump(eTrace.Q, f)

# # Q table for medium target change, converge to target and change reward of target 30 -> 10
# with open("QTable_converge_target_30_10.pkl", 'wb') as f:
#     pickle.dump(eTrace.Q, f)

# # Q table for small target change, converge to target and change reward of target 30 -> 29
# with open("QTable_converge_target_30_29.pkl", 'wb') as f:
#     pickle.dump(eTrace.Q, f)

