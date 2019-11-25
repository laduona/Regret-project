import gym
import gym_gridworld
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import random
import copy


class Elig_Trace(object):

    def __init__(self, env, params, nS, nA):
        """ Init Elig_Trace object

            Args:
              env: enviroment
              params: Hyperparameters for the Elig_Trace model.
              model: Model of the world.
              nS: number of states
              nA: number of actions
              rng: Random Number Generator.
        """
        self.params = params
        self.env = env
        self.action_space = env.action_space
        self.nS = nS
        self.nA = nA
        self.Q = np.zeros((nS, nA))
        self.e = np.zeros((nS, nA))
        self.new_Q = np.zeros((nS, nA))
        self.epsilon=0


    def run(self, initial_state, env, render=False):
        """Run algorithm until reach target or candy or reach max_steps

           Args:
              initial_state: Value on the start state.
              env: Enviroment.

            Return:
                step: max_steps or steps reach target
                regret_states: the states have max regret
                mx: the value of max regret
                info: reach info
                potential_regret: dict contain regret value at cooresponding states
        """

        potential_regret = {} # state-> regret
        state = initial_state
        movements = [0,1,2,3]
        step = 0
        done = False
        trace = []  # a  (state,action) pair list for  1 episode
        old_Q_in_trace= []
        states_to_sample = [] # a list of states for sampling

        regret=[]

        while step < self.params['max_steps']:
            action = self._get_action(state)
            trace.append((state,action))
            trace_back = trace[-1::-1] # latest step first in list

            obs, reward, done, info = env.step(action)  # move the agent
            next_state = self.map_position_to_number(env.agent_current_state, env.width)

            delta = self.Q_delta(state, action, reward, next_state)
            # print('delta:',delta)
            self.Q[state, action] = self.Q[state, action] + self.params['alpha'] * delta

            if render:
                self.env.render('human')

            state = next_state

            step += 1
            if done:
                break

        return step,regret,info,trace_back

    def Q_delta(self,state, action, reward, next_state):
        """ This functions calculate delta value.
              delta =  (r + \gamma * {Q(s',a')} -Q(s,a))
              a' is chosen from epsilon-greedy policy

            Args:
              state: Current state of the world.
              action: Current action chosen.
              reward: Reward for taking action in state.
              next_state: The next state to transition to.
        """

        delta = reward + self.params['gamma'] * np.max(self.Q[next_state, :]) - self.Q[state, action]
        return delta

    def sarsa_delta(self,state, action, reward, next_state,next_action):

        delta = reward + self.params['gamma'] * (self.Q[next_state, next_action]) - self.Q[state, action]
        return delta


    def sampling(self, env, state, actual_act, max_step):
        """ This funcion sampling an episode for (state, action)

        Return reward for this sampling episode and step
        """
        env._set_agent_at(state)
        step = 0
        total_reward =0
        trace = []
        moves = [0, 1, 2, 3]
        move_list = np.random.choice(moves, max_step, p=[0.25, 0.25, 0.25, 0.25])
        while step < max_step:

            if step == 0:
                obs, reward, done, info = env.step(actual_act)
            else:
                obs, reward, done, info = env.step(move_list[step])

            total_reward += reward
            step += 1
            if done:
                break

        return total_reward,trace,step


    def random_action(self):
        return self.env.action_space.sample()


    def _get_action(self, state):
        """ Choose action for state with a \epsilon-greedy policy.

            Args:
              state: Current state of the world.
        """

        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self._argmax(self.Q[state, :])
        return action


    def _argmax(self,array):
        m = np.amax(array)
        indices = np.nonzero(array == m)[0]
        return random.choice(indices)



    def reset(self):
        """ Reset the  algorithms.
            This resets the Q values.
        """
        self.Q.fill(0)

    def grid_index_to_number(self,index,width):
        x = index[0]
        y = index[1]
        return (x * width) + y


    def map_position_to_number(self, agent_current_state, width):  # in 4x4 grid, (1,2) return 6
        x = agent_current_state[0][0]
        y = agent_current_state[1][0]
        return (x * width) + y

