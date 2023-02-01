"""
Exercise 4.7 (programming) Write a program for policy iteration and re-solve Jack’s car
rental problem with the following changes. One of Jack’s employees at the first location
rides a bus home each night and lives near the second location. She is happy to shuttle
one car to the second location for free. Each additional car still costs $2, as do all cars
moved in the other direction. In addition, Jack has limited parking space at each location.
If more than 10 cars are kept overnight at a location (after any moving of cars), then an
additional cost of $4 must be incurred to use a second parking lot (independent of how
many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often
occur in real problems and cannot easily be handled by optimization methods other than
dynamic programming. To check your program, first replicate the results given for the
original problem.
"""
import functools
from copy import copy

import numpy as np
from multiprocessing import Pool
from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy

# Parameters
WIN_DOLLARS = 100
DISCOUNT = 0.9
POLICY_EVAL_THRESH = 1e-6
PROB_HEADS = .4


def expected_return(state, action, vals):
    # get next possible states given current state and action
    next_states = np.asarray([state + action, state - action])

    # get their respective probabilities
    joint_prob = np.asarray([PROB_HEADS, 1 - PROB_HEADS])
    joint_prob /= joint_prob.sum()

    # get their rewards
    rewards = np.asarray([0, 0])
    if next_states[0] == WIN_DOLLARS:
        rewards[0] = 1

    # get the vals of the next states
    next_vals = np.asarray([vals[s] for s in next_states])

    # compute the update
    val_updates = joint_prob * (rewards + DISCOUNT * next_vals)

    return val_updates.sum()


def initialize():
    # state space is the possible dollar balance
    states = np.arange(100)

    actions = {
        state: list(range(np.min([state, 100 - state]) + 1))
        for state in states
    }

    policy = {state: actions[state][0] for state in states}  # arbitrary choice of action as initial policy
    policies = [policy]
    values = {state: np.random.random() for state in states}

    values[0] = 0
    values[100] = 1

    return states, values, actions, policy, policies


def plot_values(vals, label=None):
    plt.plot(np.arange(WIN_DOLLARS), [vals[s] for s in np.arange(WIN_DOLLARS)], label=label)
    plt.xlabel('Capital')
    plt.ylabel('Value')
    plt.title("Values")


def plot_policy(plcy, label=None):
    plt.scatter(np.arange(WIN_DOLLARS), [plcy[s] for s in np.arange(WIN_DOLLARS)], label=label)
    plt.xlabel('Capital')
    plt.ylabel('Bet')
    plt.title("Policy")

def highest_max_val_idx(arr):
    max_val = np.asarray(arr).max()
    max_idx = -1
    for idx, val in enumerate(arr):
        if val >= max_val:
            max_idx = idx
    return max_idx

def iterate(state, actions, values):

    v_curr = copy(values[state])
    action_values = [expected_return(state, action, values) for action in actions[state]]

    policy_update = actions[state][highest_max_val_idx(action_values)]
    value_update = np.max(action_values)
    return policy_update, value_update, np.abs(v_curr - value_update), state


if __name__ == '__main__':
    states, values, actions, policy, policy_sets = initialize()

    values_sets = [values]

    print('Not Converged, Iterating Value...')
    while True:
        with Pool(processes=8) as p:
            policy_updates, value_updates, deltas, eval_states = \
                zip(*p.map(functools.partial(iterate, actions=actions, values=values), states))

        for idx, s in enumerate(eval_states):
            values[s] = value_updates[idx]
            policy[s] = policy_updates[idx]

        values_sets.append(copy(values))
        policy_sets.append(copy(policy))

        if np.max(deltas) < POLICY_EVAL_THRESH:
            break
        else:
            print("delta = ", np.max(deltas))

    idx_plot = [1, 20, 30, len(policy_sets) - 1 ]
    for i in idx_plot:
        plt.figure('policy ')
        plot_policy(policy_sets[i], label='Sweep {0}'.format(i))
        plt.legend()

        plt.figure('vals ')
        plot_values(values_sets[i], label='Sweep {0}'.format(i))


    plt.figure('vals ')
    plt.legend()
    plt.figure('policy ')
    plt.legend()
    plt.show()
