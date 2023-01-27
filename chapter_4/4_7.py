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
import math

import numpy as np
import os
import pickle
from numba import jit

# Parameters
from tqdm import tqdm

RATE_RENTAL_1 = 3
RATE_RENTAL_2 = 4
RATE_RETURN_1 = 3
RATE_RETURN_2 = 2

MAX_NUM_CARS = 20
MAX_CARS_MOVE = 5
REWARD_PER_RENTAL = 10

DISCOUNT = 0.9
POLICY_EVAL_THRESH = 1e-6


psrsa_cache_file = 'psrsa_cache.pkl'
compute = False


@jit(nopython=True, cache=True)
def charToInt(char):
    """
    Very dumb implementation for possible numbers of cars moved because numba doesn't support casting str to int.
    """
    if char == '0':
        return 0
    elif char == '1':
        return 1
    elif char == '2':
        return 2
    elif char == '3':
        return 3
    elif char == '4':
        return 4
    elif char == '5':
        return 5


@jit(nopython=True, cache=True)
def factorial(x):
    if x == 1: # base case
        return 1
    else: # recurse
        return x * factorial(x - 1)


@jit(nopython=True, cache=True)
def poisson(rate, n):
    """
    Poisson Probability Distribution
    :param rate: Arrival Rate of Poisson Process (aka Lambda)
    :param n: Number of Arrivals (in this case requests or returns)
    :return: probability of n arrivals at given rate, float in  [0, 1]
    """
    return np.float_power(rate, n) * np.exp(-rate) / factorial(n)


@jit(nopython=True, cache=True)
def cars_moved_in(action, loc):
    """
    Return the number of cars moved into a location by a given action. If a location loses cars, this returns negative.
    :param action: action string described below;
    :param loc: string/char of 1 or 2, indicating location 1 or two respectively.
    :return: Number of cars entering loc. If cars are leaving, this number is negative
    """
    if action == '0xx':
        return 0

    num_cars_moved = charToInt(action[0])
    if (action[1] == '1') and (action[2] == '2'):  # cars moving from location 1 to location 2
        return -num_cars_moved if (loc == '1') else num_cars_moved

    elif (action[1] == '2') and (action[2] == '1'):  # cars moving from location 2 to location 1
        return num_cars_moved if (loc == '1') else -num_cars_moved


@jit(nopython=True, cache=True)
def psrsa(next_state, reward, curr_state, action):
    """
    MDP Dynamics Probability for the Car Rental Problem
    :param next_state: tuple, The next state (num cars at locations 1 and 2) reached after taking the given action
    :param reward: The scalar reward earned by reaching next state
    :param curr_state: tuple, the current state (num cars at locations 1 and 2)
    :param action: action string described below; the action currently taken to reach next state
    :return: float, the probability of reaching next_state and gaining reward given current state and action taken
    """
    num_cars_moved = charToInt(action[0])

    # compute total number of rentals from reward and action: # reward = $10 * total_rentals - 2 * num_cars_moved
    total_rentals = (reward + (2 * num_cars_moved)) // 10

    total_prob = 0

    for rentals_from_1 in range(total_rentals + 1):
        rentals_from_2 = total_rentals - rentals_from_1

        returns_to_1 = next_state[0] - curr_state[0] + rentals_from_1 - cars_moved_in(action, loc=1)
        returns_to_2 = next_state[1] - curr_state[1] + rentals_from_2 - cars_moved_in(action, loc=2)
        if (returns_to_1 < 0) or (returns_to_2 < 0):
            continue

        total_prob += (
                poisson(RATE_RETURN_1, returns_to_1) * poisson(RATE_RENTAL_1, rentals_from_1)
                *
                poisson(RATE_RETURN_2, returns_to_2) * poisson(RATE_RENTAL_2, rentals_from_2)
        )
    return total_prob


@jit(nopython=True, cache=True)
def possible_future_states(state, states, action, reward):
    # compute possible future states given a current state, action, and reward.
    # the total number of cars can decrease by at most [total_rentals = (reward + (2 * num_cars_moved)) // 10]
    total_num_cars = state[0] + state[1]
    num_cars_moved = charToInt(action[0])
    total_rentals = (reward + (2 * num_cars_moved)) // 10

    return [s for s in states if (s[0] + s[1]) >= total_num_cars - total_rentals]

if compute:
    # region  Initialization

    # state space is that either location can have anywhere from [0,... 20] cars.
    states = [(cars_loc_1, cars_loc_2)
              for cars_loc_1 in range(MAX_NUM_CARS + 1)
              for cars_loc_2 in range(MAX_NUM_CARS + 1)]

    values = np.random.random(size=len(states)) - 0.5

    '''
    Possible actions are denoted by a 3-digit string, where the leftmost digit
    represents the number of cars moved, the middle digit represents the source
    location and the third digit represents the destination location. E.g.
    121 = move one car from location 2  to location 1
    212 = move 2 cars from location 1 to location 2
    0xx = move no cars (do nothing)
    
    There is only 1 action for doing nothing giving 10 total actions. 
    '''
    possible_actions = (
            [str(i) + '21' for i in range(1, MAX_CARS_MOVE + 1)]
            +
            [str(i) + '12' for i in range(1, MAX_CARS_MOVE + 1)]
            +
            ['0xx']
    )

    # actions are to move up to five cars from one location to other, not exceeding
    # 20 cars at either location
    actions = {
        state: [pa for pa in possible_actions if
                (pa == '0xx')  # case 3: do nothing always allowed
                or
                (
                        (int(pa[1]) == 1 and int(pa[2]) == 2)  # case 1, moving from loc 1 to 2
                        and
                        state[1] + int(pa[0]) <= MAX_NUM_CARS  # loc 2 has no more than 20 as a result
                        and
                        state[0] - int(pa[0]) >= 0  # loc 1 has zero or more as a result
                )
                or
                (
                        (int(pa[1]) == 2 and int(pa[2]) == 1)  # case 2: moving from loc 2 to 1
                        and
                        state[0] + int(pa[0]) <= MAX_NUM_CARS  # loc 1 has no more than 20 as a result
                        and
                        state[1] - int(pa[0]) >= 0  # loc 2 has zero or more as a result
                )
                ]
        for state in states
    }

    policy = {state: actions[state][0] for state in states}  # arbitrary choice of action as initial policy
    policies = [policy]
    values = {state: np.random.random() for state in states}
    values['terminal'] = 0
    # endregion

    # while not policy stable
    policy_stable = False
    while not policy_stable:

        # region Evaluate Policy
        # use caching for quick lookup of dynamics function;
        if os.path.exists(psrsa_cache_file):
            with open(psrsa_cache_file, 'rb') as f:
                psrsa_cache = pickle.load(f)
        else:
            psrsa_cache = {}

        while True:
            delta = 0
            for state in tqdm(states, total=len(states), desc='Evaluating States'):
                v_curr = values[state]
                running_sum = 0
                action = policy[state]
                update_cache = False
                num_cars = state[0] + state[1]
                possible_rewards = [10 * num_rentals - 2 * int(action[0]) for num_rentals in range(num_cars + 1)]
                for r in possible_rewards:
                    for next_state in possible_future_states(state, states, action,
                                                             r):  # all states are possible future states
                        if (next_state, r, state, action) in psrsa_cache:
                            running_sum += psrsa_cache[(next_state, r, state, action)]
                        else:
                            prob = psrsa(next_state, r, state, action) * (r + DISCOUNT * values[next_state])
                            running_sum += prob
                            psrsa_cache[(next_state, r, state, action)] = prob
                            update_cache = True

                values[state] = running_sum
                delta = np.max([delta, np.abs(v_curr - values[state])])

            if update_cache:
                with open(psrsa_cache_file, 'wb') as f:
                    pickle.dump(psrsa_cache, f)

            if delta < POLICY_EVAL_THRESH:
                break
        # endregion

        # region Improve Policy
        policy_stable = True
        update_cache = False
        for state in tqdm(states, total=len(states), desc='Improving Policy'):
            old_action = policy[state]

            num_cars = state[0] + state[1]
            action_values = []
            for action in actions[state]:
                possible_rewards = [10 * num_rentals - 2 * int(action[0]) for num_rentals in range(num_cars + 1)]
                running_sum = 0
                for r in possible_rewards:
                    for next_state in possible_future_states(state, states, action, r):
                        if (next_state, r, state, action) in psrsa_cache:
                            running_sum += psrsa_cache[(next_state, r, state, action)]
                        else:
                            prob = psrsa(next_state, r, state, action) * (r + DISCOUNT * values[next_state])
                            running_sum += prob
                            psrsa_cache[(next_state, r, state, action)] = prob
                            update_cache = True
                action_values.append(running_sum)

            policy[state] = actions[state][np.argmax(action_values)]
            if old_action != policy[state]:
                policy_stable = False

        if update_cache:
            with open(psrsa_cache_file, 'wb') as f:
                pickle.dump(psrsa_cache, f)

        policies.append(policy)
        # endregion

    # improvement loop

    with open('policies.pkl', 'wb') as f:
        pickle.dump(policies, f)

    with open('value.pkl', 'wb') as f:
        pickle.dump(values, f)

else:
    # assuming the following files are saved
    with open('policies.pkl', 'rb') as f:
        policies = pickle.load(f)

    with open('value.pkl', 'rb') as f:
        value = pickle.load(f)

