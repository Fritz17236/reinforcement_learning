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
from functools import cache
from itertools import repeat

import numpy as np
from multiprocessing import Pool
from scipy.stats import poisson
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy

# Parameters

RATE_RENTAL_1 = 3
RATE_RENTAL_2 = 4
RATE_RETURN_1 = 3
RATE_RETURN_2 = 2

MAX_NUM_CARS = 20
MAX_CARS_MOVE = 5
REWARD_PER_RENTAL = 10

DISCOUNT = 0.9
POLICY_EVAL_THRESH = 6.5

compute = True


def poisson(arr, rate):
    return np.float_power(np.broadcast_to(rate, arr.shape), arr) * np.exp(-rate) / scipy.special.factorial(arr)


def state_arrays_to_values(states_1, states_2, values):
    states_1_flat = states_1.flatten()
    states_2_flat = states_2.flatten()
    values_flat = np.asarray([values[state] for state in zip(states_1_flat, states_2_flat)])
    return values_flat.reshape(states_1.shape)


def expected_return(state, action, values):
    cars_moved_1 = -int(action[0]) if action[1] == '1' and action[2] == '2' else int(action[0])
    cars_moved_2 = int(action[0]) if action[1] == '1' and action[2] == '2' else -int(action[0])

    possible_rentals_1 = np.arange(min(state[0] - cars_moved_2, MAX_NUM_CARS) + 1)
    possible_returns_1 = np.arange(MAX_NUM_CARS + 1)

    possible_rentals_2 = np.arange(min(state[1] - cars_moved_1, MAX_NUM_CARS) + 1)
    possible_returns_2 = np.arange(MAX_NUM_CARS + 1)

    rentals_1, returns_1, rentals_2, returns_2 = np.meshgrid(
        possible_rentals_1, possible_returns_1,
        possible_rentals_2, possible_returns_2
    )

    prob_rentals_1 = poisson(rentals_1, RATE_RENTAL_1)
    prob_returns_1 = poisson(returns_1, RATE_RETURN_1)
    prob_rentals_2 = poisson(rentals_2, RATE_RENTAL_2)
    prob_returns_2 = poisson(returns_2, RATE_RETURN_2)

    next_states_rentals_returns_1 = state[0] - rentals_1 + returns_1 + cars_moved_1
    next_states_rentals_returns_2 = state[1] - rentals_2 + returns_2 + cars_moved_2

    next_states_rentals_returns_1[next_states_rentals_returns_1 > MAX_NUM_CARS] = MAX_NUM_CARS
    next_states_rentals_returns_2[next_states_rentals_returns_2 > MAX_NUM_CARS] = MAX_NUM_CARS

    rewards = REWARD_PER_RENTAL * (rentals_1 + rentals_2) - (2 * int(action[0]))

    next_vals = state_arrays_to_values(next_states_rentals_returns_1, next_states_rentals_returns_2, values)

    val_updates = (prob_rentals_1 * prob_returns_1 * prob_rentals_2 * prob_returns_2) * \
                  (rewards + DISCOUNT * next_vals)

    out = np.sum(val_updates)

    return out


def initialize():
    # state space is that either location can have anywhere from [0,... 20] cars.
    states = [(cars_loc_1, cars_loc_2)
              for cars_loc_1 in range(MAX_NUM_CARS + 1)
              for cars_loc_2 in range(MAX_NUM_CARS + 1)]

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
    values = {state: np.random.random() + 300 for state in states}
    values['terminal'] = 0

    return states, values, actions, policy, policies


def plot_values(values):
    x = np.arange(MAX_NUM_CARS + 1)
    y = np.arange(MAX_NUM_CARS + 1)
    xx, yy = np.meshgrid(x, y)
    vals = state_arrays_to_values(xx, yy, values)
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, vals)


def plot_policy(policy):
    x = np.arange(MAX_NUM_CARS + 1)
    y = np.arange(MAX_NUM_CARS + 1)
    xx, yy = np.meshgrid(x, y)
    vals = state_arrays_to_values(xx, yy, policy)
    vals = np.reshape(
        [int(a[0]) if a[1] == '1' else -int(a[0]) for a in vals.flatten()],
        xx.shape
    )
    plt.imshow(vals)
    plt.gca().invert_yaxis()
    plt.xlabel("Cars at Location 2")
    plt.ylabel("Cars at Location 1")
    plt.title("Policy: Cars Moved Into Location 1")
    plt.colorbar()


# if True:
states, values, actions, policy, policies = initialize()


def evaluate(state, values, policy):
    v_curr = values[state]
    action = policy[state]
    er = expected_return(state, action, values)
    return er, np.max([0, np.abs(v_curr - er)]), state


def improve(state, actions):
    action_values = [expected_return(state, action, values) for action in actions[state]]
    policy_update = actions[state][np.argmax(action_values)]

    return policy_update, state


if __name__ == '__main__':

    policy_stable = False
    values_sets = [values]

    while not policy_stable:

        print('Not Policy Stable, Evaluating Policy...')
        eval_loop_count = 0
        while True:
            with Pool(processes=8) as p:
                expected_returns, deltas, eval_states = \
                    zip(*p.map(functools.partial(evaluate, values=values, policy=policy), states))

            for idx, s in enumerate(eval_states):
                values[s] = expected_returns[idx]

            values_sets.append(values)

            if np.max(deltas) < POLICY_EVAL_THRESH:
                break
            else:
                print("delta = ", np.max(deltas))
                eval_loop_count += 1

        print("Improving Policy...")
        if eval_loop_count == 0:
            break

        with Pool(processes=8) as p:
            policy_updates, eval_states = \
                zip(*p.map(functools.partial(improve, actions=actions), states))

        policy_stable = True
        for idx, s in enumerate(states):
            if policy[s] != policy_updates[idx]:
                policy_stable = False
                policy[s] = policy_updates[idx]



        # policy_stable = True
        # for state in tqdm(states, total=len(states), desc='Improving Policy'):
        #     old_action = policy[state]
        #     action_values = [expected_return(state, action, values) for action in actions[state]]
        #     policy[state] = actions[state][np.argmax(action_values)]
        #     if old_action != policy[state]:
        #         policy_stable = False

        print('appended policy, policy_stable = ',policy_stable)
        policies.append(policy.copy())

    print("leaving: policy_stable = ", policy_stable)

    for i, pol in enumerate(policies):

        plt.figure('policy ' + str(i))
        plot_policy(pol)

        plt.figure('values ' + str(i))
        plot_values(values_sets[i])
    plt.show()
    #     # improvement loop
    #
    #     with open('policies.pkl', 'wb') as f:
    #         pickle.dump(policies, f)
    #
    #     with open('value.pkl', 'wb') as f:
    #         pickle.dump(values, f)
    #
    # else:
    #     # assuming the following files are saved
    #     with open('policies.pkl', 'rb') as f:
    #         policies = pickle.load(f)
    #
    #     with open('value.pkl', 'rb') as f:
    #         value = pickle.load(f)
    #
