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
import numpy as np
from multiprocessing import Pool
from scipy.stats import poisson
import matplotlib.pyplot as plt
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
POLICY_EVAL_THRESH = 1


def poisson(arr, rate):
    """
    Compute the Poisson PMF of the given array elementwise with given rate. The PMF gives the probability of n-arrivals
    over a unit of time.
    Poisson PMF is  rate^n * exp(-rate) / n!
    :param arr: n-dim numpy array specifying number of arrivals
    :param rate: scalar specifying number of arrivals per unit time.
    :return: n-dim numpy array with same shape as arr, whose elements are the respective probabilities.
    """
    return np.float_power(np.broadcast_to(rate, arr.shape), arr) * np.exp(-rate) / scipy.special.factorial(arr)


def state_arrays_to_values(tuples_1, tuples_2, tuple_keyed_dict):
    """
    Use tuples drawn from two n-dim numpy arrays as  keys for a dictionary.
    :param tuples_1: n-dim numpy array of integers; each element is the first part of a tuple-key;
    :param tuples_2: n-dim numpy array of integers; each element is the second part of a tuple-key;
    :param tuple_keyed_dict: dictionary whose keys are tuple integers; should contain every key specified by tuples_[12]
    :return: values: n-dim array with same shape as tuples_[12] containing  elements of  tuple_keyed_dict
    """
    states_1_flat = tuples_1.flatten()
    states_2_flat = tuples_2.flatten()
    values_flat = np.asarray([tuple_keyed_dict[state] for state in zip(states_1_flat, states_2_flat)])
    return values_flat.reshape(tuples_1.shape)


def cars_moved(action):
    """
    Compute car flows INTO each location (negative if flowing outward)
    :param action: action string (3-string indication number and direction of car flow)
    :return: cars_moved_1, cars_moved_2: number of cars flowing INTO each respective location
    """
    cars_moved_1 = -int(action[0]) if action[1] == '1' and action[2] == '2' else int(action[0])
    cars_moved_2 = int(action[0]) if action[1] == '1' and action[2] == '2' else -int(action[0])

    for cm in [cars_moved_1, cars_moved_2]:
        assert (abs(cm) <= 5), "Failure in sanity check |cars moved| <= 5: cm= {0}".format(cm)
    assert (cars_moved_1 == - cars_moved_2), \
        "Failure in sanity check cars_moved_1 = -cars_moved_2: {0}!=-{1}".format(cars_moved_1, cars_moved_2)

    return cars_moved_1, cars_moved_2


def possible_rentals_and_returns(car_dynamics, state):
    """
    Enumerate the possible rentals and returns from the current state given the flow of cars.
    :param car_dynamics: Tuple giving net flow of cars into locations 1 and 2
    :param state: current number of cars at each location
    :return: (rentals_1, rentals_2, returns_1, returns_2): 4-tuple containing all possible rentals and returns as 4-d
    numpy arrays; each axis corresponds to varying one of rentals or returns at one location
    """

    # Possible rentals can't exceed number of cars rentable tomorrow
    possible_rentals_1 = np.arange(state[0] + car_dynamics[0] + 1)
    possible_rentals_2 = np.arange(state[1] + car_dynamics[1] + 1)
    assert (np.all(possible_rentals_1 >= 0)), "Negative number of rentals at location 1: {0}".format(possible_rentals_1)
    assert (np.all(possible_rentals_2 >= 0)), "Negative number of rentals at location 2: {0}".format(possible_rentals_2)

    # Possible number of returns capped at max fleet size (2 * MAX_NUM_CARS), no strict conservation conditions here tho
    possible_returns_1 = np.arange(MAX_NUM_CARS + 1)
    possible_returns_2 = np.arange(MAX_NUM_CARS + 1)

    # Meshgrid to vectorize computation
    rentals_1, returns_1, rentals_2, returns_2 = np.meshgrid(
        possible_rentals_1, possible_returns_1,
        possible_rentals_2, possible_returns_2
    )
    return rentals_1, rentals_2, returns_1, returns_2


def next_possible_states(state, car_dynamics, rr_dynamics):
    """
    Given the current state and action taken, compute all the possible next states reachable. Returns a 3-tuple:
     each element is a 4-d numpy array, where a different part of state varies along each axis; e.g. returns at location
     1 vary along the first axis, returns at location 1 vary along the second etc. The first two tuples are the next
     states at each location  and the third is the respective joint probability of reaching those states.

    :param state: tuple specifying the current state, i.e. tuple with number of cars at locations 1 and 2
    :param car_dynamics: 2-tuple containing net flow of cars into locations 1 and 2
    :param rr_dynamics: 4-tuple giving rentals and returns at locations 1 and 2 (output of possible_rentals_and_returns)
    :return: (next_states_rentals_returns_1, next_states_rentals_returns_2, probs)
     tuple containing all future possible states at each location reachable from the present state with the given
     action,and the joint probability of reaching those states
    """
    rentals_1, rentals_2, returns_1, returns_2 = rr_dynamics
    cars_moved_1, cars_moved_2 = car_dynamics

    next_states_1 = state[0] - rentals_1 + returns_1 + cars_moved_1
    next_states_2 = state[1] - rentals_2 + returns_2 + cars_moved_2

    # clip states to MAX_NUM_CARS
    next_states_1[next_states_1 > MAX_NUM_CARS] = MAX_NUM_CARS
    next_states_2[next_states_2 > MAX_NUM_CARS] = MAX_NUM_CARS
    assert(np.all(next_states_1 >= 0)), "Negative next_state_1: ".format(next_states_1.min())
    assert(np.all(next_states_2 >= 0)), "Negative next_state_2: ".format(next_states_2.min())

    probs_1 = poisson(rentals_1, RATE_RENTAL_1) * poisson(returns_1, RATE_RETURN_1)
    probs_2 = poisson(rentals_2, RATE_RENTAL_2) * poisson(returns_2, RATE_RETURN_2)

    joint_prob = probs_1 * probs_2
    joint_prob /= joint_prob.sum()

    return next_states_1, next_states_2, joint_prob


def expected_return(state, action, values):
    car_dynamics = cars_moved(action)

    rr_dynamics = possible_rentals_and_returns(car_dynamics, state)

    next_states_1, next_states_2, joint_prob = next_possible_states(state, car_dynamics, rr_dynamics)

    rewards = REWARD_PER_RENTAL * (rr_dynamics[0] + rr_dynamics[1]) - (2 * int(action[0]))

    next_vals = state_arrays_to_values(next_states_1, next_states_2, values)

    val_updates = joint_prob * (rewards + DISCOUNT * next_vals)

    return val_updates.sum()


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
        [-int(a[0]) if a[1] == '1' else int(a[0]) for a in vals.flatten()],
        xx.shape
    )
    plt.imshow(vals.T)
    plt.gca().invert_yaxis()
    plt.xlabel("Cars at Location 2")
    plt.ylabel("Cars at Location 1")
    plt.title("Policy: Cars Moved From Location 1 to Location 2")
    plt.colorbar()


def evaluate(state, values, policy):
    v_curr = values[state]
    action = policy[state]
    er = expected_return(state, action, values)
    return er, np.max([0, np.abs(v_curr - er)]), state


def improve(state, actions, values):
    action_values = [expected_return(state, action, values) for action in actions[state]]
    policy_update = actions[state][np.argmax(action_values)]

    return policy_update, state


if __name__ == '__main__':
    states, values, actions, policy, policies = initialize()

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
                zip(*p.map(functools.partial(improve, actions=actions, values=values), states))

        policy_stable = True
        for idx, s in enumerate(states):
            if policy[s] != policy_updates[idx]:
                policy_stable = False
                policy[s] = policy_updates[idx]



        # policy_stable = True
        # for state in tqdm(states, total=len(states), desc='Improving Policy'):
        #     old_action = policy[state]
        #     action_values = [expected_return(state, action, tuple_keyed_dict) for action in actions[state]]
        #     policy[state] = actions[state][np.argmax(action_values)]
        #     if old_action != policy[state]:
        #         policy_stable = False

        print('appended policy, policy_stable = ',policy_stable)
        policies.append(policy.copy())

    print("leaving: policy_stable = ", policy_stable)

    for i, pol in enumerate(policies):

        plt.figure('policy ' + str(i))
        plot_policy(pol)

        plt.figure('tuple_keyed_dict ' + str(i))
        plot_values(values_sets[i])
    plt.show()
    #     # improvement loop
    #
    #     with open('policies.pkl', 'wb') as f:
    #         pickle.dump(policies, f)
    #
    #     with open('value.pkl', 'wb') as f:
    #         pickle.dump(tuple_keyed_dict, f)
    #
    # else:
    #     # assuming the following files are saved
    #     with open('policies.pkl', 'rb') as f:
    #         policies = pickle.load(f)
    #
    #     with open('value.pkl', 'rb') as f:
    #         value = pickle.load(f)
    #
