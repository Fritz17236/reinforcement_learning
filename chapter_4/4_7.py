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
import numpy as np

# Parameters
RATE_REQUEST_1 = 3
RATE_REQUEST_2 = 4
RATE_RETURN_1 = 3
RATE_RETURN_2 = 2

MAX_NUM_CARS = 20
MAX_CARS_MOVE = 5

DISCOUNT = 0.9


def poisson(rate, n):
    """
    Poisson Probability Distribution
    :param rate: Arrival Rate of Poisson Process (aka Lambda)
    :param n: Number of Arrivals (in this case requests or returns)
    :return: probability of n arrivals at given rate, float in  [0, 1]
    """
    return np.float_power(rate, n) * np.exp(-rate) / np.math.factorial(n)


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

values = {state: np.random.random() for state in states}
values['terminal'] = 0
# endregion

# while not policy stable

# evaluation loop

# improvement loop

