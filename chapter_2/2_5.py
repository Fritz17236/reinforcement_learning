"""
Exercise 2.5 (programming)
Design and conduct an experiment to demonstrate the
difficulties that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the q*(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean 0
and standard deviation 0.01 to all the q*(a) on each step). Prepare plots like Figure 2.2
for an action-value method using sample averages, incrementally computed, and another
action-value method using a constant step-size parameter, alpha = 0.1. Use epsilon = 0.1 and
longer runs, say of 10,000 steps.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class BanditInstance:
    """
    Class for Single Instance of k-armed bandit problem.

    Params:
    num_bandits: number of possible actions to take
    """

    def __init__(self, num_bandits, seed=None):
        if seed:
            np.random.seed(seed)
        self.q_stars = [np.random.normal(loc=0, scale=1, size=num_bandits)]  # true reward values
        self.optimal_action = np.argmax(self.q_stars[-1])
        self.iteration = 0
        self.shift_iters = []

    def reward(self, action):
        """
        :param action: Index (zero based) of action to select
        :return: Reward value of action selected
        :rtype: int
        """
        return np.random.normal(loc=self.q_stars[-1][action], scale=1)

    def shift_true_values(self, new_values):
        """
        Update the underlying true bandit values, making the distributions nonstationary
        :param new_values:
        :return: None
        """
        self.q_stars.append(new_values)
        self.shift_iters.append(self.iteration)

    def step(self):
        """
        Step bandit instance to next timestep
        :return: None
        """
        self.iteration += 1


class BanditLearner:
    def __init__(self, num_bandits, bandit_instance, initial_guesses=None, epsilon=None):
        self.k = num_bandits
        self.alphas = []
        self.rewards = []
        self.actions = []
        self.bandit_instance = bandit_instance

        if initial_guesses:
            try:
                if len(initial_guesses) != num_bandits:
                    raise IndexError(
                        "The provided guesses (length = {0}) does not match the provided number of bandits (length={1})".format(
                            len(initial_guesses), num_bandits))
                else:
                    self.qs = [[guess] for guess in initial_guesses]
            except TypeError as te:
                raise TypeError("The provided initial guesses do not appear to have a length attribute.")
        else:
            self.qs = [[0] for _ in range(num_bandits)]

        self.epsilon = epsilon

    def choose_action(self):
        if self.epsilon:
            use_random_choice = np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])
            if use_random_choice:
                return np.random.choice(np.arange(self.k))
            else:
                return np.argmax([q_a[-1] for q_a in self.qs])
        else:
            # choose optimal action
            return np.argmax([q_a[-1] for q_a in self.qs])

    def step(self, alpha):
        action = self.choose_action()
        reward = self.bandit_instance.reward(action)
        self.rewards.append(reward)
        self.actions.append(action)

        if alpha.lower() == 'average':
            alpha = 1 / len(self.qs[action])
        else:
            raise NotImplementedError("alpha = {0} mode not implemented".format(alpha))

        self.qs[action].append(self.qs[action][-1] + alpha * (reward - self.qs[action][-1]))
        self.alphas.append(alpha)

        self.bandit_instance.step()


num_runs = 2000
run_length = 1000
for eps in [None, .01, .1]:
    rewards = np.zeros((num_runs, run_length))
    opt_actions = np.zeros((num_runs, run_length))
    for i in tqdm(range(num_runs), desc='Performing Runs', total=num_runs):
        bi = BanditInstance(10)
        bl = BanditLearner(10, bandit_instance=bi, epsilon=eps)
        for j in range(1, run_length + 1):
            bl.step('average')

        rewards[i, :] = bl.rewards
        opt_action = bi.optimal_action
        opt_actions[i, :] = bl.actions == opt_action

    plt.plot(opt_actions.mean(axis=0), label=r'$\epsilon$={0}'.format(eps))
plt.legend()
plt.show()
