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
        self.q_stars = [np.random.normal(loc=0, scale=1, size=num_bandits)]  # true reward tuple_keyed_dict
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

    def shift_true_values(self, new_values=None):
        """
        Update the underlying true bandit tuple_keyed_dict, making the distributions nonstationary
        :param new_values:
        :return: None
        """
        if not new_values:
            new_values = self.q_stars[-1] + np.random.normal(loc=0, scale=.01, size=len(self.q_stars[-1]))
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

    def step(self, alph):
        action = self.choose_action()
        reward = self.bandit_instance.reward(action)
        try:
            if alph.lower() == 'average':
                alph = 1 / len(self.qs[action])
            else:
                raise NotImplementedError("alpha = {0} mode not implemented".format(alph))
        except AttributeError:
            pass  # ignore attribute error if numerical alpha provided

        self.qs[action].append(self.qs[action][-1] + alph * (reward - self.qs[action][-1]))
        self.alphas.append(alph)
        self.rewards.append(reward)
        self.actions.append(action)
        self.bandit_instance.step()


num_runs = 2000
run_length = 1000
alpha = .1
epsilon = .1
for mode in ['Sample Averages', 'Constant Step-Size']:
    rewards = np.zeros((num_runs, run_length))
    opt_actions = np.zeros((num_runs, run_length))
    for i in tqdm(range(num_runs), desc='Performing Runs', total=num_runs):
        bi = BanditInstance(10)
        bl = BanditLearner(10, bandit_instance=bi, epsilon=epsilon)
        for j in range(run_length):
            if mode == "Sample Averages":
                bl.step('average')
            elif mode == "Constant Step-Size":
                bl.step(alpha)

            bl.bandit_instance.shift_true_values()
            opt_action = bi.optimal_action
            opt_actions[i, j] = np.asarray(bl.actions[-1]) == opt_action

        rewards[i, :] = bl.rewards


    plt.figure('poa')
    plt.plot(opt_actions.mean(axis=0), label=mode)

    plt.figure('r')
    plt.plot(rewards.mean(axis=0), label=mode)

plt.figure('poa')
plt.legend()
plt.xlabel("Step")
plt.ylabel("% Optimal Action")

plt.figure('r')
plt.legend()
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.show()
