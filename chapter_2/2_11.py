"""
Exercise 2.11 (programming)
Make a figure analogous to Figure 2.6 for the nonstationary
case outlined in Exercise 2.5. Include the constant-step-size epsilon-greedy algorithm with
alpha=0.1. Use runs of 200,000 steps and, as a performance measure for each algorithm and
parameter setting, use the average reward over the last 100,000 steps.
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

    def shift_true_values(self, new_values=None):
        """
        Update the underlying true bandit values, making the distributions nonstationary
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
    def __init__(self, num_bandits, bandit_instance, mode='greedy', initial_guesses=None, epsilon=None,  c=None):
        self.k = num_bandits
        self.rewards = []
        self.actions = []
        self.bandit_instance = bandit_instance
        self.mode = mode

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
        self.c = c

    def choose_action(self):
        match self.mode:
            case 'epsilon-greedy':
                use_random_choice = np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])
                if use_random_choice:
                    return np.random.choice(np.arange(self.k))
                else:
                    return np.argmax([q_a[-1] for q_a in self.qs])

            case 'greedy':
                return np.argmax([q_a[-1] for q_a in self.qs])

            case 'ucb':
                t = self.bandit_instance.iteration
                nts = [len(self.qs[i]) for i in range(len(self.qs))]
                vals = [self.qs[i][-1] for i in range(len(self.qs))]
                return np.argmax([vals[i] + self.c * np.sqrt(np.log(t) / nts[i]) for i in range(len(vals))])

            case 'gradient-bandit':
                return np.argmax([q_a[-1] for q_a in self.qs])

    def step(self, alph):
        action = self.choose_action()
        reward = self.bandit_instance.reward(action)

        match self.mode:
            case 'epsilon-greedy':
                self.qs[action].append(self.qs[action][-1] + alph * (reward - self.qs[action][-1]))
            case 'greedy':
                self.qs[action].append(self.qs[action][-1] + alph * (reward - self.qs[action][-1]))
            case 'ucb':
                self.qs[action].append(self.qs[action][-1] + alph * (reward - self.qs[action][-1]))
            case 'gradient-bandit':

                for idx_action in range(len(self.qs)):
                    if idx_action == action:
                        next_val = self.qs[idx_action][-1] + alph * (reward - self.baseline()) * (1 - self.pi(idx_action))
                    else:
                        next_val = self.qs[idx_action][-1] - alph * (reward - self.baseline()) * self.pi(idx_action)
                    self.qs[idx_action].append(next_val)

        self.rewards.append(reward)
        self.actions.append(action)
        self.bandit_instance.step()

    def pi(self, actn):
        num = np.exp(self.qs[actn][-1])
        denom = np.sum([np.exp(self.qs[k][-1]) for k in range(len(self.qs))])
        return num / denom

    def baseline(self):
        return np.mean(self.rewards)
# epsilon greedy, action choice: greedy  w/ epsilon random
# UCB, action choice: upper confidence bound, step alpha
# gradient bandit, track average reward, action choice: argmax, track & update preferences, step alpha
# greedy with optimistic initialization, step alpha


modes = ['epsilon-greedy', 'ucb', 'gradient-bandit', 'greedy-optimistic']
params = [2**i for i in range(-7, 3)]
labels = [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$',
          r'$\frac{1}{16}$', r'$\frac{1}{8}$', r'$\frac{1}{4}$',
          r'$\frac{1}{2}$', '1', '2',
          '4']

run_length = 2000
num_runs = 1000
alpha = .1
epsilon = .1
avg_rewards = np.zeros((len(modes), len(params), num_runs))

plt.close('all')
for idx_mode, mode in enumerate(modes):
    for idx_param, param in tqdm(enumerate(params), desc='Parameter Sweep', total=len(params)):
        for idx_run in range(num_runs):
            bi = BanditInstance(10)

            match mode:
                case 'epsilon-greedy':
                    bl = BanditLearner(10, bandit_instance=bi, epsilon=epsilon, mode=mode)

                case 'ucb':
                    bl = BanditLearner(10, bandit_instance=bi, mode=mode, c=param)

                case 'gradient-bandit':
                    bl = BanditLearner(10, bandit_instance=bi, mode=mode)

                case 'greedy-optimistic':
                    bl = BanditLearner(10, bandit_instance=bi, epsilon=epsilon, mode='greedy', initial_guesses=[5 for k in range(10)])

            for _ in range(run_length):
                bl.step(alpha)
                # bl.bandit_instance.shift_true_values()

            avg_rewards[idx_mode, idx_param, idx_run] = np.mean(bl.rewards[-1000:])

    plt.figure('rvp')
    plt.plot(params, avg_rewards.mean(axis=-1)[idx_mode, :], label=mode)

plt.figure('rvp')
plt.legend()
plt.xlabel("Parameter value (" + r"$\epsilon \ \alpha \ c \ Q_0$)")
plt.gca().set_xscale('log', base=2)
plt.xticks(params,labels)
plt.ylabel("Average Reward over Last 1000 Steps")
plt.show()
