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
    def __init__(self, num_bandits, bandit_instance, mode='greedy', initial_guesses=None, epsilon=None, c=None):
        self.k = num_bandits
        self.rewards = []
        self.bandit_instance = bandit_instance
        self.mode = mode


        if initial_guesses:
            try:
                if len(initial_guesses) != num_bandits:
                    raise IndexError(
                        "The provided guesses (length = {0}) does not match the provided number "
                        "of bandits (length={1})".format(len(initial_guesses), num_bandits))
                else:
                    self.qs = initial_guesses
            except TypeError as te:
                raise TypeError("The provided initial guesses do not appear to have a length attribute.")
        else:
            self.qs = [0 for _ in range(num_bandits)]

        if self.mode == 'ucb':
            self.nts = [0 for _ in range(num_bandits)]

        if self.mode == 'gradient-bandit':
            self.reward_mean = 0

        self.epsilon = epsilon
        self.c = c

    def choose_action(self):
        match self.mode:
            case 'epsilon-greedy':
                use_random_choice = np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])
                if use_random_choice:
                    return np.random.choice(np.arange(self.k))
                else:
                    return np.argmax([q for q in self.qs])

            case 'greedy' | 'gradient-bandit':
                return np.argmax([q for q in self.qs])

            case 'ucb':
                t = self.bandit_instance.iteration

                vals = [self.qs[i] for i in range(len(self.qs))]

                sqrt_terms = [np.sqrt(np.log(t) / self.nts[i])
                              if (self.nts[i] > 0) else np.inf
                              for i in range(len(vals))]

                return np.argmax([vals[i] + self.c * sqrt_terms[i] for i in range(len(vals))])

    def step(self, alph):
        action = self.choose_action()
        reward = self.bandit_instance.reward(action)

        match self.mode:
            case 'epsilon-greedy' | 'greedy':
                self.qs[action] = self.qs[action] + alph * (reward - self.qs[action])

            case 'ucb':
                self.qs[action] = self.qs[action] + alph * (reward - self.qs[action])
                self.nts[action] += 1

            case 'gradient-bandit':
                # compute action probabilities
                num = np.exp(self.qs)
                denom = np.sum(np.exp(self.qs))
                pis = num / denom

                # update gradient
                qs_update = self.qs - alph * (reward - self.reward_mean) * pis
                qs_update[action] = self.qs[action] + alpha * (reward - self.reward_mean) * (1 - pis[action])
                self.qs = qs_update

                # update baseline rewards average
                if len(self.rewards) > 0:
                    self.reward_mean = self.reward_mean + (reward - self.reward_mean) / len(self.rewards)
                else:
                    self.reward_mean = reward

        self.rewards.append(reward)
        self.bandit_instance.step()

# epsilon greedy, action choice: greedy  w/ epsilon random
# UCB, action choice: upper confidence bound, step alpha
# gradient bandit, track average reward, action choice: argmax, track & update preferences, step alpha
# greedy with optimistic initialization, step alpha


# num_runs = 2000
# run_length = 1000
# alpha = .1
# epsilon = .1
# mode = 'Constant Step-Size'
# rewards = np.zeros((num_runs, run_length))
# opt_actions = np.zeros((num_runs, run_length))
# for i in tqdm(range(num_runs), desc='Performing Runs', total=num_runs):
#     bi = BanditInstance(10)
#     bl = BanditLearner(10, bandit_instance=bi, epsilon=epsilon, mode='epsilon-greedy')
#     for j in range(run_length):
#         bl.step(alpha)
#         # bl.bandit_instance.shift_true_values()
#         opt_action = bi.optimal_action
#         opt_actions[i, j] = np.asarray(bl.actions[-1]) == opt_action

#     rewards[i, :] = bl.rewards


# plt.figure('poa')
# plt.plot(opt_actions.mean(axis=0), label=mode)
# plt.legend()
# plt.xlabel("Step")
# plt.ylabel("% Optimal Action")
#
# plt.figure('r')
# plt.plot(rewards.mean(axis=0), label=mode)
# plt.legend()
# plt.xlabel("Step")
# plt.ylabel("Average Reward")
# plt.show()


modes = ['epsilon-greedy', 'ucb', 'gradient-bandit', 'greedy-optimistic']
# modes = ['gradient-bandit']
params = [2 ** i for i in range(-7, 3)]
labels = [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$',
          r'$\frac{1}{16}$', r'$\frac{1}{8}$', r'$\frac{1}{4}$',
          r'$\frac{1}{2}$', '1', '2',
          '4']
plot_ranges = {
    'epsilon-greedy': (0,6),
    'gradient-bandit': (2,-1),
    'ucb': (3,-1),
    'greedy-optimistic': (5, -1)
}
run_length = 200000
num_runs = 10
alpha = .1
eps = .1
avg_rewards = np.zeros((len(modes), len(params), num_runs))

plt.close('all')
for idx_mode, mde in enumerate(modes):
    for idx_param, param in tqdm(enumerate(params), desc='Parameter Sweep', total=len(params)):
        for idx_run in range(num_runs):
            bi = BanditInstance(10)

            match mde:
                case 'epsilon-greedy':
                    if param >= 1:
                        break
                    bl = BanditLearner(10, bandit_instance=bi, epsilon=param, mode=mde)

                case 'ucb':
                    bl = BanditLearner(10, bandit_instance=bi, mode=mde, c=param)

                case 'gradient-bandit':
                    bl = BanditLearner(10, bandit_instance=bi, mode=mde)

                case 'greedy-optimistic':
                    bl = BanditLearner(10, bandit_instance=bi, mode='greedy', initial_guesses=[param for k in range(10)])

            for _ in range(run_length):
                match mde:
                    case 'epsilon-greedy' | 'ucb' | 'greedy-optimistic':
                        bl.step(alpha)
                    case 'gradient-bandit':
                        bl.step(param)
                bl.bandit_instance.shift_true_values()

            avg_rewards[idx_mode, idx_param, idx_run] = np.mean(bl.rewards[-100000:])

    plt.figure('rvp')
    plt.gca().set_xscale('log', base=2)
    plt.xticks(params, labels)
    start, stop = plot_ranges[mde]
    plt.plot(params[start:stop], avg_rewards.mean(axis=-1)[idx_mode, start:stop], label=mde)

plt.figure('rvp')
plt.legend()
plt.xlabel("Parameter value (" + r"$\epsilon \ \alpha \ c \ Q_0$)")
plt.ylabel("Average Reward over Last 1000 Steps")
plt.gca().set_xscale('log', base=2)
plt.xticks(params, labels)
plt.show()
