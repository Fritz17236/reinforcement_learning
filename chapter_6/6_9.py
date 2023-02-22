"""
Re-solve the windy
gridworld assuming eight possible actions, including the diagonal moves, rather than four.
How much better can you do with the extra actions? Can you do even better by including
a ninth action that causes no movement at all other than that caused by the wind?
"""
# environment
# grid 7h x 10w
# location of agent
# 10w column wind speeds
# Goal location
# rewards: constant -1 for all actions, unless goal is reached 0
# if moving off of grid, location unchanged, but reward is still -1
# movement is action plus north by column wind velocity
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


class GridWorld:
    """
    An episode instance of Gridworld
    """

    def __init__(self, shape, goal, col_wind_speeds, king_moves=False, start=(0, 0)):
        assert (len(shape) == 2)
        assert (len(goal) == 2)
        assert (len(col_wind_speeds) == shape[0])

        self.shape = shape
        self.goal = goal
        self.col_wind_speeds = col_wind_speeds
        self.king_moves = king_moves

        self.location = start
        self.trajectory = [start]
        self.terminal_location = False

    def actions(self):
        basic = {  # dict of direction : delta location
            'n': (0, 1),
            's': (0, -1),
            'e': (1, 0),
            'w': (-1, 0)
        }
        king = {
            'nw': (-1, 1),
            'ne': (1, 1),
            'sw': (-1, -1),
            'se': (1, -1)
        }

        king.update(basic)
        if self.king_moves:
            return king
        else:
            return basic

    def step(self, action):
        """
        Receive actions, move according to wind conditions, return reward. If terminal state
        :param self:
        :param action: Desired movement; must be one of Gridworld.actions()
        :return: reward, 0 if at reward, -1 otherwise.
        """
        assert (action in self.actions())

        next_x, next_y = self.location
        delta = self.actions()[action]
        assert (len(delta) == 2)
        next_x += delta[0]
        next_y += delta[1]

        assert (len(self.location) == 2)
        next_y += self.col_wind_speeds[self.location[0]]

        next_x = min(next_x, self.shape[0] - 1)
        next_x = max(next_x, 0)
        next_y = min(next_y, self.shape[1] - 1)
        next_y = max(next_y, 0)

        self.location = (next_x, next_y)
        self.trajectory.append(self.location)
        if self.location == self.goal:
            self.terminal_location = True
            return 0
        else:
            return -1

    def visualize_trajectory(self):
        gs = GridSpec(8, 1)  # 2 rows, 3 columns
        fig = plt.figure('traj', figsize=(6, 5.5))
        plt.clf()
        xs, ys = zip(*self.trajectory)
        grid = np.zeros(self.shape)

        for idx in range(len(xs)):
            x = xs[idx]
            y = ys[idx]
            grid[x, y] = idx + 1

        ax1 = fig.add_subplot(gs[0:7, :])
        ax1.imshow(grid.T, cmap='Greys')

        ax1.set_xticks(range(0, self.shape[0]))
        ax1.set_yticks(range(0, self.shape[1]))
        ax1.set_xticklabels(range(1, self.shape[0] + 1))
        ylabels = [j for j in range(1, self.shape[1] + 1)]
        # ylabels.reverse()
        ax1.set_yticklabels(ylabels)
        plt.ylim([-1 / 2, self.shape[1] - 1 / 2])
        # ax1.invert_yaxis()
        ax1.set_aspect('auto')

        ax2 = fig.add_subplot(gs[7:, :])

        table = ax2.table([self.col_wind_speeds], bbox=[0, 0, 1, 1], cellLoc='center')
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_aspect('auto')
        table.scale(1, 8)
        plt.subplots_adjust(None, None, None, None, None, None)
        plt.tight_layout()

        plt.show()
        # this.location = next_loc

    def reset(self):
        self.location = self.trajectory[0]
        self.trajectory = [self.location]
        self.terminal_location = False


class EpsilonSarsaLearner:
    """
    Epsilon-Greedy Sarsa Learner
    """
    def __init__(self, gridworld: GridWorld, epsilon: float, alpha: float, gamma: float):
        self._gw = gridworld
        self._state_space = [(i,j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1])]
        self._action_space = gridworld.actions()
        self.q = {(state, action): 0 for state in self._state_space for action in self._action_space }
        self.episodes = []
        self.time_steps = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def loop_episode(self):
        # initialize episode

        start_time_step = self.time_steps
        self._gw.reset()
        state = self._gw.location
        action = self.choose_action(state)
        cum_reward = 0
        while not self._gw.terminal_location:
            reward = self._gw.step(action)
            next_state = self._gw.location
            next_action = self.choose_action(next_state)
            self.q[state, action] = self.q[state, action] + self.alpha * (
                reward + self.gamma * self.q[next_state, next_action] - self.q[state, action]
            )
            state = next_state
            action = next_action
            cum_reward += reward
            self.time_steps += 1

        self.episodes.append({'cumulative_reward': cum_reward, 'time_steps': self.time_steps - start_time_step})
            # start_time_step = self.time_step
            # reset gridworld
            # get gridworld state
            # choose action from state using Q (epsilon-greedy)
        # then for each step of episode while S is not terminal
            # take action, observe r, s'
            # choose a' from s' using Q (epsilon-greedy)
            # q(s,a) <-- q(s,a) + alpha * [r + gamma * q(s', a') - q(s,a)]
            # s <-- s'
            # a <-- a'

            # self.time_steps += 1

        # after reaching terminal, summarize episode statistics and plot
        # episode = {
        # cumulative_reward: sum(rewards)
        # steps: self.time_step - start_timestep
        # }
        # self.episodes.append(episode)

    def choose_action(self, state):
        """
        Choose an action based off of the state-action value estimate Q, with epsilon-greedy exploration
        :param state: tuple location in gridworld
        :return: selected action, one of Gridworld.actions()
        """
        actions = list(self._gw.actions().keys())
        explore = np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])

        if explore:
            return np.random.choice(actions)
        else:
            vals = [self.q[(state, action)] for action in actions]
            return actions[np.argmax(vals)]
        return 1

    def plot_completed_episodes(self):
        """ Plot the number of episodes completed vs. the time step from learner initialization """
        episode_count = range(1, len(self.episodes) + 1)
        time_steps = [ep['time_steps'] for ep in self.episodes]
        plt.figure('evt')
        plt.clf()
        plt.plot(np.cumsum(time_steps), episode_count)
        plt.xlabel("Time Step")
        plt.ylabel("Episode Number")
        plt.show()


# agent:
# initialize
# update value
# choose policy action

if __name__ == '__main__':
    grid_shape = (10, 7)  # 10 wide by 7 high, bottom left is (0,0), top right is (10, 7)
    end_goal = (8, 4)
    wind_speeds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    epsilon = .1
    alpha = .5
    gamma = 1
    gw = GridWorld(
        shape=grid_shape, goal=end_goal, col_wind_speeds=wind_speeds,
        king_moves=False, start=(0, 0))
    # print(gw.step('e'), gw.location)
    # print(gw.step('e'), gw.location)
    # print(gw.step('e'), gw.location)
    # print(gw.step('e'), gw.location)
    # print(gw.step('e'), gw.location)
    # gw.visualize_trajectory()

    learner = EpsilonSarsaLearner(gridworld=gw, epsilon=epsilon, alpha=alpha, gamma=1)
    for j in tqdm(range(170)):
        learner.loop_episode()
    learner.plot_completed_episodes()

    # while not gw.terminal_location:
    #     action = np.random.choice(list(gw.actions().keys()))
    #     gw.step(action)
    #
    # gw.visualize_trajectory()
    # print("running")
