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

class GridWorld:
    """
    An episode instance of Gridworld
    """
    def __init__(self, shape, goal, col_wind_speeds, king_moves=False, start=(0,0)):
        assert(len(shape) == 2)
        assert(len(goal) == 2)
        assert(len(col_wind_speeds) == shape[0])

        self.shape = shape
        self.goal = goal
        self.col_wind_speeds = col_wind_speeds
        self.king_moves = king_moves

        self.location = start
        self.trajectory = [start]



    def actions(self):
        basic = { # dict of direction : delta location
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
        assert(action in self.actions())

        next_x, next_y = self.location
        delta = self.actions()[action]
        assert(len(delta) == 2)
        next_x += delta[0]
        next_y += delta[1]

        assert(len(self.location) == 2)
        next_y += self.col_wind_speeds[self.location[0]]

        next_x = min(next_x, self.shape[0] - 1)
        next_x = max(next_x, 0)
        next_y = min(next_y, self.shape[1] - 1)
        next_y = max(next_y, 0)

        self.location = (next_x, next_y)
        self.trajectory.append(self.location)
        if self.location == self.goal:
            return 0
        else:
            return -1

    def visualize_trajectory(self):
        gs = GridSpec(8, 1)  # 2 rows, 3 columns
        fig = plt.figure('traj', figsize=(6,5.5))
        plt.clf()
        xs, ys = zip(*self.trajectory)
        grid = np.zeros(self.shape)

        for idx in range(len(xs)):
            print(idx, xs[idx], ys[idx])
            x = xs[idx]
            y = ys[idx]
            grid[x, y] = idx + 1

        ax1 = fig.add_subplot(gs[0:7,:])
        ax1.imshow(grid.T, cmap='Greys')

        ax1.set_xticks(range(0, self.shape[0]))
        ax1.set_yticks(range(0, self.shape[1]))
        ax1.set_xticklabels(range(1, self.shape[0] + 1))
        ylabels = [j for j in range(1, self.shape[1] + 1)]
        # ylabels.reverse()
        ax1.set_yticklabels(ylabels)
        plt.ylim([-1/2, self.shape[1] - 1/2])
        # ax1.invert_yaxis()
        ax1.set_aspect('auto')

        ax2 = fig.add_subplot(gs[7:,:])

        table = ax2.table([self.col_wind_speeds], bbox=[0,0, 1, 1],  cellLoc='center')
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_aspect('auto')
        table.scale(1,8)
        plt.subplots_adjust(None, None, None, None, None, None)
        plt.tight_layout()

        plt.show()
        # this.location = next_loc

        # trajectory.append(next_loc))

        # if next_loc = goal,

            #  self.is_terminal = true

            #  return (0, )
        # else:
        #       return -1
    # location

    # trajectory
    #

# agent:
# initialize
# update value
# choose policy action

# grid 7h x 10w
# location of agent
# 10w column wind speeds
# Goal location
# rewards: constant -1 for all actions, unless goal is reached 0
# if moving off of grid, location unchanged, but reward is still -1
# movement is action plus north by column wind velocity

if __name__ == '__main__':
    grid_shape = (10, 7)  # 10 wide by 7 high, bottom left is (0,0), top right is (10, 7)
    end_goal = (8, 4)
    wind_speeds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    # wind_speeds = [0 for j in wind_speeds]

    gw = GridWorld(
        shape=grid_shape, goal=end_goal, col_wind_speeds=wind_speeds,
        king_moves=True, start=(4,0))
    print(gw.step('e'), gw.location)
    print(gw.step('e'), gw.location)
    print(gw.step('e'), gw.location)
    print(gw.step('e'), gw.location)
    print(gw.step('e'), gw.location)
    gw.visualize_trajectory()
    gw.trajectory
    print("running")


