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



# agent:
# initialize
# update value
# choose policy action 