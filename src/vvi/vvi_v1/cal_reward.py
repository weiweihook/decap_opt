"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces
import numpy as np
import IPython
import os
import math

NCOL, NROW = 5, 5
# non-capacitor grids: 13
intp_mim = 108
chip_mos = 36
NCAP = intp_mim + chip_mos



def loadtxtmethod(filename):
    data = np.loadtxt(filename, dtype=np.float32)
    return data

def cal_reward(filename):

    val = loadtxtmethod(filename)
    intp_cap_val = 0
    chip_cap_val = 0
    for i in range(intp_mim):
        intp_cap_val += val[i]
    for j in range(intp_mim, NCAP):
        chip_cap_val += val[j]
    total_cost = 0.5 * (108 * 2000 - intp_cap_val) / (108 * 2000) + 0.5 * ((36 * 2000 - chip_cap_val) / (36 * 2000))
    print(intp_cap_val, chip_cap_val*0.25)

    return total_cost

if __name__ == "__main__":
    file = 'runs0905/case3/allocation.txt'
    reward = cal_reward(file)
    print(reward)
