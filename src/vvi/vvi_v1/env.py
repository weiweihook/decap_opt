"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces
import numpy as np
import IPython
import os
import math
import config
import subprocess
import concurrent.futures

NCOL, NROW = 11, 11
cons = 0.0
maxmos = 700
execute_file = ['interposer_tr_temp.sp']
cmd = ['bin/inttrvmap intp_chip1.conf vdi.raw 0.9 0.05  > /dev/null 2> /dev/null',
       'bin/inttrvmap intp_chip2.conf vdi.raw 0.9 0.05  > /dev/null 2> /dev/null',
       'bin/inttrvmap intp_chip3.conf vdi.raw 0.9 0.05  > /dev/null 2> /dev/null',
       'bin/inttrvmap intp_chip4.conf vdi.raw 0.9 0.05  > /dev/null 2> /dev/null']
vvi_files = ['chiplet1_vdd_1_vdi.csv', 'chiplet2_vdd_1_vdi.csv', 'chiplet3_vdd_1_vdi.csv', 'chiplet4_vdd_1_vdi.csv']

def run_command(commend):
    return subprocess.run(commend, shell=True, stdout=subprocess.PIPE, text=True)

def run_os(path):
    original_path = os.getcwd()
    os.chdir(path)
    os.system('ngspice  -b interposer_tr_temp.sp -r vdi.raw > /dev/null 2> /dev/null')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(run_command, cmd)
    os.chdir(original_path)

def readvvi(file):
    zvvi = readresult(file)
    z = zvvi[:, 2]
    return z

def readresult(filename):
    a1 = np.genfromtxt(filename)
    return a1

def gen_mask(intp_n, chip_n):
    intp_mask = np.ones(NCOL*NROW)
    chip_mask = np.zeros(NCOL*NROW)
    for i in range(len(intp_n)):
        intp_mask[intp_n[i]] = 0
    for j in range(len(chip_n)):
        chip_mask[chip_n[j]] = 1
    intp_mask = intp_mask.reshape(NCOL, NROW)
    chip_mask = chip_mask.reshape(NCOL, NROW)
    return intp_mask, chip_mask

def fill_non_zero(mask, cur_param):
    non_zeros_indices = np.nonzero(mask)
    dis = np.copy(mask)
    for i, j, k in zip(non_zeros_indices[0], non_zeros_indices[1], cur_param):
        dis[i][j] = k

    return dis

class DecapPlaceParallel(gym.Env):

    action_meaning = [0, -50, 50]  # index: 0,1,2
    action_space = np.array([len(action_meaning)] * 36)
    action_space_shape = (36,)
    single_action_space = (36,)
    observation_space_shape = (1,  4, 7, 7)
    single_observation_space_shape = (4, 7, 7)
    env_count = 1

    def __init__(self, env_config):

        self.path = 'config/case' + str(env_config) + '/'
        intp_mim, chip_mos, NCAP, intp_n, chip_n, init_vvi = config.read_config(self.path + 'case' + str(env_config) + '.conf')
        self.intp_mim = intp_mim
        self.chip_mos = chip_mos
        self.NCAP = NCAP
        self.intp_n = intp_n
        self.chip_n = chip_n
        self.init_curr_params_idx = readresult(self.path+'init_chip_param.txt')
        self.cur_params_idx = np.zeros_like(self.init_curr_params_idx)
        self.init_vvi = init_vvi

    def reset(self):
        self.cur_params_idx = np.zeros_like(self.init_curr_params_idx)
        obs, init_reward = self.cal_reward(self.cur_params_idx)
        return obs

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 2 mapped to the index of the corresponding parameter
        :return: obs, reward, done, VVI, add mos capacitance, violation node number
        """

        # Take action that RL agent returns to change current params
        action = list(action.reshape(-1))
        change = np.array([self.action_meaning[a] for a in action])
        self.cur_params_idx = self.cur_params_idx + change
               
        obs, reward = self.cal_reward(self.cur_params_idx)
      
        done = [True if np.sum(action) == 0 else False]
        add_mos = np.sum(self.cur_params_idx)
        vio_node = np.count_nonzero(obs)

        return obs, reward, done, np.sum(obs), add_mos, vio_node

    def cal_reward(self, cur_param_idx):

        str_dc = ''
        esrs = ''
        for j, val in enumerate(cur_param_idx + self.init_curr_params_idx):
            str_dc += '.param dcap_int_val%d=%dp\n' % (j + 1 + self.intp_mim, val)
            if val != 0:
                esrs += '.param esr%d=%.3f\n' % (j + 1, 50 / val)
            else:
                esrs += '.param esr%d=0\n' % (j + 1)

        f = open(self.path + 'chip_param_dcap.txt', 'w')
        f.write(str_dc)
        f.close()

        f1 = open(self.path + 'moscap_esr.txt', 'w')
        f1.write(esrs)
        f1.close()

        run_os(self.path)

        vvi_chip1 = readvvi(self.path + 'chiplet1_vdd_1_vdi.csv').reshape(7, 7)
        vvi_chip2 = readvvi(self.path + 'chiplet2_vdd_1_vdi.csv').reshape(7, 7)
        vvi_chip3 = readvvi(self.path + 'chiplet3_vdd_1_vdi.csv').reshape(7, 7)
        vvi_chip4 = readvvi(self.path + 'chiplet4_vdd_1_vdi.csv').reshape(7, 7)

        obs = np.array([vvi_chip1, vvi_chip2, vvi_chip3, vvi_chip4])

        if np.sum(obs) / self.init_vvi > cons:
            reward = 1 - np.sum(obs) / self.init_vvi
        else:
            reward = 2 - cons - np.sum(cur_param_idx) / (maxmos * self.chip_mos)

        return obs, reward

    def action_mask(self):
        mask = np.ones((36, len(self.action_meaning)))
        for i in range(36):
            if self.cur_params_idx[i] == 0:
                mask[i][1] = 0
            elif self.cur_params_idx[i] >= maxmos:
                mask[i][2] = 0
        action_mask = mask.reshape(-1)

        return action_mask


if __name__ == "__main__":
    env = DecapPlaceParallel(1)
    o = env.reset()
    o, r, d, v, m, n = env.step(np.array([0]*36))
    print(r,d,m,n)


