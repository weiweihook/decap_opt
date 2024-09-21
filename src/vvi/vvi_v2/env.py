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
gamma = 0.0
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
    action_space = np.array([len(action_meaning)] * NCOL * NROW * 2)
    action_space_shape = (NCOL * NROW * 2,)
    single_action_space = (NCOL * NROW * 2,)
    observation_space_shape = (1, 4, NCOL, NROW)
    single_observation_space_shape = (4, NCOL, NROW)
    env_count = 1

    def __init__(self, env_config):

        self.path = 'config/case' + str(env_config) + '/'
        intp_mim, chip_mos, NCAP, intp_n, chip_n, init_vvi = config.read_config(self.path + 'case' + str(env_config) + '.conf')
        self.intp_mim = intp_mim
        self.chip_mos = chip_mos
        self.NCAP = NCAP
        self.intp_n = intp_n
        self.chip_n = chip_n
        intp_mask, chip_mask = gen_mask(self.intp_n, self.chip_n)
        self.intp_mask = intp_mask
        self.chip_mask = chip_mask
        mask = list(np.concatenate([self.intp_mask, self.chip_mask]).reshape(-1))
        self.mask = mask
        self.init_curr_params_idx = readresult(self.path+'init_param_dcap.txt') #
        self.cur_params_idx = np.zeros_like(self.init_curr_params_idx)
        self.init_vvi = init_vvi

    def reset(self):
        self.cur_params_idx = np.zeros_like(self.init_curr_params_idx)
        vvi_dis, init_reward = self.cal_reward(self.cur_params_idx)
        state = np.array([self.intp_mask, np.zeros((NCOL, NROW)), self.chip_mask, np.zeros((NCOL, NROW))])
        return state

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 2 mapped to the index of the corresponding parameter
        :return: state, reward, done, vvi, add mos capacitance, violation node number
        """

        # Take action that RL agent returns to change current params
        action = list(action.reshape(-1))
        avail_action = [action[i] for i in range(len(action)) if self.mask[i] == 1]
        change = np.array([self.action_meaning[a] for a in avail_action])
        self.cur_params_idx = self.cur_params_idx + change
               
        vvi_dis, reward = self.cal_reward(self.cur_params_idx)
      
        done = [True if np.sum(avail_action) == 0 else False]
        add_mos = np.sum(self.cur_params_idx)
        vio_node = np.count_nonzero(vvi_dis)

        MIM_dis, MOS_dis = self.gen_dis()
        state = np.array([self.intp_mask, MIM_dis, self.chip_mask, MOS_dis])

        return state, reward, done, np.sum(vvi_dis), add_mos, vio_node

    def cal_reward(self, cur_param_idx):
        """
        :param cur_param_idx: the grid capacitance (0~500)
        :return reward
        """
        str_dc = ''
        esrs = ''
        for i, val in enumerate((cur_param_idx + self.init_curr_params_idx)[:self.intp_mim]):
            str_dc += '.param dcap_int_val%d=%dp\n' % (i + 1, val)

        for j, val in enumerate((cur_param_idx + self.init_curr_params_idx)[self.intp_mim:]):
            str_dc += '.param dcap_int_val%d=%dp\n' % (j + 1 + self.intp_mim, int(val))
            if val != 0:
                esrs += '.param esr%d=%.3f\n' % (j + 1, 50 / val)
            else:
                esrs += '.param esr%d=0\n' % (j + 1)

        f = open(self.path + 'int_param_dcap.txt', 'w')
        f.write(str_dc)
        f.close()

        f1 = open(self.path + 'moscap_esr.txt', 'w')
        f1.write(esrs)
        f1.close()

        run_os(self.path)

        vvi_chip1 = readvvi(self.path + 'chiplet1_vdd_1_vdi.csv')
        vvi_chip2 = readvvi(self.path + 'chiplet2_vdd_1_vdi.csv')
        vvi_chip3 = readvvi(self.path + 'chiplet3_vdd_1_vdi.csv')
        vvi_chip4 = readvvi(self.path + 'chiplet4_vdd_1_vdi.csv')

        vvi_dis = np.array([vvi_chip1, vvi_chip2, vvi_chip3, vvi_chip4]).reshape(-1)

        if np.sum(vvi_dis) / self.init_vvi > gamma:
            reward = 1 - np.sum(vvi_dis) / self.init_vvi
        else:
            reward = 2 - gamma - np.sum(cur_param_idx) / (maxmos * self.chip_mos)

        return vvi_dis, reward

    def gen_dis(self):
        cur_params_idx = self.cur_params_idx + self.init_curr_params_idx
        intp_cap = cur_params_idx[:self.intp_mim]
        chip_cap = cur_params_idx[self.intp_mim:]
        MIM_dis = fill_non_zero(self.intp_mask, intp_cap) / 2000
        MOS_dis = fill_non_zero(self.chip_mask, chip_cap) / 500

        return MIM_dis, MOS_dis

    def action_mask(self):
        """
        to get the available actions
        """
        mask = np.full((NCOL * NROW * 2, len(self.action_meaning)), (1, 0, 0), dtype=int)
        k = self.intp_mim
        for i in range(NCOL * NROW, NCOL * NROW * 2):
            if self.mask[i] == 1:
                if self.cur_params_idx[k] > 0:
                    mask[i][1] = 1
                elif self.cur_params_idx[k] < maxmos:
                    mask[i][2] = 1
                k += 1
        action_mask = mask.reshape(-1)

        return action_mask


if __name__ == "__main__":
    env = DecapPlaceParallel(1)
    o = env.reset()
    o, r, d, s, m, n = env.step(np.array([0]*242))
    print(o)


