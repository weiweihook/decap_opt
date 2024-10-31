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

NCOL, NROW = 11, 11
execute_files = ['interposer_ac_novss1.sp', 'interposer_ac_novss2.sp', 'interposer_ac_novss3.sp', 'interposer_ac_novss4.sp']
port_files = ['port1_impeval.txt', 'port2_impeval.txt', 'port3_impeval.txt', 'port4_impeval.txt']
commands = []
for i in range(len(execute_files)):
    commands.append(['ngspice', execute_files[i]])

def run_os(path):
    original_path = os.getcwd()
    os.chdir(path)
    p = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in commands]
    for pp in p:
        pp.wait()
    os.chdir(original_path)


def readvdi(file):
    zvdi = readresult(file)
    z = zvdi[:, 2]
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

    action_meaning = [0, -200, 200]  # index: 0,1,2
    action_space = np.array([len(action_meaning)] * NCOL * NROW * 2)
    action_space_shape = (NCOL * NROW * 2,)
    single_action_space = (NCOL * NROW * 2,)
    observation_space_shape = (1, 4, NCOL, NROW)
    single_observation_space_shape = (4, NCOL, NROW)
    env_count = 1

    def __init__(self, env_config):

        self.path = 'config/case' + str(env_config) + '/'
        intp_mim, chip_mos, NCAP, intp_n, chip_n = config.read_config(self.path + 'case' + str(env_config) + '.conf')
        self.intp_mim = intp_mim
        self.chip_mos = chip_mos
        self.NCAP = NCAP
        self.intp_n = intp_n
        self.chip_n = chip_n
        self.cur_params_idx = np.zeros(self.NCAP, dtype=np.int32)
        intp_mask, chip_mask = gen_mask(self.intp_n, self.chip_n)
        self.intp_mask = intp_mask
        self.chip_mask = chip_mask
        mask = list(np.concatenate([self.intp_mask, self.chip_mask]).reshape(-1))
        self.mask = mask

    def reset(self):
        self.cur_params_idx = np.zeros(self.NCAP, dtype=np.int32)
        state = np.array([self.intp_mask, np.zeros((NCOL, NROW)), self.chip_mask, np.zeros((NCOL, NROW))])
        return state

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 2 mapped to the index of the corresponding parameter
        :return: state, reward, done
        """

        # Take action that RL agent returns to change current params
        action = list(action.reshape(-1))
        avail_action = [action[i] for i in range(len(action)) if self.mask[i] == 1]
        change = np.array([self.action_meaning[a] for a in avail_action])
        self.cur_params_idx = self.cur_params_idx + change
               
        reward = self.cal_reward(self.cur_params_idx)
      
        done = [True if sum(avail_action) == 0 else False]

        MIM_dis, MOS_dis = self.gen_dis()
        state = np.array([self.intp_mask, MIM_dis, self.chip_mask, MOS_dis])

        return state, reward, done, {}

    def cal_reward(self, cur_param_idx):
        """
        :param cur_param_idx: the grid capacitance (0~2000)
        :return reward
        """
        str_dc = ''
        esrs = ''
        for i, val in enumerate(cur_param_idx[:self.intp_mim]):
            str_dc += '.param dcap_int_val%d=%dp\n' % (i + 1, val)

        for j, val in enumerate(cur_param_idx[self.intp_mim:]):
            str_dc += '.param dcap_int_val%d=%dp\n' % (j + 1 + self.intp_mim, int(val * 0.25))
            if val != 0:
                esrs += '.param esr%d=%.3f\n' % (j + 1, 200 / val)
            else:
                esrs += '.param esr%d=0\n' % (j + 1)
        f = open(self.path + 'int_param_dcap.txt', 'w')
        f.write(str_dc)
        f.close()

        f1 = open(self.path + 'moscap_esr.txt', 'w')
        f1.write(esrs)
        f1.close()

        run_os(self.path)

        port1_arr = readresult(self.path + 'port1_impeval.txt')
        freq1_val = port1_arr[:, 1]
        port2_arr = readresult(self.path + 'port2_impeval.txt')
        freq2_val = port2_arr[:, 1]
        port3_arr = readresult(self.path + 'port3_impeval.txt')
        freq3_val = port3_arr[:, 1]
        port4_arr = readresult(self.path + 'port4_impeval.txt')
        freq4_val = port4_arr[:, 1]

        maxlist1 = []
        for j in range(len(port1_arr)):
            maxlist1.append(max([freq1_val[j], freq2_val[j], freq3_val[j], freq4_val[j]]))

        freq_val = np.array(maxlist1)
        freq = port1_arr[:, 0]
        total_impe = 0
        intp_cap_num = 0
        chip_cap_num = 0
        for i in range(len(freq_val)):
            if freq[i] < 3.5e9:
                if freq_val[i] > 0.035:
                    total_impe -= freq_val[i] - 0.035
            else:
                if freq_val[i] > 0.035*10**(math.log(freq[i], 10) - math.log(3.5e9, 10)):		# knee freq  = 3.5Ghz
                    a = freq_val[i] - 0.035*10**(math.log(freq[i], 10) - math.log(3.5e9, 10))
                    total_impe -= a
        
        intp_cap_val = 0  
        chip_cap_val = 0        
        
        for j in cur_param_idx[:self.intp_mim]:
            if j > 0:
                intp_cap_num += 1
                intp_cap_val += j
        
        for k in cur_param_idx[self.intp_mim:]:
            if k > 0:
                chip_cap_num += 1
                chip_cap_val += k*0.25
                
        if total_impe == 0:
            total_cost = 0.5 * (self.intp_mim * 2000 - intp_cap_val) / (self.intp_mim * 2000) + 0.5 * ((self.chip_mos * 500 - chip_cap_val) / (self.chip_mos * 500))
        else:
            total_cost = total_impe 

        return total_cost

    def gen_dis(self):
        """
        get the MIM and MOS distribution
        """
        intp_cap = self.cur_params_idx[:self.intp_mim]
        chip_cap = self.cur_params_idx[self.intp_mim:]
        MIM_dis = fill_non_zero(self.intp_mask, intp_cap) / 2000
        MOS_dis = fill_non_zero(self.chip_mask, chip_cap) / 2000

        return MIM_dis, MOS_dis

    def action_mask(self):
        """
        to get the available actions
        """
        mask = np.ones((NCOL * NROW * 2, len(self.action_meaning)))
        k = 0
        for i in range(NCOL * NROW * 2):
            if self.mask[i] == 1:
                if self.cur_params_idx[k] == 0:
                    mask[i][1] = 0
                elif self.cur_params_idx[k] >= 2000:
                    mask[i][2] = 0
                k += 1
        action_mask = mask.reshape(-1)

        return action_mask


if __name__ == "__main__":
    env = DecapPlaceParallel(2)
    o = env.reset()
    o, r, d, info = env.step(np.array([2]*242))
    print(o, r)
    print(env.action_mask)


