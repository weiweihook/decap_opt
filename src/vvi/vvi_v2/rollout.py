#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ray.cluster_utils import Cluster
import argparse
import json
import os
import pickle
import IPython
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
import ray
#from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter
#from bag_deep_ckt.autockt.envs.bag_opamp_discrete import TwoStageAmp
#from envs.ngspice_vanilla_opamp import TwoStageAmp
from decap_ppo_demo import *

cluster = Cluster(
    initialize_head=True,
    head_node_args={
        "num_cpus": 7,
    })

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""
# Note: if you use any custom models or envs, register them here first, e.g.:
#
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionCartpole(10))
register_env("decap-v0", lambda config:DecapPlaceParallel(config))

def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    parser.add_argument(
        "--num_val_specs",
        type=int,
        default=50,
        help="Number of untrained objectives to test on")
    parser.add_argument(
        "--traj_len",
        type=int,
        default=30,
        help="Length of each trajectory")
    return parser

z_histo = []
n_bins = 20
def savefig1(capidx, rollout_step,reward):
    plt.rcParams['figure.figsize']=(6.4,5.4)
    res_array = readresult(str(os.getpid())+'_duo_vdd_1_vdi.csv')
    x = res_array[91:96, 0]
    y = res_array[91:96, 1]
    z = res_array[91:96, 2]
    for i in range(1,5):
        x = np.concatenate((x,res_array[91+12*i:96+12*i,0]))
        y = np.concatenate((y,res_array[91+12*i:96+12*i,1]))
        z = np.concatenate((z,res_array[91+12*i:96+12*i,2]))

    vdi = z.sum()
    fig, (ax2) = plt.subplots(nrows=1)
    ax2.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax2.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax2)
    ax2.plot(x, y, 'ko', ms=3)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    plt.subplots_adjust(hspace=0.5)
    plt.title('VDI={},reward={}\ncap={},{}'.format(vdi, round(reward,6),capidx[:15],np.sum(capidx)))
    z_histo.append(vdi)
    plt.savefig('./result_fig/'+str(os.getpid())+'_VDI_distri_'+ str(rollout_step))
    fig,axs = plt.subplots(1,1)
    axs.hist(z,bins = n_bins)
    plt.title('VDI distribution histogram')
    plt.savefig('./result_fig/'+str(os.getpid())+'_VDI_histogram_'+ str(rollout_step))
    plt.close('all')
    if rollout_step == args.num_val_specs - 1:
        f = open('decap_ppo_demo.py','r')
        f1 = f.readlines()
        f.close()
        n_binss = 20
        fig,axs = plt.subplots(1,1)
        axs.hist(z_histo,bins = n_binss)
        plt.title('%d inferences VDI histogram of %s' % (args.num_val_specs,f1[62]))
        plt.savefig('./result_fig/%s_VDI_all_histogram' %str(os.getpid()))
        plt.close()


def savefig( rollout_step,reward):
    
    #data = readresult(str(os.getpid())+'_freq_data.txt')
    os.system('cp ./%s ./result_fig/%s_port0_impeval%d.txt' % (str(os.getpid())+'_port0_impeval.txt',str(os.getpid()),rollout_step))
    os.system('cp ./%s ./result_fig/%s_port1_impeval%d.txt' % (str(os.getpid())+'_port1_impeval.txt',str(os.getpid()),rollout_step))
    os.system('cp ./%s ./result_fig/%s_port2_impeval%d.txt' % (str(os.getpid())+'_port2_impeval.txt',str(os.getpid()),rollout_step))
    os.system('cp ./%s ./result_fig/%s_param_dcap%d.txt' % (str(os.getpid())+'_int_param_dcap.txt',str(os.getpid()),rollout_step))
    for j in range(3):
        a = np.genfromtxt((str(os.getpid())+'_port%d_impeval.txt')% j)
        target_impe = []
        for i in a[154:,0]:
            tar = 0.1 * 10 ** (math.log(i,10) - math.log(a[154,0],10))
            target_impe.append(tar)
        
        plt.plot(a[:,0],a[:,1])
        plt.title('reward:%f'%reward)
        plt.plot(a[:155,0],[0.1]*155,c='k',linestyle='--')
        plt.plot(a[154:,0],target_impe,c='k',linestyle='--')
        plt.xlabel('freq   /Hz')
        plt.ylabel('impedance   / ohm')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.savefig('./result_fig/'+str(os.getpid())+'_impedance_plot%d_chip%d'%(rollout_step,j))
        plt.close()


def run(args, parser):
    config = args.config
    if not config:
        # Load configuration from file
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.json")
        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.json in either the checkpoint dir or "
                "its parent directory.")
        with open(config_path) as f:
            config = json.load(f)
        if "num_workers" in config:
            config["num_workers"] = min(5, config["num_workers"])

    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init(include_dashboard=True, dashboard_port=9999, address = cluster.address)

    #cls = get_agent_class(args.run)
    cls = get_trainable_cls(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    rollout(agent, args.env, num_steps, args.out, args.no_render)
    ray.shutdown()

def rollout(agent, env_name, num_steps, out="assdf", no_render=True):
    #if hasattr(agent, "local_evaluator"):
        #env = agent.local_evaluator.env
    env_config = {} #{"generalize":True,"num_valid":args.num_val_specs, "save_specs":False, "run_valid":True}
    if env_name == "decap-v0":
        env = DecapPlaceParallel(env_config=env_config)
    else:
        env = gym.make(env_name)

    #get unnormlaized specs
    #norm_spec_ref = env.global_g
    #spec_num = len(env.specs)
    ideal_spec =8e-10
     
    if hasattr(agent, "local_evaluator"):
        state_init = agent.local_evaluator.policy_map[
            "default"].get_initial_state()
    else:
        state_init = []
    if state_init:
        use_lstm = True
    else:
        use_lstm = False
    #state_init = []
    rollouts = []
    next_states = []
    obs_reached = []
    obs_nreached = []
    action_array = []
    action_arr_comp = []
    rollout_steps = 0
    reached_spec = 0
    while rollout_steps < args.num_val_specs:
        if out is not None:
            rollout_num = []
        state = env.reset()
        
        done = False
        reward_total = 0.0
        steps=0
        while not done and steps < args.traj_len:
            if use_lstm:
                action, state_init, logits = agent.compute_single_action(
                    state, state=state_init)
            else:
                action = agent.compute_single_action(state)
                action_array.append(action)

            next_state, reward, done, _ = env.step(action)
            print(action)
            print(reward)
            print(done)
            reward_total += reward
            if not no_render:
                env.render()
            if out is not None:
                rollout_num.append(reward)
                next_states.append(next_state)
            steps += 1
            state = next_state
        if done == True:
            reached_spec += 1
            obs_reached.append(ideal_spec)
            action_arr_comp.append(action_array)
            action_array = []
            #pickle.dump(action_arr_comp, open("action_arr_test", "wb"))
        else:
            obs_nreached.append(ideal_spec)          #save unreached observation 
            action_array=[]
        if out is not None:
            rollouts.append(rollout_num)
        #print("Episode reward", reward_total)
        savefig(rollout_steps,reward) 
        rollout_steps+=1
        #if out is not None:
            #pickle.dump(rollouts, open(str(out)+'reward', "wb"))
        #pickle.dump(obs_reached, open("opamp_obs_reached_test","wb"))
        #pickle.dump(obs_nreached, open("opamp_obs_nreached_test","wb"))
        #print("Specs reached: " + str(reached_spec) + "/" + str(len(obs_nreached)))

    #print("Num specs reached: " + str(reached_spec) + "/" + str(args.num_val_specs))

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
