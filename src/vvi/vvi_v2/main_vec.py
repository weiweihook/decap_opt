import random
import time
from arguments import get_args
from storage import RolloutStorage
import numpy as np
import torch
import model1 as model
from env import DecapPlaceParallel
from ppo import PPO
import os
import logging

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    case_list = [1] #[1,2,3,4,5]
    logname = 'log.txt'
    now_time = time.strftime("%Y%m%d-%H%M", time.localtime(time.time()))
    t1 = ''.join([x for x in now_time if x.isdigit()])
    for case in case_list:
        path = 'runs/case'+str(case)+'/' + str(t1)+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        args = get_args()
        args.num_envs = 1
        args.num_steps = 50
        args.batch_size = int(args.num_envs * (args.num_steps - args.abandon_size))
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.idx_list = [case]*args.num_envs

        # GPU or CPU
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        # device = 'cpu'

        # logging
        logger = logging.getLogger()
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(path + logname)
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        # seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        # load env
        vec_env = DecapPlaceParallel(case)
        actor_critic = model.PPONetwork(vec_env).to(device)
        agent = PPO(actor_critic,
                    args.update_epochs,
                    args.batch_size,
                    args.minibatch_size,
                    args.clip_coef,
                    args.norm_adv,
                    args.clip_vloss,
                    args.ent_coef,
                    args.vf_coef,
                    args.max_grad_norm,
                    args.target_kl,
                    args.learning_rate)

        # ALGO Logic: Storage setup
        rollouts = RolloutStorage(args.num_steps,
                                  args.num_envs,
                                  vec_env.single_observation_space_shape,
                                  vec_env.action_space_shape,
                                  vec_env.action_space)
        rollouts.to(device)

        # num_updates = args.total_timesteps // args.batch_size
        num_updates = 600
        BEST = [-50, 0, 0]  # reward, updates, steps
        BEST_Allocation = vec_env.cur_params_idx

        start_time = time.time()
        print(start_time)
        vec_obs = torch.Tensor(vec_env.reset())
        next_obs = vec_obs.unsqueeze(0).to(device)
        for update in range(1, num_updates + 1):

            vec_action_mask = torch.Tensor(vec_env.action_mask()).unsqueeze(0).to(device)
            rollouts.obs[0].copy_(next_obs)
            rollouts.action_masks[0].copy_(vec_action_mask)
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                agent.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):

                #############################################################
                ###### Obs + Action Masks => Agent => Actions################
                ###### Actions => Env => Rewards, log_probs, New Action Maks#
                vec_action_mask = torch.Tensor(vec_env.action_mask()).unsqueeze(0).to(device)

                with torch.no_grad():
                    vec_action, vec_logprob, _, vec_value = actor_critic.get_action_and_value(next_obs, vec_action_mask)

                # Cal rewards for given actions
                vec_obs, vec_reward, vec_done, vec_vvi, vec_add_mos, vec_vio_node = vec_env.step(vec_action.cpu().numpy())

                #############################################################
                #############################################################

                next_done = torch.Tensor(vec_done).to(device)
                next_obs = torch.Tensor(vec_obs).unsqueeze(0).to(device)
                if vec_reward > BEST[0]:
                    BEST = vec_reward, update, step
                    BEST_Allocation = vec_env.cur_params_idx + vec_env.init_curr_params_idx

                with open(path + 'train_int_dcap.txt', 'a') as f1:
                    train = f'{update} {step} {round(vec_reward, 4)}{vec_env.cur_params_idx + vec_env.init_curr_params_idx}\n'
                    f1.write(train)
                    f1.close()

                if next_done == True:
                    next_obs = torch.Tensor(vec_env.reset()).unsqueeze(0)

                rollouts.insert(step, next_obs,
                                vec_action.reshape([-1, vec_env.action_space_shape[0]]),
                                vec_logprob, torch.tensor(vec_reward).view(-1),
                                next_done, vec_value.flatten(), torch.Tensor(vec_env.action_mask()))

                if update % 10 == 0 and step <= args.num_steps - 1:
                    logging.info('update: {}, step: {}, reward: {}, VVI: {}, ADD MOS:{}, Violation nodes: {}'
                                 .format(update, step, vec_reward, vec_vvi, vec_add_mos, vec_vio_node))

            with torch.no_grad():
                next_value = actor_critic.get_value(next_obs)
                next_value = next_value.reshape(1, -1)
                adv, rollouts.returns = rollouts.compute_returns(args.num_steps, args.gae, next_value, next_done,
                                                                 args.gamma, args.gae_lambda,
                                                                 rollouts.values, rollouts.rewards, rollouts.dones)
            rollouts.advantages = adv

            v_loss, pg_loss, entropy_loss, loss = agent.update(rollouts,
                                                                args.num_steps,
                                                                vec_env.single_observation_space_shape,
                                                                vec_env.action_space_shape)


            if update % 1 == 0:
                # torch.save(actor_critic.state_dict(), path + 'vec_agent_params' + str(update) + '.pth')
                logging.info('Best: {}, update :{}, step :{}'.format(BEST[0], BEST[1], BEST[2]))
                np.savetxt(path + 'allocation.txt', BEST_Allocation)
                str_dc = ''
                esrs = ''
                for j, val in enumerate(BEST_Allocation):
                    str_dc += '.param dcap_int_val%d=%dp\n' % (j + 1 + vec_env.intp_mim, val)
                    if val != 0:
                        esrs += '.param esr%d=%.3f\n' % (j + 1, 50 / val)
                    else:
                        esrs += '.param esr%d=0\n' % (j + 1)
                f = open(path + 'chip_param_dcap.txt', 'w')
                f.write(str_dc)
                f.close()
                f = open(path + 'moscap_esr.txt', 'w')
                f.write(esrs)
                f.close()

            if update >= 100:
                break

        end_time = time.time()
        logging.info('time cost: {} s'.format(end_time - start_time))

        # save network parameters
        torch.save(actor_critic, path + 'vec_agent.pth')



