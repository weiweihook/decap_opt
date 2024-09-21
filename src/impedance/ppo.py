import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO():
    def __init__(self,
                 actor_critic,
                 update_epochs,
                 batch_size,
                 minibatch_size,
                 clip_coef,
                 norm_adv,
                 clip_vloss,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 target_kl,
                 lr
                 ):

        self.actor_critic = actor_critic
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)

    def update(self, rollouts, num_steps, obs_shape, action_shape):

        # flatten the batch
        b_obs = rollouts.obs[0:num_steps].reshape((-1,) + obs_shape)
        b_logprobs = rollouts.logprobs.reshape(-1)
        b_actions = rollouts.actions.reshape((-1,) + action_shape)
        b_advantages = rollouts.advantages.reshape(-1,)
        b_returns = rollouts.returns.reshape(-1,)
        b_values = rollouts.values.reshape(-1,)
        b_action_masks = rollouts.action_masks[0:num_steps].reshape((-1, rollouts.action_masks.shape[-1]))

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        v_loss_epoch = 0
        pg_loss_epoch = 0
        entropy_loss_epoch = 0
        loss_epoch = 0
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = self.actor_critic.get_action_and_value(b_obs[mb_inds],
                                                                                            b_action_masks[mb_inds],
                                                                                            b_actions.long()[mb_inds].T)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()


                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.clip_coef,
                                                                self.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()


                # Entropy loss
                entropy_loss = entropy.mean()
                # loss = policy_loss - entropy * entropy_coefficient + value_loss * value_coefficient
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                v_loss_epoch += v_loss.item()
                pg_loss_epoch += pg_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                loss_epoch += loss.item()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

        num_updates = self.update_epochs * int(self.batch_size/self.minibatch_size)

        v_loss_epoch /= num_updates
        pg_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates
        loss_epoch /= num_updates

        return v_loss_epoch, pg_loss_epoch, entropy_loss_epoch, loss_epoch

    def update2(self, rollouts, num_steps, obs_shape, action_shape):

        # flatten the batch
        b_obs = rollouts.obs[0:num_steps].reshape((-1,) + obs_shape)
        b_logprobs = rollouts.logprobs.reshape(-1)
        b_actions = rollouts.actions.reshape((-1,) + action_shape)
        b_advantages = rollouts.advantages.reshape(-1,)
        b_returns = rollouts.returns.reshape(-1,)
        b_values = rollouts.values.reshape(-1,)
        b_action_masks = rollouts.action_masks[0:num_steps].reshape((-1, rollouts.action_masks.shape[-1]))


        b_inds = np.arange(self.batch_size)
        clipfracs = []
        v_loss_epoch = 0
        pg_loss_epoch = 0
        entropy_loss_epoch = 0
        loss_epoch = 0
        for epoch in range(self.update_epochs):
            _, newlogprob, entropy, newvalue = self.actor_critic.get_action_and_value(b_obs,
                                                                                            b_action_masks,
                                                                                            b_actions.long().T)
            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()


            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

            mb_advantages = b_advantages

            if self.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if self.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns) ** 2
                v_clipped = b_values + torch.clamp(newvalue - b_values, -self.clip_coef,
                                                                self.clip_coef)
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()


            # Entropy loss
            entropy_loss = entropy.mean()
            # loss = policy_loss - entropy * entropy_coefficient + value_loss * value_coefficient
            loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            v_loss_epoch += v_loss.item()
            pg_loss_epoch += pg_loss.item()
            entropy_loss_epoch += entropy_loss.item()
            loss_epoch += loss.item()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        num_updates = self.update_epochs * int(self.batch_size/self.minibatch_size)

        v_loss_epoch /= num_updates
        pg_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates
        loss_epoch /= num_updates

        return v_loss_epoch, pg_loss_epoch, entropy_loss_epoch, loss_epoch
