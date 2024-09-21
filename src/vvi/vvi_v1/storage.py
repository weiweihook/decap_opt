import torch

class RolloutStorage(object):
    def __init__(self, num_steps, num_envs, obs_shape, action_shape, action_space):
        self.obs = torch.zeros((num_steps + 1, num_envs) + obs_shape)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape)
        self.logprobs = torch.zeros((num_steps, num_envs))
        self.rewards = torch.zeros((num_steps, num_envs))
        self.dones = torch.zeros((num_steps, num_envs))
        self.values = torch.zeros((num_steps, num_envs))
        self.advantages = torch.zeros((num_steps, num_envs))
        self.returns = torch.zeros((num_steps, num_envs))
        self.action_masks = torch.zeros((num_steps + 1, num_envs) + (action_space.sum(),))


    def to(self, device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.logprobs = self.logprobs.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        self.advantages = self.advantages.to(device)
        self.returns = self.returns.to(device)
        self.action_masks = self.action_masks.to(device)

    def insert(self, step, obs, actions, logprobs, rewards, dones, values, action_masks):
        self.obs[step+1].copy_(obs)
        self.actions[step].copy_(actions)
        self.logprobs[step].copy_(logprobs)
        self.values[step].copy_(values)
        self.rewards[step].copy_(rewards)
        self.dones[step].copy_(dones)
        self.action_masks[step + 1].copy_(action_masks)

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.action_masks[0].copy_(self.action_masks[-1])

    def compute_returns(self, num_steps, use_gae, next_value, next_done, gamma, gae_lambda, values, rewards, dones):
        returns, advantages = torch.zeros_like(values), torch.zeros_like(values)
        if use_gae:
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            for t in reversed(range(num_steps)):
                if t ==num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns - values

        return advantages, returns
