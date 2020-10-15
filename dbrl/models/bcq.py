from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCQ(nn.Module):
    def __init__(
            self,
            generator,
            gen_optim,
            perturbator,
            pert_optim,
            critic1,
            critic2,
            critic_optim,
            tau=0.001,
            gamma=0.99,
            lam=0.75,
            policy_delay=1,
            item_embeds=None,
            device=torch.device("cpu")
    ):
        super(BCQ, self).__init__()
        self.generator = generator
        self.gen_optim = gen_optim
        self.perturbator = perturbator
        self.pert_optim = pert_optim
        self.critic1 = critic1
        self.critic2 = critic2
        self.critic_optim = critic_optim
        self.tau = tau
        self.gamma = gamma
        self.lam = lam
        self.step = 1
        self.policy_delay = policy_delay
        self.perturbator_targ = deepcopy(perturbator)
        self.critic1_targ = deepcopy(critic1)
        self.critic2_targ = deepcopy(critic2)
        for p in self.perturbator_targ.parameters():
            p.requires_grad = False
        for p in self.critic1_targ.parameters():
            p.requires_grad = False
        for p in self.critic2_targ.parameters():
            p.requires_grad = False
    #    item_embeds = torch.as_tensor(item_embeds).to(device)
    #    self.item_embeds = item_embeds / (torch.norm(item_embeds, dim=1, keepdim=True) + 1e-7)
        self.item_embeds = torch.as_tensor(item_embeds).to(device)

    def update(self, data):
        generator_loss, state, mean, std = self._compute_generator_loss(
            data, self.item_embeds[data["action"]])
        state_copy = state.detach().clone()
        self.gen_optim.zero_grad()
        generator_loss.backward()
        self.gen_optim.step()

        critic_loss, y, q1, q2 = self._compute_critic_loss(data)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5, 2)
        self.critic_optim.step()

        if self.policy_delay <= 1 or (
                self.policy_delay > 1 and self.step % self.policy_delay == 0
        ):
            perturb_loss, action = self._compute_perturb_loss(state_copy)
            self.pert_optim.zero_grad()
            perturb_loss.backward()
            self.pert_optim.step()

            with torch.no_grad():
                self.soft_update(self.perturbator, self.perturbator_targ)
                self.soft_update(self.critic1, self.critic1_targ)
                self.soft_update(self.critic2, self.critic2_targ)
        else:
            perturb_loss = action = None

        self.step += 1
        info = {
            "generator_loss": generator_loss.cpu().detach().item(),
            "perturbator_loss": (
                perturb_loss.cpu().detach().item()
                if perturb_loss is not None
                else None
            ),
            "critic_loss": critic_loss.cpu().detach().item(),
            "y": y.cpu().mean().item(),
            "q1": q1.cpu().mean().item(),
            "q2": q2.cpu().mean().item(),
            "action": action,
            "mean": mean.cpu().mean().item(),
            "std": std.cpu().mean().item()
        }
        return info

    def compute_loss(self, data):
        generator_loss, state, mean, std = self._compute_generator_loss(
            data, self.item_embeds[data["action"]])
        critic_loss, y, q1, q2 = self._compute_critic_loss(data)
        perturb_loss, action = self._compute_perturb_loss(state)
        info = {
            "generator_loss": generator_loss.cpu().detach().item(),
            "perturbator_loss": (
                perturb_loss.cpu().detach().item()
                if perturb_loss is not None
                else None
            ),
            "critic_loss": critic_loss.cpu().detach().item(),
            "y": y.cpu().mean().item(),
            "q1": q1.cpu().mean().item(),
            "q2": q2.cpu().mean().item(),
            "action": action,
            "mean": mean.cpu().mean().item(),
            "std": std.cpu().mean().item()
        }
        return info

    def _compute_generator_loss(self, data, action):
        state, recon, mean, std = self.generator(data, action)
        recon_loss = F.mse_loss(recon, action)
        kl_div = -0.5 * (
                1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)
        ).mean()
        generator_loss = recon_loss + 0.5 * kl_div
        return generator_loss, state, mean, std

    def _compute_perturb_loss(self, state):
        sampled_actions = self.generator.decode(state)
        perturbed_actions = self.perturbator(state, sampled_actions)
        perturb_loss = -self.critic1(state, perturbed_actions).mean()
        return perturb_loss, perturbed_actions

    def _compute_critic_loss(self, data):
        with torch.no_grad():
            r, done = data["reward"], data["done"]
            batch_size = done.size(0)
            next_s = self.generator.get_state(data, next_state=True)
            next_s_repeat = torch.repeat_interleave(next_s, 10, dim=0)
            sampled_actions = self.generator.decode(next_s_repeat)
            perturbed_actions = self.perturbator_targ(next_s_repeat,
                                                      sampled_actions)

            q_targ1 = self.critic1_targ(next_s_repeat, perturbed_actions)
            q_targ2 = self.critic2_targ(next_s_repeat, perturbed_actions)
            q_targ = (
                    self.lam * torch.min(q_targ1, q_targ2)
                    + (1. - self.lam) * torch.max(q_targ1, q_targ2)
            )
            q_targ = q_targ.reshape(batch_size, -1).max(dim=1)[0]
            y = r + self.gamma * (1. - done) * q_targ

        s = self.generator.get_state(data).detach()
        gen_actions = self.generator.decode(s)
        a = self.perturbator(s, gen_actions).detach()
        #  a = self.item_embeds[data["action"]]
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        return critic_loss, y, q1, q2

    def soft_update(self, net, target_net):
        for targ_param, param in zip(target_net.parameters(), net.parameters()):
            targ_param.data.copy_(
                targ_param.data * (1. - self.tau) + param.data * self.tau
            )

    def select_action(self, data, repeat_num=20, multi_sample=False):
        with torch.no_grad():
            if multi_sample:
                batch_size = data["item"].size(0)
                state = self.generator.get_state(data)
                state = torch.repeat_interleave(state, repeat_num, dim=0)
                gen_actions = self.generator.decode(state)
                action = self.perturbator(state, gen_actions)
                q1 = self.critic1(state, action).view(batch_size, -1)
                indices = q1.argmax(dim=1)
                action = action.view(batch_size, repeat_num, -1)
                action = action[torch.arange(batch_size), indices, :]
            else:
                state = self.generator.get_state(data)
                gen_actions = self.generator.decode(state)
                action = self.perturbator(state, gen_actions)
        return action

    def forward(self, state):
        gen_actions = self.generator.decode(state)
        action = self.perturbator(state, gen_actions)
        action = action / (torch.norm(action, dim=1, keepdim=True) + 1e-7)
        item_embeds = self.item_embeds / (
                torch.norm(self.item_embeds, dim=1, keepdim=True) + 1e-7
        )
        scores = torch.matmul(action, item_embeds.T)
        _, rec_idxs = torch.topk(scores, 10, dim=1)
        return rec_idxs



