from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPG(nn.Module):
    def __init__(
            self,
            actor,
            actor_optim,
            critic,
            critic_optim,
            tau=0.001,
            gamma=0.99,
            policy_delay=1,
            item_embeds=None,
            device=torch.device("cpu")
    ):
        super(DDPG, self).__init__()
        self.actor = actor
        self.actor_optim = actor_optim
        self.critic = critic
        self.critic_optim = critic_optim
        self.tau = tau
        self.gamma = gamma
        self.step = 1
        self.policy_delay = policy_delay
        self.actor_targ = deepcopy(actor)
        self.critic_targ = deepcopy(critic)
        for p in self.actor_targ.parameters():
            p.requires_grad = False
        for p in self.critic_targ.parameters():
            p.requires_grad = False
    #    item_embeds = torch.as_tensor(item_embeds).to(device)
    #    self.item_embeds = item_embeds / (torch.norm(item_embeds, dim=1, keepdim=True) + 1e-7)
        self.item_embeds = torch.as_tensor(item_embeds).to(device)

    def update(self, data):
        critic_loss, y, q = self._compute_critic_loss(data)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5, 2)
        self.critic_optim.step()

        if self.policy_delay <= 1 or (
                self.policy_delay > 1 and self.step % self.policy_delay == 0
        ):
            actor_loss, action = self._compute_actor_loss(data)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            with torch.no_grad():
                self.soft_update(self.actor, self.actor_targ)
                self.soft_update(self.critic, self.critic_targ)
        else:
            actor_loss = action = None

        self.step += 1
        info = {
            "actor_loss": (
                actor_loss.cpu().detach().item()
                if actor_loss is not None
                else None
            ),
            "critic_loss": critic_loss.cpu().detach().item(),
            "y": y, "q": q,
            "action": action
        }
        return info

    def compute_loss(self, data):
        actor_loss, action = self._compute_actor_loss(data)
        critic_loss, y, q = self._compute_critic_loss(data)
        info = {
            "actor_loss": (
                actor_loss.cpu().detach().item()
                if actor_loss is not None
                else None
            ),
            "critic_loss": critic_loss.cpu().detach().item(),
            "y": y, "q": q,
            "action": action
        }
        return info

    def _compute_actor_loss(self, data):
        state, action = self.actor(data)
        actor_loss = -self.critic(state, action).mean()
        return actor_loss, action

    def _compute_critic_loss(self, data):
        with torch.no_grad():
            r, done = data["reward"], data["done"]
            next_s = self.actor_targ.get_state(data, next_state=True)
            next_a = self.actor_targ.get_action(next_s)
            q_targ = self.critic_targ(next_s, next_a)
            y = r + self.gamma * (1. - done) * q_targ

        s = self.actor.get_state(data)
        a = self.item_embeds[data["action"]]
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, y)
        return critic_loss, y, q

    def soft_update(self, net, target_net):
        for targ_param, param in zip(target_net.parameters(), net.parameters()):
            targ_param.data.copy_(
                targ_param.data * (1. - self.tau) + param.data * self.tau
            )

    def select_action(self, data, *args):
        with torch.no_grad():
            _, action = self.actor(data)
        return action

    def forward(self, state):
        action = self.actor.get_action(state)
        action = action / torch.norm(action, dim=1, keepdim=True)
        item_embeds = self.item_embeds / torch.norm(
            self.item_embeds, dim=1, keepdim=True
        )
        scores = torch.matmul(action, item_embeds.T)
        _, rec_idxs = torch.topk(scores, 10, dim=1)
        return rec_idxs
