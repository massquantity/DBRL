import torch
import torch.nn as nn
from torch.distributions import Categorical


class Reinforce(nn.Module):
    def __init__(
            self,
            policy,
            policy_optim,
            beta,
            beta_optim,
            hidden_size,
            gamma=0.99,
            k=10,
            weight_clip=2.0,
            offpolicy_correction=True,
            topk=True,
            adaptive_softmax=True,
            cutoffs=None,
            device=torch.device("cpu"),
    ):
        super(Reinforce, self).__init__()
        self.policy = policy
        self.policy_optim = policy_optim
        self.beta = beta
        self.beta_optim = beta_optim
        self.beta_criterion = nn.CrossEntropyLoss()
        self.gamma = gamma
        self.k = k
        self.weight_clip = weight_clip
        self.offpolicy_correction = offpolicy_correction
        self.topk = topk
        self.adaptive_softmax = adaptive_softmax
        if adaptive_softmax:
            assert cutoffs is not None, (
                "must provide cutoffs when using adaptive_softmax"
            )
            self.softmax_loss = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=hidden_size,
                n_classes=policy.item_embeds.weight.size(0),
                cutoffs=cutoffs,
                div_value=4.
            ).to(device)
        self.device = device

    def update(self, data):
        (
            policy_loss,
            beta_loss,
            action,
            importance_weight,
            lambda_k
        ) = self._compute_loss(data)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.beta_optim.zero_grad()
        beta_loss.backward()
        self.beta_optim.step()

        info = {'policy_loss': policy_loss.cpu().detach().item(),
                'beta_loss': beta_loss.cpu().detach().item(),
                'importance_weight': importance_weight.cpu().mean().item(),
                'lambda_k': lambda_k.cpu().mean().item(),
                'action': action}
        return info

    def _compute_weight(self, policy_logp, beta_logp):
        if self.offpolicy_correction:
            importance_weight = torch.exp(policy_logp - beta_logp).detach()
            wc = torch.tensor([self.weight_clip]).to(self.device)
            importance_weight = torch.min(importance_weight, wc)
        #    importance_weight = torch.clamp(
        #        importance_weight, self.weight_clip[0], self.weight_clip[1]
        #    )
        else:
            importance_weight = torch.tensor([1.]).float().to(self.device)
        return importance_weight

    def _compute_lambda_k(self, policy_logp):
        lam = (
            self.k * ((1. - policy_logp.exp()).pow(self.k - 1)).detach()
            if self.topk
            else torch.tensor([1.]).float().to(self.device)
        )
        return lam

    def _compute_loss(self, data):
        if self.adaptive_softmax:
            state, action = self.policy(data)
            policy_out = self.softmax_loss(action, data["action"])
            policy_logp = policy_out.output

            beta_action = self.beta(state.detach())
            beta_out = self.softmax_loss(beta_action, data["action"])
            beta_logp = beta_out.output
        else:
            state, all_logp, action = self.policy.get_log_probs(data)
            policy_logp = all_logp[:, data["action"]]

            b_logp, beta_logits = self.beta.get_log_probs(state.detach())
            beta_logp = (b_logp[:, data["action"]]).detach()

        importance_weight = self._compute_weight(policy_logp, beta_logp)
        lambda_k = self._compute_lambda_k(policy_logp)

        policy_loss = -(
                importance_weight * lambda_k * data["return"] * policy_logp
        ).mean()

        if self.adaptive_softmax:
            if "beta_label" in data:
                b_state = self.policy.get_beta_state(data)
                b_action = self.beta(b_state.detach())
                b_out = self.softmax_loss(b_action, data["beta_label"])
                beta_loss = b_out.loss
            else:
                beta_loss = beta_out.loss
        else:
            if "beta_label" in data:
                b_state = self.policy.get_beta_state(data)
                _, b_logits = self.beta.get_log_probs(b_state.detach())
                beta_loss = self.beta_criterion(b_logits, data["beta_label"])
            else:
                beta_loss = self.beta_criterion(beta_logits, data["action"])
        return policy_loss, beta_loss, action, importance_weight, lambda_k

    def compute_loss(self, data):
        (
            policy_loss,
            beta_loss,
            action,
            importance_weight,
            lambda_k
        ) = self._compute_loss(data)

        info = {'policy_loss': policy_loss.cpu().detach().item(),
                'beta_loss': beta_loss.cpu().detach().item(),
                'importance_weight': importance_weight.cpu().mean().item(),
                'lambda_k': lambda_k.cpu().mean().item(),
                'action': action}
        return info

    def get_log_probs(self, data=None, action=None):
        with torch.no_grad():
            if self.adaptive_softmax:
                if action is None:
                    _, action = self.policy.forward(data)
                log_probs = self.softmax_loss.log_prob(action)
            else:
            #    _, log_probs = self.policy.get_log_probs(data)
                if action is None:
                    _, action = self.policy.forward(data)
                log_probs = self.policy.softmax_fc(action)
        return log_probs

    def forward(self, state):
        policy_logits = self.policy.get_action(state)
        policy_dist = Categorical(logits=policy_logits)
        _, rec_idxs = torch.topk(policy_dist.probs, 10, dim=1)
        return rec_idxs

