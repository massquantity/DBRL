import torch
import torch.nn as nn


def count_vars(module):
    return sum([torch.numel(p) for p in module.parameters() if p.requires_grad])


def init_param_bcq(generator, perturbator, critic1, critic2):
    for name, param in generator.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
        elif "embeds" not in name:
            nn.init.xavier_uniform_(param)    # gain=0.1
    for name, param in perturbator.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
        else:
            nn.init.xavier_uniform_(param)
    for name, param in critic1.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
        else:
            nn.init.xavier_uniform_(param)
    for name, param in critic2.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
        else:
            nn.init.xavier_uniform_(param)  # nn.init.normal_(param.weight, 0.0, 0.01)


def init_param(actor, other):
    for name, param in actor.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
        elif "embeds" not in name:
            nn.init.xavier_uniform_(param)
    for name, param in other.named_parameters():
        if "bias" in name:
            nn.init.zeros_(param)
        else:
            nn.init.xavier_uniform_(param)


def init_param_dssm(model):
    for name, param in model.named_parameters():
        if "fc" in name:
            if "bias" in name:
                nn.init.zeros_(param)
            else:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
        elif "embed" in name:
            nn.init.xavier_uniform_(param)
        #    nn.init.normal_(param, 0.0, 0.01)
            nn.init.zeros_(param[-1])
