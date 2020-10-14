import argparse
import numpy as np
import torch


def compute_returns(rewards, discount_factor, sess_end=None, normalize=False):
    total_returns = []
    if sess_end is None:
        last_val = 0
        for r in reversed(rewards):
            last_val = r + discount_factor * last_val
            total_returns.append(last_val)
        total_returns.reverse()
    else:
        for rew in np.split(rewards, sess_end):
            returns = []
            last_val = 0
            for r in reversed(rew):
                last_val = r + discount_factor * last_val
                returns.append(last_val)
            returns.reverse()
            total_returns.extend(returns)

    total_returns = np.asarray(total_returns)
    if normalize:
    #    total_returns = (total_returns - total_returns.mean()) / (total_returns.std() + 1e-5)
        total_returns /= (np.linalg.norm(total_returns) + 1e-7)
    return total_returns


def generate_embeddings(model, n_users, n_items, feat_map, static_feat,
                        dynamic_feat, device):
    whole_data = dict()
    whole_data["user"] = torch.arange(n_users)
    # may contain unseen item in test data
    whole_data["item"] = torch.arange(n_items + 1)
    if static_feat is not None:
        for feat in static_feat:
            whole_data[feat] = torch.as_tensor(
                [feat_map[feat][u] for u in range(n_users)])
    if dynamic_feat is not None:
        for feat in dynamic_feat:
            whole_data[feat] = torch.as_tensor(
                [feat_map[feat][i] for i in range(n_items + 1)])

    with torch.no_grad():
        model.eval()
        whole_data = {k: v.to(device) for k, v in whole_data.items()}
        user_embeds, item_embeds = model.get_embedding(whole_data)
        user_embeds = user_embeds.cpu().detach().numpy().astype(np.float32)
        item_embeds = item_embeds.cpu().detach().numpy().astype(np.float32)
    return user_embeds, item_embeds


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ("yes", "true", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean liked value expected...")

