import torch
from tqdm import tqdm
from .metrics import ndcg_at_k


def batch_eval(
        info,
        n_rec,
        target_items,
        user_consumed,
        users,
        item_embeds=None,
        model_name=None,
        model=None
):
    with torch.no_grad():
        if model_name in ("bcq", "ddpg"):
            action = info["action"].detach()
            action = action / torch.norm(action, dim=1, keepdim=True)
            scores = torch.matmul(action, item_embeds.T)
            # B * n_rec
            _, rec_idxs = torch.topk(scores, n_rec, dim=1, sorted=False)
        elif model_name == "reinforce":
            action_probs = model.get_log_probs(action=info["action"])
            _, rec_idxs = torch.topk(action_probs, n_rec, dim=1, sorted=False)

        isins = (target_items[..., None] == rec_idxs).any(dim=1)
        rewards = isins.sum().tolist()

        rec_idxs = rec_idxs.cpu().numpy()
        ndcg_next_item = ndcg_at_k(target_items.cpu().numpy(), rec_idxs,
                                   next_item=True)
        ndcg_all_item = ndcg_at_k(user_consumed, rec_idxs, users.cpu().numpy(),
                                  n_rec, all_item=True)
        res = {"rewards": rewards,
               "ndcg_next_item": ndcg_next_item,
               "ndcg_all_item": ndcg_all_item}
        return res


def last_eval(
        model,
        eval_data,
        train_user_consumed,
        test_user_consumed,
        n_users,
        n_rec,
        item_embeds,
        eval_batch_size,
        mode="train",
        model_name=None,
        multi_sample=False,
        repeat_num=20
):
    with torch.no_grad():
        rec_idxs = []
        if model_name in ("bcq", "ddpg"):
            for i in tqdm(range(0, n_users, eval_batch_size), desc="last_eval"):
                batch = {"user": eval_data["user"][i: i+eval_batch_size],
                         "item": eval_data["item"][i: i+eval_batch_size]}
                action = model.select_action(batch, repeat_num, multi_sample)
                action = action / torch.norm(action, dim=1, keepdim=True)
                scores = torch.matmul(action, item_embeds.T)
                # B * n_rec
                top_scores, rec_index = torch.topk(scores, n_rec, dim=1)
                rec_idxs.append(rec_index)

            rec_idxs = torch.cat(rec_idxs, dim=0)
        #    action = action.cpu().detach()
        #    top_scores = top_scores.cpu().detach()
        #    print(f"{mode} continuous action - "
        #          f"min: {action.min():.4f}, "
        #          f"max: {action.max():.4f}, "
        #          f"mean: {action.mean():.4f}"
        #          f"\n\t{action[0][:7]}")
        #    print("top scores: ", top_scores[0][:7])

        elif model_name == "reinforce":
            for i in tqdm(range(0, n_users, eval_batch_size), desc="last_eval"):
                batch = {"user": eval_data["user"][i: i+eval_batch_size],
                         "item": eval_data["item"][i: i+eval_batch_size]}
                action_probs = model.get_log_probs(data=batch)
                _, rec_index = torch.topk(action_probs, n_rec, dim=1)
                rec_idxs.append(rec_index)
            rec_idxs = torch.cat(rec_idxs, dim=0)
        #    action_probs = action_probs.cpu().detach()
        #    print("top probs: ", action_probs[0][rec_index[0][:7]].detach())

    rec_idxs = rec_idxs.cpu().detach()
    print(f"{mode} recommendations: "
          f"\n{rec_idxs[[0, 17, 100, 684, 1000, 1584, 3000]]}")
    true_items = train_user_consumed if mode == "train" else test_user_consumed
    return ndcg_at_k(true_items, rec_idxs.numpy(), range(n_users), n_rec)


    #    action = action.unsqueeze(1)
    #    item_embeds = item_embeds.unsqueeze(0)
    #    l2_dist = (action - item_embeds).pow(2).sum(dim=2)
    #    scores = 1. - l2_dist

    #    _, rec = torch.topk(scores, n_rec * 2, dim=1)
    #    batch_size = scores.size(0)
    #    col_indices = torch.cat([torch.randperm(n_rec * 2)[:n_rec] for _ in range(batch_size)])
    #    row_indices = torch.arange(batch_size).repeat_interleave(n_rec)
    #    rec_idxs = rec[row_indices, col_indices].reshape(batch_size, -1)


