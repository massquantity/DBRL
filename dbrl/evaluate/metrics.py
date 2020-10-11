import numpy as np


def ndcg_at_k(y_true_list, y_reco_list, users=None, k=10, next_item=False,
              all_item=False):
    if next_item:
        ndcg_all = []
        y_true_list = y_true_list.tolist()
        y_reco_list = y_reco_list.tolist()
        for y_true, y_reco in zip(y_true_list, y_reco_list):
            if y_true in y_reco:
                index = y_reco.index(y_true)
                ndcg = 1. / np.log2(index + 2)
            else:
                ndcg = 0.
            ndcg_all.append(ndcg)
        return np.mean(ndcg_all)

    elif all_item:
        ndcg_all = []
        users = users.tolist()
        y_reco_list = y_reco_list.tolist()
        for i in range(len(y_reco_list)):
            y_true = y_true_list[users[i]]
            y_reco = y_reco_list[i]
            ndcg_all.append(ndcg_one(y_true, y_reco, k))
        return np.mean(ndcg_all)

    else:
        ndcg_all = list()
        for u in users:
            y_true = y_true_list[u]
            y_reco = y_reco_list[u]
            ndcg_all.append(ndcg_one(y_true, y_reco, k))
        return np.mean(ndcg_all)


def ndcg_one(y_true, y_reco, k):
    rank_list = np.zeros(k)
    common_items, indices_in_true, indices_in_reco = np.intersect1d(
        y_true, y_reco, assume_unique=False, return_indices=True)

    if common_items.size > 0:
        rank_list[indices_in_reco] = 1
        ideal_list = np.sort(rank_list)[::-1]
        #  np.sum(rank_list / np.log2(2, k+2))
        dcg = np.sum(rank_list / np.log2(np.arange(2, k + 2)))
        idcg = np.sum(ideal_list / np.log2(np.arange(2, k + 2)))
        ndcg = dcg / idcg
    else:
        ndcg = 0.
    return ndcg


