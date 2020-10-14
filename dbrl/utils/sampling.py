import math
from random import random
import numpy as np


def sample_items_random(
        data,
        n_items,
        user_consumed_list,
        neg_label,
        num_neg=1
):
    user_consumed = {u: set(items) for u, items in user_consumed_list.items()}
    user_sampled = list()
    item_sampled = list()
    label_sampled = list()
    for u, i in zip(data.user, data.item):
        user_sampled.append(u)
        item_sampled.append(i)
        label_sampled.append(1.)
        for _ in range(num_neg):
            item_neg = math.floor(n_items * random())
            while item_neg in user_consumed[u]:
                item_neg = math.floor(n_items * random())
            user_sampled.append(u)
            item_sampled.append(item_neg)
            label_sampled.append(neg_label)
    return (
        np.array(user_sampled),
        np.array(item_sampled),
        np.array(label_sampled)
    )
