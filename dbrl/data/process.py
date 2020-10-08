from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from .session import build_sess_end
from .split import split_by_ratio


def process_data(path, columns=None, test_size=0.2, time_col="time",
                 sess_mode="one", interval=None, reward_shape=None):
    """Split and process data before building dataloader.

    Parameters
    ----------
    path : str
        File path.
    columns : list, optional
        Column names for the original data.
    test_size : float
        Test data size to split from original data.
    time_col : str
        Specify which column represents time.
    sess_mode : str
        Ways of representing a session.
    interval : int
        Interval between different sessions.
    reward_shape : dict (default None)
        A dict for mapping labels to rewards.

    Returns
    -------
    n_users : int
        Number of users.
    n_items : int
        Number of items.
    train_user_consumed : dict
        Items interacted by each user in train data.
    test_user_consumed : dict
        Items interacted by each user in test data.
    train_sess_end : dict
        Session end mark for each user in train data.
    test_sess_end : dict
        Session end mark for each user in test data.
    train_rewards : dict (default None)
        A dict for mapping train users to rewards.
    test_rewards : dict (default None)
        A dict for mapping test users to rewards.
    """

    column_names = columns if columns is not None else None
    data = pd.read_csv(path, sep=",", names=column_names)
    assert time_col in data.columns, "must specify correct time column name..."

    data = data.sort_values(by=time_col).reset_index(drop=True)
    train_data, test_data = split_by_ratio(data, shuffle=False,
                                           test_size=test_size,
                                           pad_unknown=True,
                                           seed=42)

    train_data, test_data = map_unique_value(train_data, test_data)
    n_users = train_data.user.nunique()
    n_items = train_data.item.nunique()
    # groupby too slow...
    # train_user_consumed = train_data.groupby("user")["item"].apply(
    #    lambda x: list(x)).to_dict()
    # test_user_consumed = test_data.groupby("user")["item"].apply(
    #    lambda x: list(x)).to_dict()
    train_user_consumed = build_interaction(train_data)
    test_user_consumed = build_interaction(test_data)
    if reward_shape is not None:
        train_rewards = build_reward(train_data, reward_shape)
        test_rewards = build_reward(test_data, reward_shape)
    else:
        train_rewards = test_rewards = None

    train_sess_end = build_sess_end(train_data, sess_mode, time_col, interval)
    test_sess_end = build_sess_end(test_data, sess_mode, time_col, interval)

    result = (
        n_users,
        n_items,
        train_user_consumed,
        test_user_consumed,
        train_sess_end,
        test_sess_end,
        train_rewards,
        test_rewards
    )
    return result


def process_feat_data(path, columns=None, test_size=0.2, time_col="time",
                 static_feat=None, dynamic_feat=None):
    column_names = columns if columns is not None else None
    data = pd.read_csv(path, sep=",", names=column_names)
    data = data.sort_values(by=time_col).reset_index(drop=True)
    train_data, test_data = split_by_ratio(data, shuffle=False,
                                           test_size=test_size,
                                           pad_unknown=True,
                                           seed=42)

    train_data, test_data, user_map, item_map = map_unique_value_feat(
        train_data, test_data, static_feat, dynamic_feat
    )
    n_users = train_data.user.nunique()
    n_items = train_data.item.nunique()
    train_user_consumed = build_interaction(train_data)
    test_user_consumed = build_interaction(test_data)
    feat_map = build_feat_map(train_data, n_items, static_feat, dynamic_feat)
    result = (
        n_users,
        n_items,
        train_user_consumed,
        test_user_consumed,
        train_data,
        test_data,
        user_map,
        item_map,
        feat_map
    )
    return result


def map_unique_value(train_data, test_data):
    for col in ["user", "item"]:
        # unique_vals = np.unique(train_data[col])
        # mapping = dict(zip(unique_vals, range(len(unique_vals))))
        # map according to frequency
        counts = train_data[col].value_counts()
        freq = counts.index.tolist()
        mapping = dict(zip(freq, range(len(freq))))
        train_data[col] = train_data[col].map(mapping)
        test_data[col] = test_data[col].map(mapping)
        if test_data[col].isnull().any():
            col_type = train_data[col].dtype
            test_data[col].fillna(len(freq), inplace=True)
            test_data[col] = test_data[col].astype(col_type)
    return train_data, test_data


def map_unique_value_feat(train_data, test_data, static_feat=None,
                          dynamic_feat=None):
    user_map = dict()
    item_map = dict()
    total_cols = ["user", "item"]
    if static_feat is not None:
        total_cols.extend(static_feat)
    if dynamic_feat is not None:
        total_cols.extend(dynamic_feat)
    for col in total_cols:
        # unique_vals = np.unique(train_data[col]).tolist()
        # mapping = dict(zip(unique_vals, range(len(unique_vals))))
        # map according to frequency
        counts = train_data[col].value_counts()
        freq = counts.index.tolist()
        mapping = dict(zip(freq, range(len(freq))))
        train_data[col] = train_data[col].map(mapping)
        test_data[col] = test_data[col].map(mapping)
        if col == "user":
            user_map = mapping.copy()
        elif col == "item":
            item_map = mapping.copy()
        if test_data[col].isnull().any():
            col_type = train_data[col].dtype
            test_data[col].fillna(len(freq), inplace=True)
            test_data[col] = test_data[col].astype(col_type)
    return train_data, test_data, user_map, item_map


def build_interaction(data):
    consumed = defaultdict(list)
    for u, i in zip(data.user.tolist(), data.item.tolist()):
        consumed[u].append(i)
    return consumed


def build_reward(data, reward_shape):
    label_all = defaultdict(list)
    for u, l in zip(data.user.tolist(), data.label.tolist()):
        label_all[u].append(l)

    reward_all = defaultdict(dict)
    for user, label in label_all.items():
        for key, rew in reward_shape.items():
            index = np.where(np.array(label) == key)[0]
            if len(index) > 0:
                reward_all[user].update({key: index})
    return reward_all


def build_feat_map(data, n_items, static_feat=None, dynamic_feat=None):
    feat_map = dict()
    if static_feat is not None:
        for feat in static_feat:
            feat_map[feat] = dict(zip(data["user"], data[feat]))
            feat_map[feat + "_vocab"] = data[feat].nunique()
    if dynamic_feat is not None:
        for feat in dynamic_feat:
            feat_map[feat] = dict(zip(data["item"], data[feat]))
            feat_map[feat + "_vocab"] = data[feat].nunique()
            # avoid oov item features
            feat_map[feat][n_items] = feat_map[feat + "_vocab"]
    return feat_map


# def build_batch_data(mode, train_user_consumed, hist_num, n_users, device):
#    if mode == "train":
#        items = [train_user_consumed[u][:hist_num] for u in range(n_users)]
#    else:
#        items = [train_user_consumed[u][-hist_num:] for u in range(n_users)]
#    data = {"user": torch.as_tensor(range(n_users)).to(device),
#            "item": torch.as_tensor(items).to(device)}
#    return data


def build_batch_data(mode, train_user_consumed, hist_num, n_users, pad_val,
                     device):
    items = np.array(
        [
            last_history(train_user_consumed[u], hist_num, pad_val, mode)
            for u in range(n_users)
        ]
    )
    data = {"user": torch.as_tensor(range(n_users)).to(device),
            "item": torch.from_numpy(items).to(device)}
    return data


def last_history(history, hist_num, pad_val, mode):
    hist_len = len(history)
    if hist_len < hist_num:
        rec_hist = np.full(hist_num, pad_val, dtype=np.int64)
        rec_hist[-hist_len:] = history
    else:
        rec_hist = (
            np.array(history[:hist_num])
            if mode == "train"
            else np.array(history[-hist_num:])
        )
    return rec_hist

