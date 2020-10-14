import math
import random
import numpy as np
from .split import groupby_user
from dbrl.utils.misc import compute_returns


def rolling_window(a, window):
    assert window <= a.shape[-1], "window size too large..."
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def pad_session(hist_len, hist_num, hist_items, pad_val):
    """Pad items sequentially.

    For example, a user's whole item interaction is [1,2,3,4,5],
    then it will be converted to the following matrix:
    x x x x 1 2
    x x x 1 2 3
    x x 1 2 3 4
    x 1 2 3 4 5

    Where x denotes the padding-value. Then for the first line, [x x x x 1]
    will be used as state, and [2] as action.

    If the length of interaction is longer than hist_num, the rest will be
    handled by function `rolling_window`, which converts the rest of
    interaction to:
    1 2 3 4 5 6
    2 3 4 5 6 7
    3 4 5 6 7 8
    ...

    In this case no padding value is needed. So basically every user in the
    data will call `pad_session`, but only users with long interactions will
    need to call `rolling_window`.
    """
    sess_len = hist_len - 1 if hist_len - 1 < hist_num - 1 else hist_num - 1
    session_first = np.full((sess_len, hist_num + 1), pad_val, dtype=np.int64)
    for i in range(sess_len):
        offset = i + 2
        session_first[i, -offset:] = hist_items[:offset]
    return session_first


def build_session(
        n_users,
        n_items,
        hist_num,
        train_user_consumed,
        test_user_consumed=None,
        train=True,
        sess_end=None,
        sess_mode="one",
        neg_sample=None,
        train_rewards=None,
        test_rewards=None,
        reward_shape=None
):
    user_sess, item_sess, reward_sess, done_sess = [], [], [], []
    user_consumed_set = {
        u: set(items) for u, items in train_user_consumed.items()
    }
    for u in range(n_users):
        if train:
            items = np.asarray(train_user_consumed[u])
        else:
            items = np.asarray(
                train_user_consumed[u][-hist_num:] + test_user_consumed[u]
            )

        hist_len = len(items)
        expanded_items = pad_session(hist_len, hist_num, items, pad_val=n_items)

        if hist_len > hist_num:
            full_size_sess = rolling_window(items, hist_num + 1)
            expanded_items = np.concatenate(
                [expanded_items, full_size_sess],
                axis=0
            )

        if train and neg_sample is not None:
            expanded_items, num_neg, _ = sample_neg_session(
                expanded_items, user_consumed_set[u], n_items, neg_sample
            )

        sess_len = len(expanded_items)
        user_sess.append(np.tile(u, sess_len))
        item_sess.append(expanded_items)

        if reward_shape is not None:
            reward = assign_reward(
                sess_len, u, train, train_rewards, test_rewards,
                train_user_consumed, hist_num, reward_shape
            )
        else:
            reward = np.ones(sess_len, dtype=np.float32)

        if train and neg_sample is not None and num_neg > 0:
            reward[-num_neg:] = 0.
        reward_sess.append(reward)

        done = np.zeros(sess_len, dtype=np.float32)
        if train and sess_mode == "interval":
            end_mask = sess_end[u]
            done[end_mask] = 1.
        if train and neg_sample is not None and num_neg > 0:
            done[-num_neg - 1] = 1.
        else:
            done[-1] = 1.
        done_sess.append(done)

    res = {"user": np.concatenate(user_sess),
           "item": np.concatenate(item_sess, axis=0),
           "reward": np.concatenate(reward_sess),
           "done": np.concatenate(done_sess)}
    return res


def sample_neg_session(items, consumed, n_items, sample_mode):
    size = len(items)
    if size <= 3:
        return items, 0, items

    num_neg = size // 2
    item_sampled = []
    for _ in range(num_neg):
        item_neg = math.floor(n_items * random.random())
        while item_neg in consumed:
            item_neg = math.floor(n_items * random.random())
        item_sampled.append(item_neg)

    if sample_mode == "random":
        indices = np.random.choice(size, num_neg, replace=False)
    else:
        indices = np.arange(size - num_neg, size)
    assert len(indices) == num_neg, "indices and num_neg must equal."
    neg_items = items[indices]
    neg_items[:, -1] = item_sampled
    return np.concatenate([items, neg_items], axis=0), num_neg, items


def assign_reward(sess_len, user, train_flag, train_rewards, test_rewards,
                  train_user_consumed, hist_num, reward_shape):
    reward = np.ones(sess_len, dtype=np.float32)

    if train_flag and train_rewards is not None:
        for label, index in train_rewards[user].items():
            # skip first item as it will never become label
            index = index - 1
            index = index[index >= 0]
            if len(index) > 0:
                reward[index] = reward_shape[label]
    elif (
            not train_flag
            and test_rewards is not None
            and train_rewards is not None
    ):
        train_len = len(train_user_consumed[user])
        train_dummy_reward = np.ones(train_len, dtype=np.float32)
        boundary = (
            hist_num - 1
            if hist_num - 1 < train_len - 1
            else train_len - 1
        )
        for label, index in train_rewards[user].items():
            index = index - 1
            index = index[index >= 0]
            if len(index) > 0:
                train_dummy_reward[index] = reward_shape[label]
        reward[:boundary] = train_dummy_reward[-boundary:]

        if test_rewards[user]:
            for label, index in test_rewards[user].items():
                index = index + boundary
                reward[index] = reward_shape[label]

    return reward


def build_sess_end(data, sess_mode="one", time_col=None, interval=3600):
    if sess_mode == "one":
        sess_end = one_sess_end(data)
    elif sess_mode == "interval":
        sess_end = interval_sess_end(data, time_col, sess_interval=interval)
    else:
        raise ValueError("sess_mode must be 'one' or 'interval'")
    return sess_end


def one_sess_end(data):
    return data.groupby("user").apply(len).to_dict()


# default sess_interval is 3600 secs
def interval_sess_end(data, time_col="time", sess_interval=3600):
    sess_times = data[time_col].astype('int').to_numpy()
    user_split_indices = groupby_user(data.user.to_numpy())
    sess_end = dict()
    for u in range(len(user_split_indices)):
        u_idxs = user_split_indices[u]
        user_ts = sess_times[u_idxs]
        # if neighbor time interval > sess_interval, then end of a session
        sess_end[u] = np.where(np.ediff1d(user_ts) > sess_interval)[0]
    return sess_end


def build_return_session(
        n_users,
        n_items,
        hist_num,
        train_user_consumed,
        test_user_consumed=None,
        train=True,
        gamma=0.99,
        sess_end=None,
        sess_mode="one",
        neg_sample=None,
        train_rewards=None,
        test_rewards=None,
        reward_shape=None
):
    (
        user_sess,
        item_sess,
        return_sess,
        beta_users,
        beta_items,
        beta_labels
    ) = [], [], [], [], [], []
    user_consumed_set = {
        u: set(items) for u, items in train_user_consumed.items()
    }
    for u in range(n_users):
        if train:
            items = np.asarray(train_user_consumed[u])
        else:
            items = np.asarray(
                train_user_consumed[u][-hist_num:] + test_user_consumed[u]
            )

        hist_len = len(items)
        expanded_items = pad_session(hist_len, hist_num, items, pad_val=n_items)

        if hist_len > hist_num:
            full_size_sess = rolling_window(items, hist_num + 1)
            expanded_items = np.concatenate(
                [expanded_items, full_size_sess],
                axis=0
            )

        if train and neg_sample is not None:
            neg_items, num_neg, expanded_items = sample_neg_session(
                expanded_items, user_consumed_set[u], n_items, neg_sample
            )

        sess_len = len(expanded_items)
        user_sess.append(np.tile(u, sess_len))
        item_sess.append(expanded_items)

        if reward_shape is not None:
            reward = assign_reward(
                sess_len, u, train, train_rewards, test_rewards,
                train_user_consumed, hist_num, reward_shape
            )
        else:
            reward = np.ones(sess_len, dtype=np.float32)

        sess_end_u = (
            sess_end[u] + 1
            if train and sess_mode == "interval"
            else None
        )
        return_sess.append(
            compute_returns(reward, gamma, sess_end_u, normalize=False)
        )

        if train and neg_sample is not None and num_neg > 0:
            beta_len = len(neg_items)
            beta_users.append(np.tile(u, beta_len))
            beta_items.append(neg_items[:, :-1])
            beta_labels.append(neg_items[:, -1])

    if train and neg_sample is not None and num_neg > 0:
        res = {"user": np.concatenate(user_sess),
               "item": np.concatenate(item_sess, axis=0),
               "return": np.concatenate(return_sess),
               "beta_user": np.concatenate(beta_users),
               "beta_item": np.concatenate(beta_items, axis=0),
               "beta_label": np.concatenate(beta_labels)}
    else:
        res = {"user": np.concatenate(user_sess),
               "item": np.concatenate(item_sess, axis=0),
               "return": np.concatenate(return_sess)}
    return res
