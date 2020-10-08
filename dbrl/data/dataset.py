from torch.utils.data import Dataset, DataLoader
from .session import build_session, build_return_session


class RLDataset(Dataset):
    def __init__(self, data, has_return=False):
        self.data = data
        self.has_return = has_return

    def __getitem__(self, index):
        user = self.data["user"][index]
        items = self.data["item"][index]
        if not self.has_return:
            res = {"user": user,
                   "item": items[:-1],
                   "action": items[-1],
                   "reward": self.data["reward"][index],
                   "done": self.data["done"][index],
                   "next_item": items[1:]}
        else:
            if "beta_label" in self.data:
                res = {"user": user,
                       "item": items[:-1],
                       "action": items[-1],
                       "return": self.data["return"][index],
                       "beta_user": self.data["beta_user"][index],
                       "beta_item": self.data["beta_item"][index],
                       "beta_label": self.data["beta_label"][index]}
            else:
                res = {"user": user,
                       "item": items[:-1],
                       "action": items[-1],
                       "return": self.data["return"][index]}
        return res

    def __len__(self):
        return len(self.data["item"])


class EvalRLDataset(Dataset):
    def __init__(self, data, has_return=False):
        self.data = data
        self.has_return = has_return

    def __getitem__(self, index):
        user = self.data["user"][index]
        items = self.data["item"][index]
        if not self.has_return:
            res = {"user": user,
                   "item": items[:-1],
                   "action": items[-1],
                   "reward": self.data["reward"][index],
                   "done": self.data["done"][index],
                   "next_item": items[1:]}
        else:
            if "beta_label" in self.data:
                res = {"user": user,
                       "item": items[:-1],
                       "action": items[-1],
                       "return": self.data["return"][index],
                       "beta_user": self.data["beta_user"][index],
                       "beta_item": self.data["beta_item"][index],
                       "beta_label": self.data["beta_label"][index]}
            else:
                res = {"user": user,
                       "item": items[:-1],
                       "action": items[-1],
                       "return": self.data["return"][index]}
        return res

    def __len__(self):
        return len(self.data["item"])


def build_dataloader(
        n_users,
        n_items,
        hist_num,
        train_user_consumed,
        test_user_consumed,
        batch_size,
        sess_mode="one",
        train_sess_end=None,
        test_sess_end=None,
        n_workers=0,
        compute_return=False,
        neg_sample=None,
        train_rewards=None,
        test_rewards=None,
        reward_shape=None
):
    """Construct DataLoader for pytorch model.

    Parameters
    ----------
    n_users : int
        Number of users.
    n_items : int
        Number of items.
    hist_num : int
        A fixed number of history items that a user interacted. If a user has
        interacted with more than `hist_num` items, the front items will be
        truncated.
    train_user_consumed : dict
        Items interacted by each user in train data.
    test_user_consumed : dict
        Items interacted by each user in test data.
    batch_size : int
        How many samples per batch to load.
    sess_mode : str
        Ways of representing a session.
    train_sess_end : dict
        Session end mark for each user in train data.
    test_sess_end : dict
        Session end mark for each user in test data.
    n_workers : int
        How many subprocesses to use for data loading.
    compute_return : bool
        Whether to use compute_return session mode.
    neg_sample : str (default None)
        Whether to sample negative samples during training, also specify
        sample mode.
    train_rewards : dict (default None)
        A dict for mapping train users to rewards.
    test_rewards : dict (default None)
        A dict for mapping test users to rewards.
    reward_shape : dict (default None)
        A dict for mapping labels to rewards.

    Returns
    -------
    train_rl_loader : DataLoader
        Train dataloader for training.
    test_rl_loader : DataLoader
        Test dataloader for testing.
    """

    if not compute_return:
        train_session = build_session(
            n_users,
            n_items,
            hist_num,
            train_user_consumed,
            test_user_consumed,
            train=True,
            sess_end=train_sess_end,
            sess_mode=sess_mode,
            neg_sample=neg_sample,
            train_rewards=train_rewards,
            test_rewards=test_rewards,
            reward_shape=reward_shape
        )
        test_session = build_session(
            n_users,
            n_items,
            hist_num,
            train_user_consumed,
            test_user_consumed,
            train=False,
            sess_end=test_sess_end,
            sess_mode=sess_mode,
            neg_sample=None,
            train_rewards=train_rewards,
            test_rewards=test_rewards,
            reward_shape=reward_shape
        )
        train_rl_data = RLDataset(train_session)
        test_rl_data = EvalRLDataset(test_session)

    else:
        train_session = build_return_session(
            n_users,
            n_items,
            hist_num,
            train_user_consumed,
            test_user_consumed,
            train=True,
            gamma=0.99,
            neg_sample=neg_sample,
            train_rewards=train_rewards,
            test_rewards=test_rewards,
            reward_shape=reward_shape,
            sess_end=train_sess_end,
            sess_mode=sess_mode,
        )
        test_session = build_return_session(
            n_users,
            n_items,
            hist_num,
            train_user_consumed,
            test_user_consumed,
            train=False,
            gamma=0.99,
            neg_sample=None,
            train_rewards=train_rewards,
            test_rewards=test_rewards,
            reward_shape=reward_shape,
            sess_end=test_sess_end,
            sess_mode=sess_mode,
        )
        train_rl_data = RLDataset(train_session, has_return=True)
        test_rl_data = EvalRLDataset(test_session, has_return=True)

    train_rl_loader = DataLoader(train_rl_data, batch_size=batch_size,
                                 shuffle=True, num_workers=n_workers)
    test_rl_loader = DataLoader(test_rl_data, batch_size=batch_size,
                                shuffle=False, num_workers=n_workers)
    return train_rl_loader, test_rl_loader


class FeatDataset(Dataset):
    def __init__(self, user_indices, item_indices, labels, feat_map=None,
                 static_feat=None, dynamic_feat=None):
        self.users = user_indices
        self.items = item_indices
        self.labels = labels
        self.feat_map = feat_map
        self.static_feat = static_feat
        self.dynamic_feat = dynamic_feat

    def __getitem__(self, index):
        user, item, label = (
            self.users[index], self.items[index], self.labels[index]
        )
        data = {"user": user, "item": item, "label": label}
        if self.static_feat is not None:
            for feat in self.static_feat:
                data[feat] = self.feat_map[feat][user]
        if self.dynamic_feat is not None:
            for feat in self.dynamic_feat:
                data[feat] = self.feat_map[feat][item]
        return data

    def __len__(self):
        return len(self.labels)
