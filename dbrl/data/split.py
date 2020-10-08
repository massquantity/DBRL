import numpy as np


def split_by_ratio(data, shuffle=False, test_size=None, pad_unknown=True,
                   filter_unknown=False, seed=42):
    np.random.seed(seed)
    n_users = data.user.nunique()
    user_indices = data.user.to_numpy()
    user_split_indices = groupby_user(user_indices)

    split_indices_all = [[], []]
    for u in range(n_users):
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len <= 3:   # keep items of rare users in trainset
            split_indices_all[0].extend(u_data)
        else:
            train_threshold = round((1 - test_size) * u_data_len)
            split_indices_all[0].extend(list(u_data[:train_threshold]))
            split_indices_all[1].extend(list(u_data[train_threshold:]))

    if shuffle:
        split_data_all = tuple(
            np.random.permutation(data[idx]) for idx in split_indices_all
        )
    else:
        split_data_all = list(data.iloc[idx] for idx in split_indices_all)

    if pad_unknown:
        split_data_all = _pad_unknown_item(split_data_all)
    elif filter_unknown:
        split_data_all = _filter_unknown_user_item(split_data_all)
    return split_data_all


def groupby_user(user_indices):
    users, user_position, user_counts = np.unique(user_indices,
                                                  return_inverse=True,
                                                  return_counts=True)
    user_split_indices = np.split(np.argsort(user_position, kind="mergesort"),
                                  np.cumsum(user_counts)[:-1])
    return user_split_indices


def _filter_unknown_user_item(data_list):
    train_data, test_data = data_list
    unique_values = dict(user=set(train_data.user.tolist()),
                         item=set(train_data.item.tolist()))

    print(f"test data size before filtering: {len(test_data)}")
    out_of_bounds_row_indices = set()
    for col in ["user", "item"]:
        for j, val in enumerate(test_data[col]):
            if val not in unique_values[col]:
                out_of_bounds_row_indices.add(j)

    mask = np.arange(len(test_data))
    test_data_clean = test_data[~np.isin(mask, list(out_of_bounds_row_indices))]
    print(f"test data size after filtering: {len(test_data_clean)}")
    return train_data, test_data_clean


def _pad_unknown_item(data_list):
    train_data, test_data = data_list
    n_items = train_data.item.nunique()
    unique_items = set(train_data.item.tolist())
    test_data.loc[~test_data.item.isin(unique_items), "item"] = n_items
    return train_data, test_data
