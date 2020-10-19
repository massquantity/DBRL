import os
import sys
sys.path.append(os.pardir)
import warnings
warnings.filterwarnings("ignore")
import argparse
import time
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="run_prepare_data")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bucket_age(age):
    if age < 30:
        return 1
    elif age < 40:
        return 2
    elif age < 50:
        return 3
    else:
        return 4


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    np.random.seed(args.seed)
    start_time = time.perf_counter()

    user_feat = pd.read_csv("resources/user.csv", header=None,
                            names=["user", "sex", "age", "pur_power"])
    item_feat = pd.read_csv("resources/item.csv", header=None,
                            names=["item", "category", "shop", "brand"])
    behavior = pd.read_csv("resources/user_behavior.csv", header=None,
                           names=["user", "item", "behavior", "time"])

    behavior = behavior.sort_values(by="time").reset_index(drop=True)
    behavior = behavior.drop_duplicates(subset=["user", "item", "behavior"])

    user_counts = behavior.groupby("user")[["user"]].count().rename(
        columns={"user": "count_user"}
    ).sort_values("count_user", ascending=False)
    # sample users with short and long sequences
    short_users = np.array(
        user_counts[
            (user_counts.count_user > 5) & (user_counts.count_user <= 50)
        ].index
    )
    long_users = np.array(
        user_counts[
            (user_counts.count_user > 50) & (user_counts.count_user <= 200)
        ].index
    )
    short_chosen_users = np.random.choice(short_users, 60000, replace=False)
    long_chosen_users = np.random.choice(long_users, 20000, replace=False)
    chosen_users = np.concatenate([short_chosen_users, long_chosen_users])

    behavior = behavior[behavior.user.isin(chosen_users)]
    print(f"n_users: {behavior.user.nunique()}, "
          f"n_items: {behavior.item.nunique()}, "
          f"behavior length: {len(behavior)}")

    # merge with all features
    behavior = behavior.merge(user_feat, on="user")
    behavior = behavior.merge(item_feat, on="item")
    behavior["age"] = behavior["age"].apply(bucket_age)
    behavior = behavior.sort_values(by="time").reset_index(drop=True)
    behavior.to_csv("resources/tianchi.csv", header=None, index=False)
    print(f"prepare data done!, "
          f"time elapsed: {(time.perf_counter() - start_time):.2f}")
