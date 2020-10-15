import os
import sys
sys.path.append(os.pardir)
import warnings
warnings.filterwarnings("ignore")
import argparse
from pprint import pprint
import numpy as np
import torch
from torch.optim import Adam
from dbrl.data import process_data
from dbrl.data import build_dataloader
from dbrl.models.youtube_topk import Reinforce
from dbrl.network import PolicyPi, Beta
from dbrl.trainer import train_model
from dbrl.utils import count_vars, init_param


def parse_args():
    parser = argparse.ArgumentParser(description="run_reinforce")
    parser.add_argument("--data", type=str, default="tianchi.csv")
    parser.add_argument("--user_embeds", type=str,
                        default="tianchi_user_embeddings.npy")
    parser.add_argument("--item_embeds", type=str,
                        default="tianchi_item_embeddings.npy")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--hist_num", type=int, default=10,
                        help="num of history items to consider")
    parser.add_argument("--n_rec", type=int, default=10,
                        help="num of items to recommend")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sess_mode", type=str, default="interval",
                        help="Specify when to end a session")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("A list all args: \n======================")
    pprint(vars(args))
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH = os.path.join("resources", args.data)
    with open(os.path.join("resources", args.user_embeds), "rb") as f:
        user_embeddings = np.load(f)
    with open(os.path.join("resources", args.item_embeds), "rb") as f:
        item_embeddings = np.load(f)
    item_embeddings[-1] = 0.   # last item is used for padding

    n_epochs = args.n_epochs
    hist_num = args.hist_num
    batch_size = eval_batch_size = args.batch_size
    embed_size = item_embeddings.shape[1]
    hidden_size = args.hidden_size
    input_dim = embed_size * (hist_num + 1)
    action_dim = len(item_embeddings)
    policy_lr = args.lr
    beta_lr = args.lr
    weight_decay = args.weight_decay
    gamma = args.gamma
    n_rec = args.n_rec
    pad_val = len(item_embeddings) - 1
    sess_mode = args.sess_mode
    debug = True
    one_hour = int(60 * 60)
    reward_map = {"pv": 1., "cart": 2., "fav": 2., "buy": 3.}
    columns = ["user", "item", "label", "time", "sex", "age", "pur_power",
               "category", "shop", "brand"]

    cutoffs = [
        len(item_embeddings) // 20,
        len(item_embeddings) // 10,
        len(item_embeddings) // 3
    ]

    (
        n_users,
        n_items,
        train_user_consumed,
        test_user_consumed,
        train_sess_end,
        test_sess_end,
        train_rewards,
        test_rewards
    ) = process_data(PATH, columns, 0.2, time_col="time", sess_mode=sess_mode,
                     interval=one_hour, reward_shape=reward_map)

    train_loader, eval_loader = build_dataloader(
        n_users,
        n_items,
        hist_num,
        train_user_consumed,
        test_user_consumed,
        batch_size,
        sess_mode=sess_mode,
        train_sess_end=train_sess_end,
        test_sess_end=test_sess_end,
        n_workers=0,
        compute_return=True,
        neg_sample=False,
        train_rewards=train_rewards,
        test_rewards=test_rewards,
        reward_shape=reward_map
    )

    policy = PolicyPi(
        input_dim, action_dim, hidden_size, user_embeddings,
        item_embeddings, None, pad_val, 1, device
    ).to(device)
    beta = Beta(input_dim, action_dim, hidden_size).to(device)
    init_param(policy, beta)

    policy_optim = Adam(policy.parameters(), policy_lr, weight_decay=weight_decay)
    beta_optim = Adam(beta.parameters(), beta_lr, weight_decay=weight_decay)

    model = Reinforce(
        policy,
        policy_optim,
        beta,
        beta_optim,
        hidden_size,
        gamma,
        k=10,
        weight_clip=2.0,
        offpolicy_correction=True,
        topk=True,
        adaptive_softmax=False,
        cutoffs=cutoffs,
        device=device,
    )

    var_counts = tuple(count_vars(module) for module in [policy, beta])
    print(f'Number of parameters: policy: {var_counts[0]}, '
          f' beta: {var_counts[1]}')

    train_model(
        model,
        n_epochs,
        n_rec,
        n_users,
        train_user_consumed,
        test_user_consumed,
        hist_num,
        train_loader,
        eval_loader,
        item_embeddings,
        eval_batch_size,
        pad_val,
        device,
        debug=debug,
        eval_interval=10
    )

    torch.save(policy.state_dict(), "resources/model_reinforce.pt")
    print("train and save done!")
