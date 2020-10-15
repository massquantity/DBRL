import time
import torch
from tqdm import tqdm
from dbrl.data.process import build_batch_data
from dbrl.evaluate import batch_eval, last_eval
from dbrl.utils.info import Collector


def train_model(
        model,
        n_epochs,
        n_rec,
        n_users,
        train_user_consumed,
        test_user_consumed,
        hist_num,
        train_loader,
        eval_loader,
        item_embeds,
        eval_batch_size,
        pad_val=None,
        device=torch.device("cpu"),
        debug=True,
        multi_sample=False,
        repeat_num=20,
        eval_interval=10
):

    assert eval_interval > 0, "eval_interval must be positive."
    print(f"Caution: Will compute loss every {eval_interval} step(s)")
    model_name = model.__class__.__name__.lower()
    if model_name in ("bcq", "ddpg"):
        item_embeds = torch.as_tensor(item_embeds)
        # ignore last item since it's all zero.
        item_embeds[:-1] = item_embeds[:-1] / torch.norm(
            item_embeds[:-1], dim=1, keepdim=True
        )
        item_embeds = item_embeds.to(device)

    train_batch_data = build_batch_data(
        "train", train_user_consumed, hist_num, n_users, pad_val, device
    )
    eval_batch_data = build_batch_data(
        "eval", train_user_consumed, hist_num, n_users, pad_val, device
    )
    train_collector = Collector(model_name)
    eval_collector = Collector(model_name)
    for epoch in range(1, n_epochs + 1):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\nEpoch {epoch} start-time: {start_time}\n")
        model.train()
        train_step = 0
        for batch in tqdm(train_loader, desc="train"):
            train_step += 1
            if device is not None:
                batch = {k: v.to(device) for k, v in batch.items()}
            info = model.update(batch)
            if (
                    debug
                    and train_step % eval_interval == 0
                    and info.get("action", 1) is not None
            ):
                info.update(
                    batch_eval(info, n_rec, batch["action"],
                               train_user_consumed, batch["user"],
                               item_embeds, model_name, model)
                )
                train_collector.gather_info(info)

        if debug:
            train_collector.metrics["ndcg"] = last_eval(
                model, train_batch_data, train_user_consumed,
                test_user_consumed, n_users, n_rec, item_embeds,
                eval_batch_size, "train", model_name, multi_sample,
                repeat_num
            )
            train_collector.print_and_clear_info()

        with torch.no_grad():
            print("\n" + "*" * 20 + " EVAL " + "*" * 20)
            model.eval()
            eval_step = 0
            for batch in tqdm(eval_loader, desc="eval"):
                eval_step += 1
                if device is not None:
                    batch = {k: v.to(device) for k, v in batch.items()}
                eval_info = model.compute_loss(batch)
                if (
                        debug
                        and eval_step % eval_interval == 0
                        and eval_info.get("action", 1) is not None
                ):
                    eval_info.update(
                        batch_eval(eval_info, n_rec, batch["action"],
                                   test_user_consumed, batch["user"],
                                   item_embeds, model_name, model)
                    )
                    eval_collector.gather_info(eval_info)

            if debug:
                eval_collector.metrics["ndcg"] = last_eval(
                    model, eval_batch_data, train_user_consumed,
                    test_user_consumed, n_users, n_rec, item_embeds,
                    eval_batch_size, "eval", model_name, multi_sample,
                    repeat_num
                )
                eval_collector.print_and_clear_info()
        print("=" * 80)




