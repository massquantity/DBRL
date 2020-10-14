import json
import os
import numpy as np


def save_npy(user_embeds, item_embeds, path):
    user_path = os.path.join(path, "tianchi_user_embeddings.npy")
    item_path = os.path.join(path, "tianchi_item_embeddings.npy")
    with open(user_path, "wb") as f:
        np.save(f, user_embeds)
    with open(item_path, "wb") as f:
        np.save(f, item_embeds)


def save_json(user_map, item_map, user_embeds, item_embeds, path):
    with open(os.path.join(path, "user_map.json"), "w") as f:
        json.dump(user_map, f, separators=(',', ':'))
    with open(os.path.join(path, "item_map.json"), "w") as f:
        json.dump(item_map, f, separators=(',', ':'))
    embeds = dict()
    embeds["user"] = user_embeds.tolist()
    embeds["item"] = item_embeds.tolist()
    with open(os.path.join(path, "embeddings.json"), "w") as f:
        json.dump(embeds, f, separators=(',', ':'))
