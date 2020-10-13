import torch
import torch.nn as nn
import torch.nn.functional as F


class DSSM(nn.Module):
    def __init__(
            self,
            main_embed_size,
            feat_embed_size,
            n_users,
            n_items,
            hidden_size,
            feat_map,
            static_feat,
            dynamic_feat,
            use_bn=True
    ):
        super(DSSM, self).__init__()
        self.total_feat = static_feat + dynamic_feat
        self.embed_user = nn.Embedding(n_users + 1, main_embed_size,
                                       padding_idx=n_users)
        self.embed_item = nn.Embedding(n_items + 1, main_embed_size,
                                       padding_idx=n_items)
        self.embed_feat = nn.ModuleDict({
            feat: nn.Embedding(feat_map[feat + "_vocab"] + 1, feat_embed_size,
                               padding_idx=feat_map[feat + "_vocab"])
            for feat in self.total_feat
        })

        self.static_feat = static_feat
        self.dynamic_feat = dynamic_feat
        input_dim_user = main_embed_size + feat_embed_size * len(static_feat)
        self.fcu1 = nn.Linear(input_dim_user, hidden_size[0])
        self.fcu2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fcu3 = nn.Linear(hidden_size[1], main_embed_size)

        input_dim_item = main_embed_size + feat_embed_size * len(dynamic_feat)
        self.fci1 = nn.Linear(input_dim_item, hidden_size[0])
        self.fci2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fci3 = nn.Linear(hidden_size[1], main_embed_size)

        self.use_bn = use_bn
        if use_bn:
            self.bnu1 = nn.BatchNorm1d(hidden_size[0])
            self.bnu2 = nn.BatchNorm1d(hidden_size[1])
            self.bni1 = nn.BatchNorm1d(hidden_size[0])
            self.bni2 = nn.BatchNorm1d(hidden_size[1])

    def get_embedding(self, data):
        user_part = [self.embed_user(data["user"])]
        for feat in self.static_feat:
            embedding = self.embed_feat[feat]
            user_part.append(embedding(data[feat]))

        user_part = torch.cat(user_part, dim=1)
        out_user = self.fcu1(user_part)
        if self.use_bn:
            out_user = self.bnu1(out_user)
        out_user = F.relu(out_user)
        out_user = self.fcu2(out_user)
        if self.use_bn:
            out_user = self.bnu2(out_user)
        out_user = F.relu(out_user)
        out_user = self.fcu3(out_user)
        out_user = out_user / torch.norm(out_user, dim=1, keepdim=True)

        item_part = [self.embed_item(data["item"])]
        for feat in self.dynamic_feat:
            embedding = self.embed_feat[feat]
            item_part.append(embedding(data[feat]))

        item_part = torch.cat(item_part, dim=1)
        out_item = self.fci1(item_part)
        if self.use_bn:
            out_item = self.bni1(out_item)
        out_item = F.relu(out_item)
        out_item = self.fci2(out_item)
        if self.use_bn:
            out_item = self.bni2(out_item)
        out_item = F.relu(out_item)
        out_item = self.fci3(out_item)
        out_item = out_item / torch.norm(out_item, dim=1, keepdim=True)
        return out_user, out_item

    def forward(self, data):
        out_user, out_item = self.get_embedding(data)
        out = torch.sum(torch.mul(out_user, out_item), dim=1).squeeze()
        return out_user, out_item, out
