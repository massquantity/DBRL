# DBRL

### [`English`](https://github.com/massquantity/DBRL)  &nbsp;  `简体中文`

<br>

DBRL 是一个用于训练强化学习推荐模型的工具。DBRL 意为：Dataset Batch Reinforcement Learning，和传统强化学习的训练不同，DBRL 中只使用静态数据来训练模型，而不与环境作任何进一步的交互。详情可参阅 [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/pdf/2005.01643.pdf) 。

训练完成后的模型可用于在线服务，本仓库的在线部分位于 [FlinkRL](https://github.com/massquantity/flink-reinforcement-learning) ，主要使用 Flink 和训练后的强化学习模型作在线推荐。下面是整个系统架构：

![](https://s1.ax1x.com/2020/10/19/0x5JAI.png)



## 算法

`DBRL` 目前提供三种算法:

+ REINFORCE ([YouTube top-k off-policy](https://arxiv.org/pdf/1812.02353.pdf))
+ Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/pdf/1509.02971.pdf))
+ Batch Constrained Deep Q-Learning ([BCQ](https://arxiv.org/pdf/1812.02900.pdf))



## 数据

数据来源于天池的一个比赛，详情可参阅[官方网站](https://tianchi.aliyun.com/competition/entrance/231721/information?lang=zh-cn) ，注意这里只是用了第二轮的数据。也可以从 [Google Drive](https://drive.google.com/file/d/1erBjYEOa7IuOIGpI8pGPn1WNBAC4Rv0-/view?usp=sharing) 下载。



## 使用步骤

依赖库： python>=3.6, numpy, pandas, torch>=1.3, tqdm.

```shell
$ git clone https://github.com/massquantity/DBRL.git
```

下载完数据后，解压并放到 `DBRL/dbrl/resources` 文件夹中。原始数据有三张表：`user.csv`, `item.csv`, `user_behavior.csv` 。首先用脚本 `run_prepare_data.py` 过滤掉一些行为太少的用户并将所有特征合并到一张表。接着用 `run_pretrain_embeddings.py` 为每个用户和物品预训练 embedding：

```shell
$ cd DBRL/dbrl
$ python run_prepare_data.py
$ python run_pretrain_embeddings.py --lr 0.001 --n_epochs 4
```

可以调整一些参数如 `lr` 和 `n_epochs`  来获得更好的评估效果。接下来开始训练模型，现在在 `DBRL` 中有三种模型，任选一种即可：

```shell
$ python run_reinforce.py --n_epochs 5 --lr 1e-5
$ python run_ddpg.py --n_epochs 5 --lr 1e-5
$ python run_bcq.py --n_epochs 5 --lr 1e-5
```

这样 `DBRL/resources` 中应该至少有 6 个文件：

+ `model_xxx.pt`, 训练好的 PyTorch 模型。
+ `tianchi.csv`, 转换过的数据集。
+ `tianchi_user_embeddings.npy`,  `npy` 格式的 user 预训练 embedding。
+ `tianchi_item_embeddings.npy`,  `npy` 格式的 item 预训练 embedding。
+ `user_map.json`,  将原始用户 id 映射到模型中 id 的 json 文件。
+ `item_map.json`,  将原始物品 id 映射到模型中 id 的 json 文件。





