# 我们在很多APP或网站上都能看到推荐功能。
# 比如在购物网站，每当你登陆或者选购一件商品后，系统就会给你推荐一些相关的产品。
# 在这一小节中，我们就来做一个类似的应用，不过我们推荐的不是商品，而是文本，比如帖子、文章、新闻等。

# 我们以新闻为例，先说一下基本逻辑：
#
# 首先要有一个基础的文章库，可能包括标题、内容、标签等。
# 计算已有文章的Embedding并存储。
# 根据用户浏览记录，推荐和浏览记录最相似的文章

from dataclasses import dataclass
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
from typing import List
import openai
import os
import random
import numpy as np


@dataclass
class User:
    user_name: str


@dataclass
class UserPrefer:
    user_name: str
    prefers: List[int]


@dataclass
class Item:
    item_id: str
    item_props: dict


@dataclass
class Action:
    action_type: str
    action_props: dict


@dataclass
class UserAction:
    user: User
    item: Item
    action: Action
    action_time: str


u1 = User("u1")
up1 = UserPrefer("u1", [1, 2])
# sdf.iloc[1] 正好是sport（类别为2）
i1 = Item("i1", {
    "id": 1,
    "catetory": "sport",
    "title": "Swimming: Shibata Joins Japanese Gold Rush",
    "description": "\
    ATHENS (Reuters) - Ai Shibata wore down French teen-ager  Laure Manaudou to win the women's 800 meters \
    freestyle gold  medal at the Athens Olympics Friday and provide Japan with  their first female swimming \
    champion in 12 years.",
    "content": "content"
})
a1 = Action("浏览", {
    "open_time": "2023-04-01 12:00:00",
    "leave_time": "2023-04-01 14:00:00",
    "type": "close",
    "duration": "2hour"
})
ua1 = UserAction(u1, i1, a1, "2023-04-01 12:00:00")


class Recall:
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def user_prefer_recall(self, user, n):
        up = self.get_user_prefers(user)
        idx = random.randrange(0, len(up.prefers))
        return self.pick_by_idx(idx, n)

    def hot_recall(self, n):
        # 随机进行示例
        df = self.data.sample(n)
        return df

    def user_action_recall(self, user, n):
        actions = self.get_user_actions(user)
        interest = self.get_most_interested_item(actions)
        recoms = self.recommend_by_interest(interest, n)
        return recoms

    def get_most_interested_item(self, user_action):
        """
        可以选近一段时间内用户交互时间、次数、评论（相关属性）过的Item
        """
        # 就是sdf的第2行，idx为1的那条作为最喜欢（假设）
        # 是一条游泳相关的Item
        idx = user_action.item.item_props["id"]
        im = self.data.iloc[idx]
        return im

    def recommend_by_interest(self, interest, n):
        cate_id = interest["Class Index"]
        q_emb = interest["embedding"]
        # 确定类别
        base = self.data[self.data["Class Index"] == cate_id]
        # 此处可以复用QA那一段代码，用给定embedding计算base中embedding的相似度
        base_arr = np.array(
            [v.embedding for v in base.itertuples()]
        )
        q_arr = np.expand_dims(q_emb, 1)
        sims = cosine_similarity(base_arr, q_arr)
        # 排除掉自己
        idxes = sims.argsort(0).squeeze()[-(n + 1):-1]
        return base.iloc[reversed(idxes.tolist())]

    def pick_by_idx(self, category, n):
        df = self.data[self.data["Class Index"] == category]
        return df.sample(n)

    def get_user_actions(self, user):
        dct = {"u1": ua1}
        return dct[user.user_name]

    def get_user_prefers(self, user):
        dct = {"u1": up1}
        return dct[user.user_name]

    def run(self, user):
        ur = self.user_action_recall(user, 5)
        if len(ur) == 0:
            ur = self.user_prefer_recall(user, 5)
        hr = self.hot_recall(3)
        return pd.concat([ur, hr], axis=0)



class Recommend:
    sdf = None

    def __init__(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_base = "https://api.openai-proxy.com/v1"
        df = pd.read_csv("../../dataset/AG_News.csv")
        sdf = df.sample(10) # 采样10条数据
        sdf["embedding"] = sdf.apply(lambda x:
                                     get_embedding(x.Title + x.Description, engine="text-embedding-ada-002"), axis=1)
        self.sdf = sdf

    def run(self):
        recall = Recall(self.sdf)
        return recall.run(u1)
