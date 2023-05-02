

# 聚类的意思是把彼此相近的样本聚集在一起，本质也是在使用一种表示和相似度衡量来处理文本。比如我们有大量的未分类文本，如果事先能知道有几种类别，就可以用聚类的方法先将样本大致分一下。

#
#
# QA是问答的意思，Q表示Question，A表示Answer，QA是NLP非常基础和常用的任务。简单来说，就是当用户提出一个问题时，我们能从已有的问题库中找到一个最相似的，并把它的答案返回给用户。这里有两个关键点：
#
# 事先需要有一个QA库。
# 用户提问时，系统要能够在QA库中找到一个最相似的。
# ChatGPT（或生成方式）做这类任务相对有点麻烦，尤其是当：
#
# QA库非常庞大时
# 给用户的答案是固定的，不允许自由发挥时
# 生成方式做起来是事倍功半。但是Embedding确实天然的非常适合，因为该任务的核心就是在一堆文本中找出给定文本最相似的。简单来说，其实就是个相似度计算问题
# 这里，我们就把link当作答案构造数据对。基本的流程如下:
# 1、对每个question计算embedding
# 2、存储embedding,同时存储每个Question对应的答案
# 3、从存储的地方检索相似的Question
# matplotlib inline
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class Clustering:
    cdf = None

    def __init__(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_base = "https://api.openai-proxy.com/v1"
        df = pd.read_csv("dataset/DBPEDIA_val.csv")
        # sdf = df.sample(200)
        cdf = df[
            (df.l1 == "Place") | (df.l1 == "Work") | (df.l1 == "Species")
        ]
        cdf["embedding"] = cdf.text.apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))
        arr = np.array(cdf.embedding.to_list())
        pca = PCA(n_components=3)
        vis_dims = pca.fit_transform(arr)
        cdf["embed_vis"] = vis_dims.tolist()
        self.cdf = cdf

    def classification(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
        cmap = plt.get_cmap("tab20")
        categories = sorted(self.cdf.l1.unique())

        # 分别绘制每个类别
        for i, cat in enumerate(categories):
            sub_matrix = np.array(self.cdf[self.cdf.l1 == cat]["embed_vis"].to_list())
            x = sub_matrix[:, 0]
            y = sub_matrix[:, 1]
            z = sub_matrix[:, 2]
            colors = [cmap(i / len(categories))] * len(sub_matrix)
            ax.scatter(x, y, z, c=colors, label=cat)

        ax.legend(bbox_to_anchor=(1.2, 1))
        plt.show()



