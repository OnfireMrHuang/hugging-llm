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

import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity


class QA:
    df = None
    vec_base = []

    def __init__(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_base = "https://api.openai-proxy.com/v1"
        df = pd.read_csv("../../dataset/Kaggle related questions on Qoura - Questions.csv")
        for v in df.head().itertuples():
            emb = get_embedding(v.Questions)
            im = {
                "question": v.Questions,
                "embedding": emb,
                "answer": v.Link
            }
            self.vec_base.append(im)

    def inquire(self, query: str):
        emb = get_embedding(query)
        arr = np.array([v.get("embedding") for v in self.vec_base])
        q_arr = np.expand_dims(emb, 0)
        sims = sk_cosine_similarity(arr, q_arr)
        # sims = [cosine_similarity(emb, v.get("embedding")) for v in self.vec_base]
        index = 0
        for i, sim in enumerate(sims):
            if sim > sims[index]:
                index = i
        return self.vec_base[index]["answer"]



