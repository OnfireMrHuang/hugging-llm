import unittest
import os
import numpy as np
from embedding.embeding import OpenAiEmbedding, compare_similarity


class TestEmbedding(unittest.TestCase):
    def test_gen_256_dim_embedding(self):
        rng = np.random.default_rng(42)
        # 词表大小N=16,维度D=256
        table = rng.uniform(size=(16, 256))
        print("shape: ")
        print(table.shape)
        print("table: ")
        print(table)

    def test_openai_embedding(self):
        emb = OpenAiEmbedding(os.environ.get("OPENAI_API_KEY"))
        rsp = emb.get_embedding("我喜欢你")
        print("rsp: ")
        print(rsp)

    def test_embedding_similarity(self):
        emb = OpenAiEmbedding(os.environ.get("OPENAI_API_KEY"))
        rsp1 = emb.get_embedding("我喜欢你")
        rsp2 = emb.get_embedding("我爱你")
        print("rsp1: ")
        print(rsp1)
        print("rsp2: ")
        print(rsp2)
        print("similarity: ")
        print(compare_similarity(rsp1, rsp2))

