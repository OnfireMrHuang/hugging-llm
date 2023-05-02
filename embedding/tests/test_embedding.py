import unittest
import os
import numpy as np
from embedding.embeding import Embedding


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
        emb = Embedding(os.environ.get("OPENAI_API_KEY"))
        rsp = emb.request("我喜欢你")
        print("rsp: ")
        print(rsp)


