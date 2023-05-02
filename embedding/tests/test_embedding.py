import unittest
from embedding.qa import QA
from embedding.recommend import Recommend


class TestEmbedding(unittest.TestCase):
    def test_qa(self):
        qa = QA()
        sims = qa.inquire("Is Kaggle dead?")
        print(sims)

    def test_recommend(self):
        rd = Recommend()
        rsp = rd.run()
        print(rsp)








