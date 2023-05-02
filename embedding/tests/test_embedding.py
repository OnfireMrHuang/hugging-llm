import unittest
from embedding.qa import QA
from embedding.clustering import Clustering


class TestEmbedding(unittest.TestCase):
    def test_qa(self):
        qa = QA()
        sims = qa.inquire("Is Kaggle dead?")
        print(sims)








