import unittest
import os
import numpy as np
from embedding.qa import QA


class TestEmbedding(unittest.TestCase):
    def test_qa(self):
        qa = QA()
        sims = qa.inquire("Is Kaggle dead?")
        print(sims)
        # sims = qa.batch_inquire(["What are some alternatives to Kaggle?", "How to use openai?"])
        # print(sims)




