class TestEmbedding(unittest.TestCase):
    def __init__(self):
        pass

    def test_embedding(self):
        rng = np.random.default_rng(42)
        # 词表大小设置N=16,维度D=256
        table = rng.uniform(size=(16, 256))
        table.shape
        self.assertEqual(table.shape, (16, 256))


