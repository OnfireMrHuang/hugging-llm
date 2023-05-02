
import openai


class Embedding:
    open_api_key = ""
    model = "text-embedding-ada-002"

    def __init__(self, open_api_key):
        self.open_api_key = open_api_key
        openai.api_key = self.open_api_key
        openai.api_base = "https://api.openai-proxy.com/v1"

    def request(self, text):
        # 请求远端内容，获取响应
        emb_req = openai.Embedding.create(input=[text], model=self.model)
        emb = emb_req.data[0].embedding
        return emb

