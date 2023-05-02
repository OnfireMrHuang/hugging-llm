import openai


# from openai.embeddings_utils import cosine_similarity


def compare_similarity(emb1, emb2):
    pass
    # return cosine_similarity(emb1, emb2)


class OpenAiEmbedding:
    open_api_key = ""
    model = "text-embedding-ada-002"

    def __init__(self, open_api_key):
        self.open_api_key = open_api_key
        openai.api_key = self.open_api_key
        openai.api_base = "https://api.openai-proxy.com/v1"

    # 获取文本的embedding数值
    def get_embedding(self, text):
        # 请求远端内容，获取响应
        emb_req = openai.Embedding.create(input=[text], model=self.model)
        emb = emb_req.data[0].embedding
        return emb

    def dialogue(self, content):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": content}]
        )
        return response.get("choices")[0].get("message").get("content")
