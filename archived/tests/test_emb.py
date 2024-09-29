import numpy as np
from llama_index.core import Settings
from llama_index.embeddings.upstage import UpstageEmbedding
from dotenv import load_dotenv

load_dotenv()

Settings.embed_model = UpstageEmbedding(model='solar-embedding-1-large')

text = "Jeju Olle Trails"
texts = [
    "Jeju Olle Trails",
    "Seongsan Ilchulbong",
    "Cheonjeyeon Waterfall"
]

text_embed = Settings.embed_model.get_text_embedding(text)
texts_embed = Settings.embed_model.get_text_embedding_batch(texts)

text_embed_np = np.array(text_embed)
texts_embed_np = np.array(texts_embed)

print(text_embed_np.shape)    # (4096,)
print(texts_embed_np.shape)   # (3, 4096)
