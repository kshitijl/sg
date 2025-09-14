from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print(tok(sentences, padding=True, truncation=True)["input_ids"])
