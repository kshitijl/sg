from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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


tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

inputs = tok(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    out = mod(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (out * mask).sum(1) / mask.sum(1)
    emb = F.normalize(emb, p=2, dim=1)

print(emb @ emb.T)
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

inputs = tok(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    out = mod(**inputs).last_hidden_state  # [batch, seq_len, hidden]
    cls_emb = out[:, 0, :]  # take CLS token (index 0)

print("CLS embedding shape:", cls_emb.shape)
print("First sentence CLS embedding (first 5 dims):", cls_emb[0, :5])
