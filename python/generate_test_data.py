from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="ea78891063587eb050ed4166b20062eaf978037c",
)

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
print(f"{similarities[0, 1]:.6f} {similarities[0, 2]:.6f} {similarities[1, 2]:.6f}")

tok = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="ea78891063587eb050ed4166b20062eaf978037c",
)
print(tok(sentences, padding=True, truncation=True)["input_ids"])


tok = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="ea78891063587eb050ed4166b20062eaf978037c",
)
mod = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="ea78891063587eb050ed4166b20062eaf978037c",
)

inputs = tok(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    out = mod(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (out * mask).sum(1) / mask.sum(1)
    emb = F.normalize(emb, p=2, dim=1)

print(emb @ emb.T)
tok = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="ea78891063587eb050ed4166b20062eaf978037c",
)
mod = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="ea78891063587eb050ed4166b20062eaf978037c",
)

inputs = tok(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    out = mod(**inputs).last_hidden_state  # [batch, seq_len, hidden]
    cls_emb = out[:, 0, :]  # take CLS token (index 0)

print("CLS embedding shape:", cls_emb.shape)
print("First sentence CLS embedding (first 5 dims):", cls_emb[0, :5])


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="ea78891063587eb050ed4166b20062eaf978037c",
)
model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="ea78891063587eb050ed4166b20062eaf978037c",
)

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)
print(sentence_embeddings @ sentence_embeddings.T)

# 1. Load a pretrained CrossEncoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# The texts for which to predict similarity scores
query = "How many people live in Berlin?"
passages = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
    "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
]

# 2a. Either predict scores pairs of texts
scores = model.predict([(query, passage) for passage in passages])
print(scores)
# => [8.607139 5.506266 6.352977]

# 2b. Or rank a list of passages for a query
ranks = model.rank(query, passages, return_documents=True)

print("Query:", query)
for rank in ranks:
    print(f"- #{rank['corpus_id']} ({rank['score']:.2f}): {rank['text']}")
"""
Query: How many people live in Berlin?
- #0 (8.61): Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
- #2 (6.35): In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
- #1 (5.51): Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.
"""
