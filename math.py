import numpy as np
import sentence_transformers as SentenceTransformer

model = SentenceTransformer("all-miniLM-L6-v2")

query = "How do i reset my password?"

candidates = ["Click here to update your billing info",
              "I forgot my login credentials and need a reset.",
              "The weather is nice today.",]

q_vec = model.encode(query, normalize_embeddings=True)
cand_vec