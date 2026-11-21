import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "How do i reset my password?"

candidates = ["Click here to update your billing info",
              "I forgot my login credentials and need a reset.",
              "The weather is nice today.",]


# 1. Encode using task-specific methods
q_emb = model.encode_query(query)
cand_emb = model.encode_document(candidates)

# 2. Compute similarity directly with the model's built-in method
# Returns a 2D PyTorch tensor of shape (1, 3)
scores = model.similarity(q_emb, cand_emb)[0]
 
# 3. Extract the max score and best match
best_idx = scores.argmax().item()
max_score = scores[best_idx].item()

print(f"Max Score: {max_score:.4f}")
print(f"Best Match: '{candidates[best_idx]}'")