import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# loading the pretrained model from hugging face
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str) -> list[str]:
# Split raw text into clean valid sentences using NLTK's Punkt tokenizer

    if not text or not text.strip():
        return []

    raw_sentences = nltk.sent_tokenize(text.strip())

    # filter out tiny noise fragments (fewer than 3 words)

    clean_sentences =[s.strip() for s in raw_sentences if len(s.split()) >= 3]

    return clean_sentences

def embed_sentences(sentences: list[str], model) -> np.ndarray:
    if not sentences:
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype=np.float32)

    return model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

def compute_similarity_matrix(source_embeddings: np.ndarray, summary_embeddings: np.ndarray) -> np.ndarray:

    if source_embeddings.size == 0 or summary_embeddings.size == 0:
        return np.empty((summary_embeddings.shape[0], source_embeddings.shape[0]), dtype=np.float32)


    similarity_matrix = summary_embeddings @ source_embeddings.T

    return np.clip(similarity_matrix, -1.0, 1.0)

def extract_hard_facts(text: str) -> set[str]:
    