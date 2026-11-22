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

# Function for re (catch all factual details that vector embeddings tend to miss).
def extract_hard_facts(text: str) -> set[str]:

    if not text or not text.strip():
        return set()

    #1. Extract and normalize numeric values
    num_pattern = r"\$?\b\d+(?:,\d{3})*(?:\.\d+)?%?(?!\w)"
    numbers = {match.replace(",", "") for match in re.findall(num_pattern, text)}

    #2. Extract mid-sentence proper nouns (Using sent_tokenize directly so 2-word sentences aren't skipped)
    proper_nouns: set[str] = set()
    noun_pattern = re.compile(r"^[A-Z][a-zA-Z0-9-]{1,}$")

    for sentence in nltk.sent_tokenize(text.strip()):
        words = sentence.split()
        for raw_word in words[1:]:
            cleaned = raw_word.strip(".,!?;:\"'()")
            if noun_pattern.match(cleaned):
                proper_nouns.add(cleaned)


    return numbers | proper_nouns


def score_sentence(sim_row: np.ndarray, summary_sentence: str, source_sentences: list[str], source_facts: set[str], penalty_factor: float = 0.5,) -> dict:

    if sim_row.size == 0 or not source_sentences:
        raw_sim = 0.0
        best_source = ""
    else:
        best_idx = int(np.argmax(sim_row))
        raw_sim = float(sim_row[best_idx])
        best_source = source_sentences[best_idx]

    

    # Checking if the summary sentence introduced numbers or proper nouns that never appeared in the original text.
    summary_facts = extract_hard_facts(summary_sentence)
    unverified_facts = sorted(summary_facts - source_facts)

    # Calculate the penalized final score
    base_score = max(0.0, raw_sim)

    final_score = base_score * penalty_factor if unverified_facts else base_score

    if final_score >= 0.75:
        status = "High"
    elif final_score >= 0.50:
        status = "Moderate"
    else:
        status = "Low"

    return {"sentence": summary_sentence, "raw_similarity": round(raw_sim, 4), "final_score": round(final_score, 4), "status": status, "best_match_source": best_source, "unverified_facts": unverified_facts,}


def analyze_summary(source_text: str, summary_text: str, model: SentenceTransformer, penalty_factor: float = 0.5,) -> dict:
    
    # Validating inputs and split sentences
    if not source_text or not source_text.strip() or not summary_text or not summary_text.strip():
        return {"error": "Both source and summary text are required."}

    source_sentences = split_sentences(source_text)
    summary_sentences = split_sentences(summary_text)

    if not source_sentences or not summary_sentences:
        return {"error": "Please enter at least one full sentence (3+ words) for both texts."}

    source_embeddings = embed_sentences(source_sentences, model)
    summary_embeddings = embed_sentences(summary_sentences, model)
    sim_matrix = compute_similarity_matrix(source_embeddings, summary_embeddings)
    source_facts = extract_hard_facts(source_text)

    sentence_breakdown = []
    weighted_score_sum = 0.0
    total_words = 0

    for i, sentence in enumerate(summary_sentences):
        result = score_sentence(sim_matrix[i], sentence, source_sentences, source_facts, penalty_factor)
        sentence_breakdown.append(result)
        word_count = len(sentence.split())
        weighted_score_sum += word_count * result["final_score"]
        total_words += word_count

    overall_score = round((weighted_score_sum / total_words) * 100, 1)if total_words > 0 else 0.0
    total_unverified = sum(len(item["unverified_facts"]) for item in sentence_breakdown)

    return {"overall score": overall_score,"total_unverified_facts": total_unverified, "sentence_breakdown": sentence_breakdown}

if __name__ == "__main__":
    import pprint

    test_model = SentenceTransformer("all-MiniLM-L6-v2")
    src = "Mount Lyell is a historic copper mine in Queenstown, Tasmania. Operations began in 1893."
    summ = "Mount Lyell is a copper mine in Queenstown, Tasmania. It opened in 1995 and mined diamonds."

    pprint.pprint(analyze_summary(src, summ, test_model))