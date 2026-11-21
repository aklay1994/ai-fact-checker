import nltk
from nltk.tokenize import sent_tokenize

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