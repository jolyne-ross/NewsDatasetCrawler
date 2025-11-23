import re
import unicodedata
import spacy
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from trafilatura import bare_extraction

NLP = None
_SENTIMENT = None

## callable given to ProcessPoolExecutor to call on creation of workers
def init_worker():
    global NLP, _SENTIMENT
    if NLP == None:
        NLP = spacy.load("en_core_web_sm", disable=["textcat"])
        NLP.add_pipe("sentencizer")
    
    if _SENTIMENT == None:
        _SENTIMENT = SentimentIntensityAnalyzer()

def preload_models():
    global NLP, _SENTIMENT
    NLP = spacy.load("en_core_web_sm", disable=["textcat"])
    NLP.add_pipe("sentencizer")
    _SENTIMENT = SentimentIntensityAnalyzer()

def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if ch.isprintable() or ch in"\n\t")
    return s

## cpu bound synchronous processing.
def process_html(html: str) -> dict:
    ## raw retrieval && error handling
    try:
        data = bare_extraction(filecontent=html, with_metadata=True)
    except Exception as e:
        return {
            "ok": False,
            "error_type": "extraction",
            "error": str(e)
        }

    if data == None: return {
        "ok": False,
        "error_type": "extraction",
        "error": "data_none"
    }

    if data.text == None or data.text == "": return {
        "ok": False,
        "error_type": "extraction",
        "error": "text_none"
    }

    ## get text
    text = data.text

    ## normalize unicode and replace any whitespace w/ a single space
    clean_text = normalize_unicode(text)

    sentence_candidates = re.findall(r"(?<=[.!?\n]).*?(?=[.!?])", clean_text)
    sentences = [s.strip() for s in sentence_candidates if len(s.strip()) > 0]

    token_candidates = re.findall(r"[A-Za-z0-9\-+]+")
    tokens_basic = [t.lower() for t in token_candidates if len(t)>1]

    length_chars = len(clean_text)
    length_words = len(tokens_basic)
    length_sentences = len(sentences)

    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    quality_flags = {
        "empty": length_words==0,
        "short": length_words<50,
        "very_short": length_words<20,
        "low_quality": length_sentences<2 or length_words<30
    }

    return {
        "ok": True,
        "trafilatura_data": {
            "site_name": data.sitename or "",
            "author": data.author or "",
            "categories": data.categories or "",
            "tags": data.tags or "",
            "fingerprint": data.fingerprint or "",
            "license": data.license or "",
            "comments": data.comments or "",
            "description": data.description or ""
        },
        "processed_data": {
            "text": clean_text,
            "sentences": sentences,
            "tokens": tokens_basic,
            "length_chars": length_chars,
            "length_sentences": length_sentences,
            "length_words": length_words,
            "quality_flags": quality_flags
        }
    }
