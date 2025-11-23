import re
import unicodedata
from json import loads
from trafilatura import extract
from trafilatura import bare_extraction

def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if ch.isprintable() or ch in"\n\t")
    return s

## cpu bound synchronous processing.
def process_html(html: str, article: dict) -> dict:
    ## raw retrieval && error handling
    try:
        data = bare_extraction(filecontent=html, with_metadata=True)
    except Exception as e:
        return {
            "ok": False,
            "id": article['id'],
            "media_url": article['media_url'],
            "url": article['url'],
            'title': article['title'],
            "publish_date": article['publish_date'],
            "error_type": "extract_exception",
            "error": str(e)
        }

    if data == None: return {
        "ok": False,
        "id": article['id'],
        "media_url": article['media_url'],
        "url": article['url'],
        'title': article['title'],
        "publish_date": article['publish_date'],
        "error_type": "extract_none"
    }

    if data.text == None or data.text == "": return {
        "ok": False,
        "id": article['id'],
        "media_url": article['media_url'],
        "url": article['url'],
        'title': article['title'],
        "publish_date": article['publish_date'],
        "error_type": "text_none"
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
        "id": article['id'],
        "media_url": article['media_url'],
        "url": article['url'],
        'title': article['title'],
        "publish_date": article['publish_date'],
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
