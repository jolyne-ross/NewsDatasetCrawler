import re
import unicodedata
import spacy
import math
import statistics
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

## ran in Linux systems for forking functionaliy
def preload_models():
    global NLP, _SENTIMENT
    NLP = spacy.load("en_core_web_sm", disable=["textcat"])
    NLP.add_pipe("sentencizer")
    _SENTIMENT = SentimentIntensityAnalyzer()


## NLP functions

def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if ch.isprintable() or ch in"\n\t")
    return s

def extract_keywords(tokens, max_keywords=15):
    CANDIDATE_POS = {"NOUN", "PROPN", "ADJ"}
    words = [
        token.lemma_.lower()
        for token in tokens
        if token.pos in CANDIDATE_POS
        and token.is_alpha
        and not token.is_stop
        and len(token.lemma) > 2
    ]
    freq = Counter(words)
    return [{
        "keyword": w, 
        "count": c, 
        "sentiment_sum": 0, 
        "sentiment_n": 0
    } for w, c in freq.most_common(max_keywords)]

def extract_entities(ents, labels: list[str], max_entities: int | None = None):
    def label_ok(label: str):
        return not labels or label in labels
    
    filtered = [ent for ent in ents if label_ok(ent.label_) and ent.text.strip()]
    freq = Counter((ent.text.strip(), ent.label_) for ent in filtered)
    return [{
        "entity": text,
        "label": label,
        "count": count,
        "sentiment_sum": 0.0,
        "sentiment_n": 0.0
    } for (text, label), count in freq.most_common(max_entities)]

def sentiment_measures(
        doc,
        entities: list[dict],
        keywords: list[dict],
        threshold=0.5, strong_threshold=0.75, neg_pos_threshold=0.1, neu_threshold=0.5,  fact_threshold=0.15,
    ):
    def _safe_stdev(values: list):
        return statistics.stdev(values) if len(values) >= 2 else 0.0
    
    doc_sentiment = _SENTIMENT.polarity_scores(doc.text)

    component_sums = [0.0, 0.0, 0.0] ## [neg, neu, pos]
    compound = {"list": [], "sum": 0.0} ## (list, sum)

    weighted_pos = {"list": [], "sum": 0.0} ## (list, sum)
    weighted_neg = {"list": [], "sum": 0.0} ## (list, sum) 
    weighted_pos_len = {"sum": 0.0, "weight_sum": 0.0} ## (sum, weights)

    balance = {"list": [], "sum": 0.0} ## (list, sum)

    sent_default_signals = {"neg": 0, "neu": 0, "pos": 0} ## (negs, neus, poss); measure of how many sentences meet thresholds thresholds
    sent_strong_signals = {"neg": 0, "pos": 0} ## (negs, poss); measure of how many sentences meet strong compond thresholds

    component_signals = {"neg": 0, "neu": 0, "pos": 0} ## (negs, neus, poss); measure of how many sentences meet component thresholds
    factual_signals = {"neg": 0, "pos": 0} ## (negs, poss); measure of how many sentences meet factual signals

    sent_list = [s for s in doc.sents if s.text.strip()]
    n = len(sent_list)
    sentence_sentiments = []

    if n <= 0: return {"ok": False, "error_type": "nlp", "error": "empty_doc"}

    for i, s in enumerate(sent_list):
        if not s.text.strip(): continue
        sents = _SENTIMENT.polarity_scores(s.text.strip())
        l = len(s)
        sentence_sentiments.append({"sentence": s.text.strip(), **sents})

        neg, neu, pos, comp = sents.values()
        for j, key in enumerate(["neg", "neu", "pos"]): component_sums[j] += sents[key]

        compound["list"].append(comp)
        compound["sum"] += comp

        weighted_neg["list"].append(comp*neg)
        weighted_neg["sum"] += comp*neg

        weighted_pos["list"].append(comp*pos)
        weighted_pos["sum"] += comp*pos

        w_pos = (i+1)/n
        w_len = max(l, 1)
        w = w_pos * w_len
        weighted_pos_len["sum"] += comp * w
        weighted_pos_len["weight_sum"] += w

        ## balance score (entropy of neg/neu/pos)
        eps = 1e-12
        probs = [max(neg, eps), max(neu, eps), max(pos, eps)]
        norm = sum(probs)
        probs = [p/norm for p in probs]

        entropy = -sum(p * math.log(p) for p in probs)
        balance_score = entropy / math.log(3) ## normalized to [0,1]

        balance["list"].append(balance_score)
        balance["sum"] += balance_score

        ## signal detection (just sorting into various bins)
        if comp <= -threshold: 
            sent_default_signals["neg"] += 1
            if comp <= -strong_threshold: sent_strong_signals["neg"] += 1

        elif comp >= threshold: 
            sent_default_signals["pos"] += 1
            if comp >= strong_threshold: sent_strong_signals["pos"] +=1

        else: sent_default_signals["neu"] += 1

        if neg > neg_pos_threshold: component_signals["neg"] += 1
        if pos > neg_pos_threshold: component_signals["pos"] += 1
        if neu > neu_threshold: component_signals["neu"] += 1

        if neg >= fact_threshold and comp > -threshold: factual_signals["neg"] += 1
        if pos >= fact_threshold and comp < threshold: factual_signals["pos"] += 1

        ## keyword and entity summation
        if entities:
            for ent in entities:
                if ent["entity"].lower() in s.text.lower():
                    ent["sentiment_sum"] += comp
                    ent["sentiment_n"] += 1
        
        if keywords:
            lemmas = [t.lemma_.lower() for t in s if t.is_alpha]
            for kw in keywords:
                if kw["keyword"] in lemmas:
                    kw["sentiment_sum"] += comp
                    kw["sentiment_n"] += 1


    for ent in entities:
        if ent["sentiment_n"] > 0: ent["sentiment_avg"] = ent["sentiment_sum"]/ent["sentiment_n"]
        else: ent["sentiment_avg"] = 0
    for kw in keywords:
        if kw["sentiment_n"] > 0: kw["sentiment_avg"] = kw["sentiment_sum"]/kw["sentiment_n"]
        else: kw["sentiment_avg"] = 0

    return {
        "doc_sentiment": doc_sentiment,
        "sentence_sentiments": sentence_sentiments,

        "component_avgs": {
            "neg": component_sums[0]/n,
            "neu": component_sums[1]/n,
            "pos": component_sums[2]/n,
        },

        "compound": {
            "avg": compound["sum"]/n,
            "sd": _safe_stdev(compound["list"]),
        },

        "weighted_neg": {
            "avg": weighted_neg["sum"]/n,
            "sd": _safe_stdev(weighted_neg["list"]),
        },

        "weighted_pos": {
            "avg": weighted_pos["sum"]/n,
            "sd": _safe_stdev(weighted_pos["list"]),
        },

        "weighted_position_length": {
            "avg": weighted_pos_len["sum"]/weighted_pos_len["weight_sum"] if weighted_pos_len["weight_sum"] > 0 else 0.0,
            "sd": -1,
        },

        "balance": {
            "avg": balance["sum"]/n,
            "sd": _safe_stdev(balance["list"])
        },

        "neg_signals": {
            "ratio": sent_default_signals["neg"]/n,
            "strong_ratio": sent_strong_signals["neg"]/n,
            "factual_ratio": factual_signals["neg"]/n,
            "density": component_signals["neg"]/n
        },

        "pos_signals": {
            "ratio": sent_default_signals["pos"]/n,
            "strong_ratio": sent_strong_signals["pos"]/n,
            "factual_ratio": factual_signals["pos"]/n,
            "density": component_signals["pos"]/n
        },

        "neu_signals": {
            "ratio": sent_default_signals["neu"]/n,
            "strong_ratio": -1,
            "factual_ratio": -1,
            "density": component_signals["neu"]/n
        },

        "keywords": keywords,
        "entities": entities,
    }

## cpu NLP Pipeline
def nlp_pipeline(clean_text: str) -> dict:
    doc = NLP(clean_text)

    tokens = [{
        "text": t.text,
        "lemma": t.lemma_,
        "pos": t.pos_,
        "dep": t.dep_,
        "is_stop": t.is_stop
    } for t in doc if t.is_alpha]

    entity_labels = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "EVENT", "LAW"]
    entities = extract_entities(doc, entity_labels, 15)
    keywords = extract_keywords(doc, 10)

    sentiment_report = sentiment_measures(
        doc=doc,
        entities=entities,
        keywords=keywords,
    )

    return {
        "tokens": tokens,
        **sentiment_report,
    }


## clean text w/ trafilature (entry point); mainly seperating this so i can create a test driver for the NLP Segment
def process_html(html: str):
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

    ## normalize unicode and replace any whitespace w/ a single space
    clean_text = normalize_unicode(data.text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    ## call nlp pipeline
    results = nlp_pipeline(clean_text=clean_text)

    ## return dict
    return {
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
        "text": clean_text,
        **results
    }
