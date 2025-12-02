import re
import unicodedata
import spacy
import math
import statistics
from multiprocessing import current_process
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
        NLP.add_pipe("sentencizer", before="parser")

    
    if _SENTIMENT == None:
        _SENTIMENT = SentimentIntensityAnalyzer()

## ran in Linux systems for forking functionaliy
def preload_models():
    global NLP, _SENTIMENT
    NLP = spacy.load("en_core_web_sm", disable=["textcat"])
    NLP.add_pipe("sentencizer", before="parser")
    print(NLP.pipe_names)
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
        if token.pos_ in CANDIDATE_POS
        and token.is_alpha
        and not token.is_stop
        and len(token.lemma_) > 2
    ]
    freq = Counter(words)
    return [{
        "keyword": w, 
        "count": c, 
        "sentiment_sum": 0, 
        "sentiment_n": 0,
        "sentiment_list": []
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
        "sentiment_n": 0.0,
        "sentiment_list": []
    } for (text, label), count in freq.most_common(max_entities)]

MANUAL_KEYWORDS = [
    "transgender", "trans", "nonbinary", "non-binary", "enby", "transition",
    "transitioning", "detransition", "gender", "identity", "pronoun",
    "misgender", "deadname", "transphobia", "transphobic", "intersex",
    "genderfluid", "genderqueer", "agender", "bigender", "mtf", "ftm",
    "bind", "tuck", "hormone", "puberty", "dysphoria", "cisgender", "cis",
    "transfem", "transmasc", "transsexual", "transvestite", "tranny",
    "transgenderism"
]

FLAGGED_PHRASES = [
    "biological male", "biological female", "gender ideology", "woke agenda",
    "woke ideology", "men in women's sports", "pronoun police", 
    "sex change surgery", "born a man", "born a woman", "real woman",
    "real man", "trans agenda", "radical gender", "grooming children",
    "gender ideology"
]

TOPIC_CLUSTERS = {
    "healthcare": ["hormone", "puberty", "transition", "surgery", "doctor", "dysphoria", "care", "clinic"],
    "politics": ["bill", "legislation", "senate", "policy", "law", "ban"],
    "violence": ["attack", "murder", "violence", "assault", "crime", "hate"],
    "media_culture": ["tv", "actor", "award", "ad", "backlash", "boycott"],
    "education": ["school", "curriculum", "teacher", "book", "library"],
    "sports": ["athlete", "olympics", "compete", "team", "ban", "fairness"]
}

def sentiment_measures(
        doc,
        entities: list[dict],
        keywords: list[dict],
        title: str
    ):
    def _safe_stdev(values: list):
        return statistics.stdev(values) if len(values) >= 2 else 0.0
    
    doc_sentiment = _SENTIMENT.polarity_scores(doc.text)
    title_sentiment = _SENTIMENT.polarity_scores(title)

    component_sums = [0.0, 0.0, 0.0] ## [neg, neu, pos]

    negs = {"list": [], "sum": 0.0}
    neus = {"list": [], "sum": 0.0}
    poss = {"list": [], "sum": 0.0}
    compound = {"list": [], "sum": 0.0} ## (list, sum)

    weighted_pos = {"list": [], "sum": 0.0} ## (list, sum)
    weighted_neg = {"list": [], "sum": 0.0} ## (list, sum) 
    weighted_pos_len = {"sum": 0.0, "weight_sum": 0.0} ## (sum, weights)

    balance = {"list": [], "sum": 0.0} ## (list, sum)
    skew = {"list": [], "sum": 0.0} ## (list, sum)

    '''
    sent_default_signals = {"neg": 0, "neu": 0, "pos": 0} ## (negs, neus, poss); measure of how many sentences meet thresholds thresholds
    sent_strong_signals = {"neg": 0, "pos": 0} ## (negs, poss); measure of how many sentences meet strong compond thresholds

    component_signals = {"neg": 0, "neu": 0, "pos": 0} ## (negs, neus, poss); measure of how many sentences meet component thresholds
    factual_signals = {"neg": 0, "pos": 0} ## (negs, poss); measure of how many sentences meet factual signals
    '''

    sent_list = [s for s in doc.sents if s.text.strip()]
    n = len(sent_list)
    sentence_sentiments = []

    manual_keywords = [{ 
        "keyword": kw,
        "count": 0,
        "sentiment_n": 0,
        "sentiment_sum": 0.0,
        "sentiment_list": []
    } for kw in MANUAL_KEYWORDS]

    flagged_phrases = [{
        "phrase": phrase,
        "count": 0,
        "sentiment_n": 0,
        "sentiment_sum": 0.0,
        "sentiment_list": []
    } for phrase in FLAGGED_PHRASES]

    topic_clusters = [{
        "topic": topic,
        "keywords": kw,
        "count": 0,
        "sentiment_n": 0,
        "sentiment_sum": 0.0,
        "sentiment_list": [],
        "temp_count": 0
    } for topic, kw in TOPIC_CLUSTERS.items()]

    if n <= 0: return {"ok": False, "error_type": "nlp", "error": "empty_doc"}

    for i, s in enumerate(sent_list):
        if not s.text.strip(): continue
        sents = _SENTIMENT.polarity_scores(s.text.strip())
        l = len(s)

        neg, neu, pos, comp = sents.values()
        for j, key in enumerate(["neg", "neu", "pos"]): component_sums[j] += sents[key]


        negs["list"].append(neg)
        negs["sum"] += neg

        neus["list"].append(neu)
        neus["sum"] += neu

        poss["list"].append(pos)
        poss["sum"] += pos

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

        ## skew score (pos component - neg component)
        skew_score = pos-neg
        skew["list"].append(skew_score)
        skew["sum"] += skew_score

        ## signal detection (just sorting into various bins)
        '''if comp <= -threshold: 
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
        if pos >= fact_threshold and comp < threshold: factual_signals["pos"] += 1'''

        ## keyword and entity summation
        lemmas = [t.lemma_.lower() for t in s if t.is_alpha]

        if entities:
            for ent in entities:
                if ent["entity"].lower() in s.text.lower():
                    ent["sentiment_sum"] += comp
                    ent["sentiment_n"] += 1
                    ent["sentiment_list"].append(comp)
        
        if keywords:
            for kw in keywords:
                if kw["keyword"] in lemmas:
                    kw["sentiment_sum"] += comp
                    kw["sentiment_n"] += 1
                    kw["sentiment_list"].append(comp)
        
        if MANUAL_KEYWORDS:
            for kw in manual_keywords:
                count = sum([1 if kw["keyword"] in l else 0 for l in lemmas])
                kw["count"] += count
                if count>0:
                    kw["sentiment_sum"] += comp
                    kw["sentiment_n"] += 1
                    kw["sentiment_list"].append(comp)
        
        if FLAGGED_PHRASES:
            for phrase in flagged_phrases:
                if phrase["phrase"] in s.text.lower():
                    phrase["count"] += 1
                    phrase["sentiment_sum"] += comp
                    phrase["sentiment_n"] += 1
                    phrase["sentiment_list"].append(comp)
        
        sentence_topic = ""
        if TOPIC_CLUSTERS:
            for topic in topic_clusters:
                topic["temp_count"] = sum([1 if l in topic["keywords"] else 0 for l in lemmas])
                if topic["topic"] == "politics": sum([1 if ent.label_ == "LAW" else 0 for ent in s.ents])
            
            largest = topic_clusters[0]
            for topic in topic_clusters:
                if topic["temp_count"] > largest["temp_count"]: largest = topic
            largest["count"] += 1
            largest["sentiment_sum"] += comp
            largest["sentiment_list"].append(comp)
            sentence_topic = largest["topic"] if largest["temp_count"] > 2 else ""

        sentence_sentiments.append({"sentence": s.text.strip(), "topic": sentence_topic, **sents})

    for ent in entities:
        if ent["sentiment_n"] > 0: 
            ent["sentiment_avg"] = ent["sentiment_sum"]/ent["sentiment_n"]
            ent["sentiment_sd"] = _safe_stdev(ent["sentiment_list"])
            ent["sentiment_min"] = min(ent["sentiment_list"])
            ent["sentiment_max"] = max(ent["sentiment_list"])
            ent["sentiment_median"] = statistics.median(ent["sentiment_list"])
            del ent["sentiment_sum"], ent["sentiment_n"]
        else: 
            ent["sentiment_avg"] = 0
            del ent["sentiment_sum"], ent["sentiment_n"]

    for kw in keywords:
        if kw["sentiment_n"] > 0: 
            kw["sentiment_avg"] = kw["sentiment_sum"]/kw["sentiment_n"]
            kw["sentiment_sd"] = _safe_stdev(kw["sentiment_list"])
            kw["sentiment_min"] = min(kw["sentiment_list"])
            kw["sentiment_max"] = max(kw["sentiment_list"])
            kw["sentiment_median"] = statistics.median(kw["sentiment_list"])
            del kw["sentiment_sum"], kw["sentiment_n"]
        else: 
            kw["sentiment_avg"] = 0
            del kw["sentiment_sum"], kw["sentiment_n"]

    manual_keywords_filtered = []
    for kw in manual_keywords:
        if kw["sentiment_n"] > 0:
            kw["sentiment_avg"] = kw["sentiment_sum"] / kw["sentiment_n"]
            kw["sentiment_sd"] = _safe_stdev(kw["sentiment_list"])
            kw["sentiment_min"] = min(kw["sentiment_list"])
            kw["sentiment_max"] = max(kw["sentiment_list"])
            kw["sentiment_median"] = statistics.median(kw["sentiment_list"])
            del kw["sentiment_sum"], kw["sentiment_n"]
            manual_keywords_filtered.append(kw)
    manual_keywords = manual_keywords_filtered

    if len(manual_keywords)==0: manual_keywords = [{
        "keyword": "",
        "count": -1,
        "sentiment_avg": 0.0,
        "sentiment_sd": 0.0,
        "sentiment_min": 0.0,
        "sentiment_max": 0.0,
        "sentiment_median": 0.0,
        "sentiment_list": [0.0]
    }]

    new_phrases = []
    for phrase in flagged_phrases:
        if phrase["sentiment_n"] > 0:
            phrase["sentiment_avg"] = phrase["sentiment_sum"] / phrase["sentiment_n"]
            phrase["sentiment_sd"] = _safe_stdev(phrase["sentiment_list"])
            phrase["sentiment_min"] = min(phrase["sentiment_list"])
            phrase["sentiment_max"] = max(phrase["sentiment_list"])
            phrase["sentiment_median"] = statistics.median(phrase["sentiment_list"])
            del phrase["sentiment_sum"], phrase["sentiment_n"]
            new_phrases.append(phrase)
    flagged_phrases = new_phrases

    if len(flagged_phrases)==0: flagged_phrases = [{
        "phrase": "",
        "count": -1,
        "sentiment_avg": 0.0,
        "sentiment_sd": 0.0,
        "sentiment_min": 0.0,
        "sentiment_max": 0.0,
        "sentiment_median": 0.0,
        "sentiment_list": [0.0]
    }]
        
    topic_clusters_filtered = []
    largest = 0
    largest_topic = ""
    for topic in topic_clusters:
        if topic["count"] > 0: 
            if topic["count"] > largest: 
                largest = topic["count"]
                largest_topic = topic["topic"]
            if topic["count"] == largest: largest_topic += ", " + topic["topic"]
            topic["sentiment_avg"] = topic["sentiment_sum"] / topic["count"]
            topic["sentiment_sd"] = _safe_stdev(topic["sentiment_list"])
            topic["sentiment_min"] = min(topic["sentiment_list"])
            topic["sentiment_max"] = max(topic["sentiment_list"])
            topic["sentiment_median"] = statistics.median(topic["sentiment_list"])
            del topic["sentiment_sum"]
            topic_clusters_filtered.append(topic)

    topic_clusters = topic_clusters_filtered

    if len(topic_clusters)==0: topic_clusters = [{
        "topic": "",
        "keywords": [""],
        "count": -1,
        "sentiment_avg": 0.0,
        "sentiment_sd": 0.0,
        "sentiment_min": 0.0,
        "sentiment_max": 0.0,
        "sentiment_median": 0.0,
        "sentiment_list": [0.0],
    }]

    return {
        "ok": True,
        "doc_sentiment": doc_sentiment,
        "title_sentiment": title_sentiment,
        "sentence_sentiments": sentence_sentiments,

        "neg": {
            "avg": negs["sum"]/n,
            "sd": _safe_stdev(negs["list"]),
            "min": min(negs["list"]),
            "max": max(negs["list"]),
            "median": statistics.median(negs["list"]),
        },

        "neu": {
            "avg": neus["sum"]/n,
            "sd": _safe_stdev(neus["list"]),
            "min": min(neus["list"]),
            "max": max(neus["list"]),
            "median": statistics.median(neus["list"]),
        },

        "pos": {
            "avg": poss["sum"]/n,
            "sd": _safe_stdev(poss["list"]),
            "min": min(poss["list"]),
            "max": max(poss["list"]),
            "median": statistics.median(poss["list"]),
        },

        "compound": {
            "avg": compound["sum"]/n,
            "sd": _safe_stdev(compound["list"]),
            "min": min(compound["list"]),
            "max": max(compound["list"]),
            "median": statistics.median(compound["list"]),
        },

        "weighted_neg": {
            "avg": weighted_neg["sum"]/n,
            "sd": _safe_stdev(weighted_neg["list"]),
            "min": min(weighted_neg["list"]),
            "max": max(weighted_neg["list"]),
            "median": statistics.median(weighted_neg["list"]),
        },

        "weighted_pos": {
            "avg": weighted_pos["sum"]/n,
            "sd": _safe_stdev(weighted_pos["list"]),
            "min": min(weighted_pos["list"]),
            "max": max(weighted_pos["list"]),
            "median": statistics.median(weighted_pos["list"]),
        },

        "weighted_position_length": {
            "avg": weighted_pos_len["sum"]/weighted_pos_len["weight_sum"] if weighted_pos_len["weight_sum"] > 0 else 0.0,
            "sd": -1,
            "min": -1,
            "max": -1,
            "median": -1,
        },

        "balance": {
            "avg": balance["sum"]/n,
            "sd": _safe_stdev(balance["list"]),
            "min": min(balance["list"]),
            "max": max(balance["list"]),
            "median": statistics.median(balance["list"]),
        },

        "skew": {
            "avg": skew["sum"]/n,
            "sd": _safe_stdev(skew["list"]),
            "min": min(skew["list"]),
            "max": max(skew["list"]),
            "median": statistics.median(skew["list"]),
        },

        "keywords": keywords,
        "entities": entities,
        "manual_keywords": manual_keywords,
        "flagged_phrases": flagged_phrases,
        "topic_clusters": topic_clusters
    }

## cpu NLP Pipeline
def nlp_pipeline(clean_text: str, title: str) -> dict:
    print(f"[WORKER {current_process().name:<14} PROCESS: {current_process().pid}] NLP: {title}")
    doc = NLP(clean_text)

    tokens = [{
        "text": t.text,
        "lemma": t.lemma_,
        "pos": t.pos_,
        "dep": t.dep_,
        "is_stop": t.is_stop
    } for t in doc if t.is_alpha]

    entity_labels = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "EVENT", "LAW"]
    entities = extract_entities(doc.ents, entity_labels, 15)
    keywords = extract_keywords(doc, 10)

    sentiment_report = sentiment_measures(
        doc=doc,
        entities=entities,
        keywords=keywords,
        title=title
    )

    return {
        "tokens": tokens,
        **sentiment_report,
    }

## clean text w/ trafilature (entry point); mainly seperating this so i can create a test driver for the NLP Segment
def process_html(html: str, title: str):
    print(f"[WORKER {current_process().name:<14} PROCESS: {current_process().pid}] EXTRACTION: {title}")

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

    if not isinstance(data.text, str): return {
        "ok": False,
        "error_type": "extraction",
        "error": "text_not_str"
    }

    ## normalize unicode and replace any whitespace w/ a single space
    try:
        clean_text = normalize_unicode(data.text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        clean_title = normalize_unicode(title)
        clean_title = re.sub(r"\s+", " ", clean_title).strip()
    except Exception as e: return {
        "ok": False,
        "error_type": "normalizing",
        "error": str(e),
        "text": clean_text if isinstance(clean_text, str) else None
    }

    if not isinstance(data.text, str): return {
        "ok": False,
        "error_type": "normalizing",
        "error": "text_not_str"
    }

    ## call nlp pipeline
    try:
        results = nlp_pipeline(clean_text=clean_text, title=clean_title)
    except Exception as e:
        return {
            "ok": False,
            "error_type": "nlp_pipe",
            "error": str(e),
            "text": clean_text
        }

    author = ""
    if data.author is None:
        author =  ""
    if isinstance(data.author, str):
        author =  data.author.strip()

    # a list of authors → join strings
    if isinstance(data.author, list):
        try:
            good = [x.strip() for x in data.author if isinstance(x, str)]
            author = ", ".join(good)
        except Exception:
            author =  ""


    ## return dict
    return {
        "author": author,
        "text": clean_text,
        **results
    }
