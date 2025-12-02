import pandas as pd
import re

def sub_sample(args, data: pd.DataFrame) -> pd.DataFrame:
    return data.sample(
        n = args.n_rows if args.n_rows != None else None,
        frac = args.frac if args.frac != None else None,
        replace = args.replace,
        random_state = args.seed if args.seed != None else None,
    )

# A generous keyword list for trans-issue detection
KEYWORDS = [
    r"\btransgender\b", r"\btrans\b", r"\btrans woman\b", r"\btrans man\b",
    r"\btranswomen\b", r"\btransmen\b", r"\btrans people\b",
    r"\bnon[- ]?binary\b", r"\bnonbinary\b", r"\bgender[- ]?fluid\b",
    r"\bgender[- ]?queer\b", r"\bagender\b", r"\bbigender\b",
    r"\btransition\b", r"\btransitioning\b", r"\bdetransition\b", r"\bde-transition\b",
    r"\bdysphoria\b",
    r"\blgb[a-z0-9]*\b",
    r"\bintersex\b",
    r"\bpronoun\b", r"\bpronouns\b",
    r"\bmisgender\b", r"\bmisgendered\b", r"\bmisgendering\b",
    r"\bdeadname\b", r"\bdeadnamed\b",
    r"\btransphobia\b", r"\btransphobic\b",
    r"\bmtf\b", r"\bftm\b",
    r"\bhormone\b", r"\bhormone therapy\b", r"\bhormone replacement\b",
    r"\bpuberty blocker\b", r"\bpuberty blockers\b", r"\bpuberty\b"
    r"\bcisgender\b", r"\bcis\b",
    r"\bbiology\b", r"\bsports\b", r"\bbiological\b", r"\bbirth\b",
    r"\btransgenderism\b", r"\bideology\b",
    r"\brobinson\b", r"\bboyfriend\b", r"\bsuspect\b", r"\bassassin\b" ## news about the charlie kirk assassin is heavily represented; this limits the stories about it to the main thing concering trans issues
]

PATTERNS = re.compile("|".join(KEYWORDS), flags=re.IGNORECASE)


def filter_articles(df: pd.DataFrame) -> pd.DataFrame:
    text_series = (
        df["title"].fillna("").astype(str)
        + " " +
        df["media_name"].fillna("").astype(str)
    )

    mask = text_series.str.contains(PATTERNS, regex=True)

    return df[mask].copy()


def main(args):
    data = pd.read_csv(args.input)
    if args.pd_query != None: data = data.query(args.pd_query)

    data = filter_articles(data)
    if args.frac == None or args.frac < 1: data = sub_sample(args, data)
    data.to_csv(args.output)
        
