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
KEYWORDS = []

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
        
