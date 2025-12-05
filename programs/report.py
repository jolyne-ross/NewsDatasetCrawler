import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# From a column of dict-like objects (or None), extract `field`
# into a new numeric column.
def _extract_struct_field(
    df: pd.DataFrame,
    source_col: str,
    new_col: str,
    field: str,
) -> None:
    def getter(x):
        if isinstance(x, dict):
            return x.get(field, np.nan)
        return np.nan

    df[new_col] = df[source_col].map(getter)


def _parse_publish_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["publish_dt"] = pd.to_datetime(df["publish_date"], errors="coerce")
    return df


def _save_hist(
    series: pd.Series,
    out_path: str,
    title: str,
    xlabel: str,
    bins: int = 40,
) -> None:
    # always using *_avg series (or mean-of-avg) for histograms
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        print(f"[WARN] No data for histogram: {title}")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved histogram: {out_path}")


def _save_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    out_path: str,
    title: str,
    ylabel: str,
) -> None:
    # df should already hold per-date mean of an *_avg metric
    df = df[[date_col, value_col]].dropna()
    if df.empty:
        print(f"[WARN] No data for time series: {title}")
        return

    df = df.sort_values(date_col)

    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[value_col])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved time series: {out_path}")


def _save_time_series_by_domain(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    domain_col: str,
    out_path: str,
    title: str,
    ylabel: str,
    max_domains: int = 6,
) -> None:
    # df[value_col] should be an *_avg metric (e.g., compound_avg)
    df = df[[date_col, value_col, domain_col]].dropna()
    if df.empty:
        print(f"[WARN] No data for domain time series: {title}")
        return

    # top domains by article count
    top_domains = (
        df[domain_col]
        .value_counts()
        .head(max_domains)
        .index
        .tolist()
    )

    df = df[df[domain_col].isin(top_domains)]

    if df.empty:
        print(f"[WARN] No data left after filtering to top domains for: {title}")
        return

    grouped = (
        df
        .groupby([date_col, domain_col])[value_col]
        .mean()  # mean of article-level *_avg
        .reset_index()
    )

    # pivot for multi-line plot
    pivot = grouped.pivot(index=date_col, columns=domain_col, values=value_col)
    pivot = pivot.sort_index()

    plt.figure(figsize=(12, 6))
    for dom in pivot.columns:
        plt.plot(pivot.index, pivot[dom], label=str(dom))

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend(title="media_url", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved domain time series: {out_path}")


# Flatten a Series of list[dict] into a DataFrame via explode + json_normalize.
# Returns empty DataFrame if nothing usable.
def _flatten_list_column(
    series: pd.Series,
) -> pd.DataFrame:
    series = series.dropna()
    if series.empty:
        return pd.DataFrame()

    exploded = series.explode()
    exploded = exploded.dropna()
    if exploded.empty:
        return pd.DataFrame()

    return pd.json_normalize(exploded)


def _save_keyword_barplot(
    df_keywords: pd.DataFrame,
    key_col: str,
    count_col: str,
    sent_col: str,
    out_path: str,
    title: str,
    top_n: int = 20,
) -> None:
    if df_keywords.empty:
        print(f"[WARN] No data for keyword barplot: {title}")
        return

    # summarize by keyword, always on *_avg
    grouped = (
        df_keywords
        .groupby(key_col)
        .agg(
            total_count=(count_col, "sum"),
            mean_sentiment=(sent_col, "mean"),  # mean of sentiment_avg
        )
        .reset_index()
    )

    grouped = grouped.sort_values("total_count", ascending=False).head(top_n)

    if grouped.empty:
        print(f"[WARN] No data after grouping for: {title}")
        return

    x = grouped[key_col]
    counts = grouped["total_count"]
    sentiments = grouped["mean_sentiment"]

    plt.figure(figsize=(12, 6))
    plt.bar(x, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(key_col)
    plt.ylabel("Total count")

    # annotate bars with mean sentiment (rounded)
    for idx, (xi, yi, si) in enumerate(zip(x, counts, sentiments)):
        if pd.isna(si):
            label = "NA"
        else:
            label = f"{si:.2f}"
        plt.text(idx, yi, label, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved keyword barplot: {out_path}")


def _save_topic_barplot(
    df_topics: pd.DataFrame,
    out_path: str,
    title: str,
    top_n: int = 20,
) -> None:
    if df_topics.empty:
        print(f"[WARN] No data for topic barplot: {title}")
        return

    grouped = (
        df_topics
        .groupby("topic")
        .agg(
            total_count=("count", "sum"),
            mean_sentiment=("sentiment_avg", "mean"),  # mean of sentiment_avg
        )
        .reset_index()
    )

    grouped = grouped.sort_values("total_count", ascending=False).head(top_n)

    if grouped.empty:
        print(f"[WARN] No data after grouping topics for: {title}")
        return

    x = grouped["topic"]
    counts = grouped["total_count"]
    sentiments = grouped["mean_sentiment"]

    plt.figure(figsize=(12, 6))
    plt.bar(x, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("topic")
    plt.ylabel("Total count")

    for idx, (xi, yi, si) in enumerate(zip(x, counts, sentiments)):
        if pd.isna(si):
            label = "NA"
        else:
            label = f"{si:.2f}"
        plt.text(idx, yi, label, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved topic barplot: {out_path}")


def _save_scatter_freq_vs_sent(
    df: pd.DataFrame,
    label_col: str,
    count_col: str,
    sent_col: str,
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    top_n: int = 50,
) -> None:
    """
    Generic scatter: for keywords/entities/etc.

    - x: total_count   (sum of `count_col`)
    - y: mean_sentiment (mean of *_avg metric in `sent_col`)
    - label: label_col (keyword/entity/etc.)
    """
    if df.empty:
        print(f"[WARN] No data for scatter: {title}")
        return

    grouped = (
        df
        .groupby(label_col)
        .agg(
            total_count=(count_col, "sum"),
            mean_sentiment=(sent_col, "mean"),  # mean of sentiment_avg
        )
        .reset_index()
    )

    grouped = grouped.sort_values("total_count", ascending=False).head(top_n)

    if grouped.empty:
        print(f"[WARN] No data after grouping for scatter: {title}")
        return

    grouped = grouped.dropna(subset=["total_count", "mean_sentiment"])
    if grouped.empty:
        print(f"[WARN] No usable numeric data for scatter: {title}")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(grouped["total_count"], grouped["mean_sentiment"])

    for _, row in grouped.iterrows():
        plt.text(
            row["total_count"],
            row["mean_sentiment"],
            str(row[label_col]),
            fontsize=7,
            ha="center",
            va="bottom",
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved scatter: {out_path}")


def _save_domain_scatter(
    df: pd.DataFrame,
    domain_col: str,
    sent_col: str,
    out_path: str,
    title: str,
    xlabel: str = "Article count",
    ylabel: str = "Mean compound_avg",
    top_n: int = 50,
) -> None:
    """
    Scatter for top domains:

    - x: article_count (per domain)
    - y: mean_sentiment_avg (here: mean of compound_avg per domain)
    """
    df = df[[domain_col, sent_col]].dropna()
    if df.empty:
        print(f"[WARN] No data for domain scatter: {title}")
        return

    grouped = (
        df
        .groupby(domain_col)
        .agg(
            article_count=(sent_col, "size"),
            mean_sentiment=(sent_col, "mean"),  # mean of compound_avg
        )
        .reset_index()
    )

    grouped = grouped.sort_values("article_count", ascending=False).head(top_n)

    if grouped.empty:
        print(f"[WARN] No data after grouping for domain scatter: {title}")
        return

    grouped = grouped.dropna(subset=["article_count", "mean_sentiment"])
    if grouped.empty:
        print(f"[WARN] No usable numeric data for domain scatter: {title}")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(grouped["article_count"], grouped["mean_sentiment"])

    for _, row in grouped.iterrows():
        plt.text(
            row["article_count"],
            row["mean_sentiment"],
            str(row[domain_col]),
            fontsize=7,
            ha="center",
            va="bottom",
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved domain scatter: {out_path}")


def main(args):
    input_path = args.input
    output_dir = args.output

    print(f"[INFO] Reading parquet file: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"[INFO] Loaded {len(df)} rows")

    _ensure_output_dir(output_dir)

    # Parse publish_date to datetime
    df = _parse_publish_date(df)

    # All struct extractions use the "avg" field
    # compound["avg"] -> compound_avg
    _extract_struct_field(df, "compound", "compound_avg", "avg")

    # balance["avg"], skew["avg"]
    _extract_struct_field(df, "balance", "balance_avg", "avg")
    _extract_struct_field(df, "skew", "skew_avg", "avg")

    # 1. Compound average distribution (uses compound_avg)
    _save_hist(
        df["compound_avg"],
        os.path.join(output_dir, "compound_distribution.png"),
        title="Distribution of Article-level Compound Sentiment (avg)",
        xlabel="compound_avg",
        bins=40,
    )

    # 2. Compound over time (all domains) — using daily mean of compound_avg
    daily_compound = (
        df[["publish_dt", "compound_avg"]]
        .dropna()
        .groupby("publish_dt")["compound_avg"]
        .mean()  # mean of article-level compound_avg
        .reset_index()
    )

    _save_time_series(
        daily_compound,
        date_col="publish_dt",
        value_col="compound_avg",
        out_path=os.path.join(output_dir, "compound_over_time_all.png"),
        title="Mean Article Compound Sentiment Over Time (all domains)",
        ylabel="Mean compound_avg",
    )

    # 3. Compound over time by domain (media_url) — mean of compound_avg
    _save_time_series_by_domain(
        df,
        date_col="publish_dt",
        value_col="compound_avg",
        domain_col="media_url",
        out_path=os.path.join(output_dir, "compound_over_time_by_domain.png"),
        title="Mean Article Compound Sentiment Over Time by Domain",
        ylabel="Mean compound_avg",
        max_domains=6,
    )

    # 4. Balance & skew distributions — use balance_avg and skew_avg
    _save_hist(
        df["balance_avg"],
        os.path.join(output_dir, "balance_distribution.png"),
        title="Distribution of Balance (avg)",
        xlabel="balance_avg",
        bins=40,
    )

    _save_hist(
        df["skew_avg"],
        os.path.join(output_dir, "skew_distribution.png"),
        title="Distribution of Skew (avg)",
        xlabel="skew_avg",
        bins=40,
    )

    # 5. Manual keyword-wide graphs (all use sentiment_avg)
    manual_kw_df = _flatten_list_column(df["manual_keywords"])
    if not manual_kw_df.empty:
        # Bar plot: total count vs keyword, annotated with mean sentiment_avg
        _save_keyword_barplot(
            manual_kw_df,
            key_col="keyword",
            count_col="count",
            sent_col="sentiment_avg",
            out_path=os.path.join(output_dir, "manual_keywords_top.png"),
            title="Top Manual Keywords (Total Count, mean sentiment_avg annotated)",
            top_n=20,
        )

        # Scatter: total_count vs mean sentiment_avg
        _save_scatter_freq_vs_sent(
            manual_kw_df,
            label_col="keyword",
            count_col="count",
            sent_col="sentiment_avg",
            out_path=os.path.join(output_dir, "manual_keywords_scatter.png"),
            title="Manual Keywords: Frequency vs Sentiment (avg)",
            xlabel="Total count",
            ylabel="Mean sentiment_avg",
            top_n=50,
        )
    else:
        print("[WARN] manual_keywords column is empty or could not be flattened")

    # 6. Dynamic keywords-wide graphs (keywords column)
    dyn_kw_df = _flatten_list_column(df["keywords"])
    if not dyn_kw_df.empty:
        _save_keyword_barplot(
            dyn_kw_df,
            key_col="keyword",
            count_col="count",
            sent_col="sentiment_avg",
            out_path=os.path.join(output_dir, "dynamic_keywords_top.png"),
            title="Top Dynamic Keywords (Total Count, mean sentiment_avg annotated)",
            top_n=20,
        )

        _save_scatter_freq_vs_sent(
            dyn_kw_df,
            label_col="keyword",
            count_col="count",
            sent_col="sentiment_avg",
            out_path=os.path.join(output_dir, "dynamic_keywords_scatter.png"),
            title="Dynamic Keywords: Frequency vs Sentiment (avg)",
            xlabel="Total count",
            ylabel="Mean sentiment_avg",
            top_n=50,
        )
    else:
        print("[WARN] keywords column is empty or could not be flattened")

    # 7. Topic clusters graphs (use sentiment_avg)
    topics_df = _flatten_list_column(df["topic_clusters"])
    if not topics_df.empty and "topic" in topics_df.columns:
        _save_topic_barplot(
            topics_df,
            out_path=os.path.join(output_dir, "topics_top.png"),
            title="Top Topics (Total Count, mean sentiment_avg annotated)",
            top_n=20,
        )

        # Scatter: topic count vs mean topic sentiment_avg
        _save_scatter_freq_vs_sent(
            topics_df,
            label_col="topic",
            count_col="count",
            sent_col="sentiment_avg",
            out_path=os.path.join(output_dir, "topics_scatter.png"),
            title="Topics: Frequency vs Sentiment (avg)",
            xlabel="Total count",
            ylabel="Mean sentiment_avg",
            top_n=50,
        )
    else:
        print("[WARN] topic_clusters column is empty or lacks 'topic' field")

    # 8. Entities scatter (flatten entities; use sentiment_avg)
    ent_df = _flatten_list_column(df["entities"])
    if not ent_df.empty and "entity" in ent_df.columns:
        _save_scatter_freq_vs_sent(
            ent_df,
            label_col="entity",
            count_col="count",
            sent_col="sentiment_avg",
            out_path=os.path.join(output_dir, "entities_scatter.png"),
            title="Entities: Frequency vs Sentiment (avg)",
            xlabel="Total count",
            ylabel="Mean sentiment_avg",
            top_n=50,
        )
    else:
        print("[WARN] entities column is empty or lacks 'entity' field")

    # 9. Top domains scatter (article_count vs mean compound_avg)
    _save_domain_scatter(
        df,
        domain_col="media_url",
        sent_col="compound_avg",
        out_path=os.path.join(output_dir, "domains_scatter.png"),
        title="Domains: Article Count vs Mean Compound (avg)",
        xlabel="Article count",
        ylabel="Mean compound_avg",
        top_n=50,
    )

    # 10. Dataset summary file
    def _write_summary_file(df, output_dir):
        path = os.path.join(output_dir, "report_summary.txt")

        lines = []
        lines.append("==== DATASET SUMMARY ====\n")

        # Basic
        lines.append(f"Total articles: {len(df)}")
        lines.append(f"Unique media domains: {df['media_url'].nunique()}")
        lines.append(f"Date range: {df['publish_dt'].min()}  ->  {df['publish_dt'].max()}")
        lines.append(f"Rows missing publish date: {df['publish_dt'].isna().sum()}")
        lines.append("")

        # Sentiment distribution
        for col in ["compound_avg", "balance_avg", "skew_avg"]:
            s = pd.to_numeric(df[col], errors='coerce')
            lines.append(f"--- {col} ---")
            lines.append(f"  count: {s.count()}")
            lines.append(f"  avg:   {s.mean():.5f}")
            lines.append(f"  med:   {s.median():.5f}")
            lines.append(f"  min:   {s.min():.5f}")
            lines.append(f"  max:   {s.max():.5f}")
            lines.append(f"  std:   {s.std():.5f}")
            lines.append("")

        # Keywords / entities
        def count_unique(df_listcol, field):
            df_flat = _flatten_list_column(df_listcol)
            if df_flat.empty or field not in df_flat.columns:
                return 0
            return df_flat[field].nunique()

        lines.append("Manual keywords present: " + str(count_unique(df["manual_keywords"], "keyword")))
        lines.append("Dynamic keywords present: " + str(count_unique(df["keywords"], "keyword")))
        lines.append("Entities present: " + str(count_unique(df["entities"], "entity")))
        lines.append("Topics present: " + str(count_unique(df["topic_clusters"], "topic")))
        lines.append("")

        # Top domains
        domain_counts = df["media_url"].value_counts()
        lines.append("Top 10 domains by article count:")
        for d, c in domain_counts.head(10).items():
            lines.append(f"  {d}: {c}")
        lines.append("")

        # Domain sentiment
        domain_sent = (
            df.groupby("media_url")["compound_avg"]
            .mean()
            .sort_values(ascending=False)
        )

        if not domain_sent.empty:
            lines.append(f"Highest sentiment domain: {domain_sent.index[0]} ({domain_sent.iloc[0]:.4f})")
            lines.append(f"Lowest sentiment domain:  {domain_sent.index[-1]} ({domain_sent.iloc[-1]:.4f})")
        lines.append("")

        # Validation metrics
        lines.append("Validation:")
        lines.append(f"Missing text: {df['text'].isna().sum()}")
        lines.append(f"Empty tokens lists: {df['tokens'].apply(lambda x: len(x) if isinstance(x, list) else 0).eq(0).sum()}")
        lines.append(f"Empty sentence_sentiments: {df['sentence_sentiments'].apply(lambda x: len(x) if isinstance(x, list) else 0).eq(0).sum()}")
        lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[OK] Wrote summary file: {path}")

    _write_summary_file(df, output_dir)
    

    print("[INFO] Report generation complete.")
