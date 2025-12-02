#!/usr/bin/env python3
import argparse
from sys import exit
from programs import csvManip, getArticles

def build_parser():
    parser = argparse.ArgumentParser(
        description="Talia Jolyne Ross COMP-4112 Final Project using Media Cloud"
    )

    subparsers = parser.add_subparsers(
        dest="program", 
        required=True,
        title="Programs"
    )

    ## base data manip args
    p_csv_manip = subparsers.add_parser("clean-data", help="Manipulate a Media Cloud CSV file")
    p_csv_manip.add_argument("-i", "--input", required=True, help="Media Cloud dataset file for input")
    p_csv_manip.add_argument("-q", "--pd_query", help="The pandas style query to draw from (Default: None)")
    p_csv_manip.add_argument("--columns", nargs="+", help="Optional list of columns to keep in the output")
    p_csv_manip.add_argument("-o", "--output", required=True, help="Output CSV file for the sub-sample")

    size_group = p_csv_manip.add_mutually_exclusive_group(required=True)
    size_group.add_argument("-n", "--n-rows", type=int, help="Number of rows to sample")
    size_group.add_argument("-f", "--frac", type=float, help="Fraction of rows to sample (0 < frac ≤ 1)")

    p_csv_manip.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    p_csv_manip.add_argument("--replace", action="store_true", help="Sample with replacement (default: without replacement)")

    p_csv_manip.set_defaults(func=csvManip.main)

    ## getter args
    p_get_articles = subparsers.add_parser("get-articles", help="Downloads the Articles from the given URL")
    p_get_articles.add_argument("-i", "--input", required=True, help="Media Cloud CSV File for input")
    p_get_articles.add_argument("-o", "--output", required=True, help="Where to write output (dir)")

    retrieval_args = p_get_articles.add_argument_group("Article Retrieval Arguments")
    retrieval_args.add_argument("-cg", "--con-gets", default=50, type=int, help="# of concurrent HTTP Get Requests from given URLs (default 50)")
    retrieval_args.add_argument("-cd", "--con-dom", default=2, type=int, help="# of concurrent HTTP Gets per domain (default 2)")
    retrieval_args.add_argument("-d", "--min-del", default=0.5, type=float, help="Minimun delay for requets per domain in seconds (default 0.5, 0 to disable)")

    analysis_args = p_get_articles.add_argument_group("Article Analysis Arguments")
    analysis_args.add_argument("-c", "--cores", default=4, type=int, help="# of cores to use for analysis (default 4)")

    writer_args = p_get_articles.add_argument_group("Output dataset Writing arguments")
    writer_args.add_argument("-t", "--text", default=True, type=bool, help="Whether not to write cleaned text to its own file (default: True)")
    writer_args.add_argument("-f", "--flush", default=100, type=int, help="The # of rows to trigger flushing the writing buffer (default: 100)")

    p_get_articles.set_defaults(func=getArticles.main, callback=None)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    args.func(args)

if __name__ == "__main__":
    main()