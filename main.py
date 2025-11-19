import argparse
import pathlib
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
    p_csv_manip = subparsers.add_parser("data-manip", help="Manipulate a Media Cloud CSV file")
    p_csv_manip.add_argument("-i", "--input", required=True, help="Media Cloud dataset file for input", type=pathlib.Path)
    p_csv_manip.add_argument("-q", "--pd_query", help="The pandas style query to draw from (Default: None)")
    p_csv_manip.add_argument("--columns", nargs="+", help="Optional list of columns to keep in the output")
    p_csv_manip.add_argument("-o", "--output", type=pathlib.Path, required=True, help="Output CSV file for the sub-sample")

    csv_subparsers = p_csv_manip.add_subparsers(
        dest="function",
        required=True,
        title="Functions"
    )

    ## sub sampler args
    csv_sub_sample = csv_subparsers.add_parser("sub-sample", help="Sub Sample the CSV File (using pandas sampling)")

    size_group = csv_sub_sample.add_mutually_exclusive_group(required=True)
    size_group.add_argument("-n", "--n-rows", type=int, help="Number of rows to sample")
    size_group.add_argument("-f", "--frac", type=float, help="Fraction of rows to sample (0 < frac ≤ 1)")

    csv_sub_sample.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    csv_sub_sample.add_argument("--replace", action="store_true", help="Sample with replacement (default: without replacement)")

    csv_sub_sample.set_defaults(func=csvManip.main, callback=csvManip.sub_sample)

    ## data cleaner args
    csv_clean_data = csv_subparsers.add_parser("clean-data", help="Removes Duplicate articles and cleans up the data")


    ## getter args
    p_get_articles = subparsers.add_parser("get-articles", help="Downloads the Articles from the given URL")
    p_get_articles.add_argument("-i", "--input", required=True, help="Media Cloud CSV File for input")
    p_get_articles.add_argument("-o", "--output", required=True, help="Where to write output (does not maintain order)")
    p_get_articles.add_argument("--con-gets", default=50, help="# of concurrent HTTP Get Requests from given URLs")
    p_get_articles.add_argument("--cores", default=4, help="# of cores to use for analysis step")
    p_get_articles.set_defaults(func=getArticles.main)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    try: args.func(args, args.callback)
    except Exception as e:
        print(e)
        parser.print_help()

if __name__ == "__main__":
    main()