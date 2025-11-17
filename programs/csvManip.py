import pandas as pd

def sub_sample(args, data: pd.DataFrame) -> pd.DataFrame:
    return data.sample(
        n = args.n_rows if args.n_rows != None else None,
        frac = args.frac if args.frac != None else None,
        replace = args.replace,
        random_state = args.seed if args.seed != None else None,
    )

def main(args, callback):
    data = pd.read_csv(args.input)
    data = data.query(args.pd_query)
    callback(args, data).to_csv(args.output, columns=args.columns)
