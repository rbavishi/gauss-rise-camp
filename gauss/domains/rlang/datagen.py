"""
Routines for random data generation.
Specifically, contains specialized routines for part of the domain which return an input as well as choices
that need to be made by the generator in order to produce a meaningful transformation.
"""
import itertools
import random

import pandas as pd

from gauss.domains.rlang.interpreter import RInterpreter
from gauss.domains.utils.dfgen import generate_random_dataframe, DfConfig


MAX_ROWS = 4
MIN_ROWS = 1
MAX_COLS = 4
MIN_COLS = 1


def datagen_default():
    return [generate_random_dataframe(DfConfig(min_width=MIN_COLS, max_width=MAX_COLS,
                                               min_height=MIN_ROWS, max_height=MAX_ROWS,
                                               max_index_levels=1, max_column_levels=1))], {}


def datagen_spread():
    num_index_values = random.choice([1, 2, 3])
    num_idx_cols = random.choice([1, 2, 3])
    index_col_vals = []
    for idx in range(num_idx_cols):
        if random.choice([True, False]):
            index_col_vals.append([f"_S{idx}_VAL_{i}" for i in range(num_index_values)])
        else:
            index_col_vals.append([i + random.randint(10, 1000) for i in range(num_index_values)])

    if random.choice([True, False]):
        column_vals = [f"_SVAL_{i + 3}" for i in range(random.choice([1, 2, 3]))]
    else:
        column_vals = [i + random.randint(1001, 2001) for i in range(random.choice([1, 2, 3]))]

    if random.choice([True, False]):
        values = [random.randint(10, 100000) for _ in range(num_index_values * len(column_vals))]
    else:
        values = [round(random.uniform(10, 100000), 2) for _ in range(num_index_values * len(column_vals))]

    new_index_col_vals = []
    new_column_vals = []
    for index_vals in index_col_vals[1:]:
        index_vals, new_column_vals = list(zip(*itertools.product(index_vals, column_vals)))
        new_index_col_vals.append(index_vals)

    df = pd.DataFrame()
    for idx, index_vals in enumerate(new_index_col_vals):
        df[f"_IDX_COL{idx}"] = index_vals

    df["_COLS_COL"] = new_column_vals
    df["_VALS_COL"] = values
    new_cols = list(df.columns)
    random.shuffle(new_cols)
    df = df[new_cols]

    return [df], {
        "spread_columns": ["_COLS_COL"],
        "spread_values": ["_VALS_COL"],
    }


def datagen_separate():
    df = generate_random_dataframe(DfConfig(min_width=3,
                                            max_index_levels=1,
                                            max_column_levels=1))
    unite_cols = random.sample(list(df.columns), random.choice([2, 3]))
    df["NEW-VALS"] = ["@".join(map(str, vals)) for vals in zip(*[df[c] for c in unite_cols])]
    df.drop(columns=unite_cols, inplace=True)
    new_cols = list(df.columns)
    random.shuffle(new_cols)
    return [df], {
        "separate_split_col": ["NEW-VALS"]
    }


def datagen_mutate():
    operation = random.choice(["normalize", "div"])
    if operation == 'normalize':
        while True:
            df = generate_random_dataframe(DfConfig(min_width=2,
                                                    max_width=MAX_COLS,
                                                    min_height=MIN_ROWS,
                                                    max_height=MAX_ROWS,
                                                    max_index_levels=1,
                                                    max_column_levels=1))
            numeric_cols = df.select_dtypes('number').columns
            if len(numeric_cols) == 0:
                continue

            return [df], {
                "mutate_operation": ["normalize"],
                "mutate_col_args_normalize": [random.choice(numeric_cols)]
            }

    else:
        while True:
            df = generate_random_dataframe(DfConfig(min_width=2,
                                                    max_width=MAX_COLS,
                                                    min_height=MIN_ROWS,
                                                    max_height=MAX_ROWS,
                                                    max_index_levels=1,
                                                    max_column_levels=1))
            numeric_cols = df.select_dtypes('number').columns
            if len(numeric_cols) < 2:
                continue

            col_args = random.sample(list(numeric_cols), 2)
            return [df], {
                "mutate_operation": ["div"],
                "mutate_col_arg1": [col_args[0]],
                "mutate_col_arg2": [col_args[1]],
            }


def datagen_filter():
    mode = random.choice(["equality-inequality", "relop"])
    if mode == "equality-inequality":
        while True:
            try:
                df = generate_random_dataframe(DfConfig(min_width=2,
                                                        max_width=MAX_COLS,
                                                        min_height=MIN_ROWS,
                                                        max_height=MAX_ROWS,
                                                        max_index_levels=1,
                                                        max_column_levels=1))
                col = random.choice(list(df.columns))
                value = random.choice(list(set(df.loc[:, col])))
                op = random.choice(["==", "!="])

                return [df], {
                    "filter_mode": [mode],
                    "filter_column_eq": [col],
                    "filter_value_eq": [value],
                    "filter_eq_op": [op]
                }

            except:
                pass

    else:
        while True:
            try:
                df = generate_random_dataframe(DfConfig(min_width=2,
                                                        max_width=MAX_COLS,
                                                        min_height=MIN_ROWS,
                                                        max_height=MAX_ROWS,
                                                        max_index_levels=1,
                                                        max_column_levels=1))

                numeric_cols = df.select_dtypes('number').columns
                if len(numeric_cols) < 1:
                    continue

                col = random.choice(list(numeric_cols))
                value = random.choice(list(set(df.loc[:, col])))
                op = random.choice(["<", ">"])

                return [df], {
                    "filter_mode": [mode],
                    "filter_column_relop": [col],
                    "filter_value_relop": [value],
                    "filter_relop": [op]
                }

            except:
                pass


def datagen_inner_join():
    while True:
        df1 = generate_random_dataframe(DfConfig(min_width=MIN_COLS,
                                                 max_width=3,
                                                 max_index_levels=1,
                                                 max_column_levels=1))
        on_columns = random.sample(list(df1.columns), random.randint(1, df1.shape[1]))
        df2_width = random.randint(len(on_columns), MAX_COLS)
        df2 = generate_random_dataframe(DfConfig(num_cols=df2_width,
                                                 max_index_levels=1,
                                                 max_column_levels=1,
                                                 col_prefix="DF2"))
        replaced_cols = random.sample(list(df2.columns), len(on_columns))
        df2 = df2.rename(columns=dict(zip(replaced_cols, on_columns)))

        df1_items = [tuple(i) for i in df1.loc[:, on_columns].values]
        df2_items = [tuple(i) for i in df2.loc[:, on_columns].values]

        new_df1_items = random.sample(df1_items + df2_items, df1.shape[0])
        new_df2_items = random.sample(df1_items + df2_items, df2.shape[0])

        for idx, items in enumerate(new_df1_items):
            df1.loc[idx, on_columns] = items

        for idx, items in enumerate(new_df2_items):
            df2.loc[idx, on_columns] = items

        try:
            res = RInterpreter.inner_join(df1, df2)
            if res.shape[0] == 0 or res.shape[1] == 0:
                continue

        except:
            continue

        return [df1, df2], {}


datagen_dict = {
    "rf.gather": datagen_default,
    "rf.spread": datagen_spread,
    "rf.separate": datagen_separate,
    "rf.unite": datagen_default,
    "rf.group_by": datagen_default,
    "rf.select": datagen_default,
    "rf.mutate": datagen_mutate,
    "rf.filter": datagen_filter,
    "rf.inner_join": datagen_inner_join,
}

