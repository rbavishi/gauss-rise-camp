import itertools
import random
import pandas as pd

from gauss.domains.pandas_lite.generators import candidates_groupby_agg_op, candidates_groupby_transform_op
from gauss.domains.utils.dfgen import generate_random_dataframe, DfConfig

MAX_ROWS = 4
MIN_ROWS = 1
MAX_COLS = 4
MIN_COLS = 1


def datagen_default(seed: int):
    return [generate_random_dataframe(DfConfig(min_width=MIN_COLS, max_width=MAX_COLS,
                                               min_height=MIN_ROWS, max_height=MAX_ROWS,
                                               max_index_levels=1, max_column_levels=1))], [], {}


def datagen_groupby_agg(seed: int):
    while True:
        op = candidates_groupby_agg_op[seed % len(candidates_groupby_agg_op)]
        num_group_cols = (seed // len(candidates_groupby_agg_op)) % 2 + 1

        df = generate_random_dataframe(DfConfig(min_width=max(MIN_COLS, num_group_cols + 1),
                                                max_width=MAX_COLS,
                                                min_height=MIN_ROWS, max_height=MAX_ROWS,
                                                max_index_levels=1, max_column_levels=1,
                                                nan_prob=0.4 if op in ['all', 'any'] else 0.05))

        #  Find group columns such that there is at least one duplicate key
        for group_cols in itertools.combinations(list(df.columns), num_group_cols):
            if len(df.groupby(list(group_cols)).groups) < df.shape[0]:
                return [df], [], {'groupby_agg_by_cols': [list(group_cols)],
                                  'groupby_agg_op': [op]}


def datagen_groupby_transform(seed: int):
    while True:
        op = candidates_groupby_transform_op[seed % len(candidates_groupby_transform_op)]
        num_group_cols = (seed // len(candidates_groupby_transform_op)) % 2 + 1

        df = generate_random_dataframe(DfConfig(min_width=max(MIN_COLS, num_group_cols + 1),
                                                max_width=MAX_COLS,
                                                min_height=MIN_ROWS, max_height=MAX_ROWS,
                                                max_index_levels=1, max_column_levels=1))

        #  Find group columns such that there is at least one duplicate key
        for group_cols in itertools.combinations(list(df.columns), num_group_cols):
            if len(df.groupby(list(group_cols)).groups) < df.shape[0]:
                return [df], [], {'groupby_transform_by_cols': [list(group_cols)],
                                  'groupby_transform_op': [op]}


def datagen_pivot_table(seed: int):
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
    index_cols = []
    for idx, index_vals in enumerate(new_index_col_vals):
        df[f"_IDX_COL{idx}"] = index_vals
        index_cols.append(f"_IDX_COL{idx}")

    df["_COLS_COL"] = new_column_vals
    df["_VALS_COL"] = values
    new_cols = list(df.columns)
    random.shuffle(new_cols)
    df = df[new_cols]

    return [df], [], {
        "pivot_columns": ["_COLS_COL"],
        "pivot_values": ["_VALS_COL"],
        "pivot_index": [index_cols],
    }


def datagen_filtering_expr(seed: int):
    while True:
        try:
            df = generate_random_dataframe(DfConfig(min_width=MIN_COLS,
                                                    max_width=MAX_COLS,
                                                    min_height=MIN_ROWS, max_height=MAX_ROWS,
                                                    max_index_levels=1, max_column_levels=1))
            column = random.choice(list(df.columns))
            value = random.choice(list(df[column]))
            op = [">", "<", "==", "!="][seed % 4]

            expr = f"`{column}` {op} {value!r}"
            df.query(expr)

            return [df], [expr], {
                "filtering_expr_expression": [expr]
            }

        except:
            pass


def datagen_filtering_contains(seed: int):
    while True:
        try:
            df = generate_random_dataframe(DfConfig(min_width=MIN_COLS,
                                                    max_width=MAX_COLS,
                                                    min_height=MIN_ROWS, max_height=MAX_ROWS,
                                                    max_index_levels=1, max_column_levels=1))
            column = random.choice(list(df.columns))
            values = set(random.sample(list(df[column]), random.randint(1, df.shape[1] - 1)))

            return [df], [values], {
                "filtering_contains_filter_col": [column],
                "filtering_contains_collection": [values],
            }

        except:
            pass


def datagen_fillna(seed: int):
    mode = ["method", "value"][seed % 2]
    axis, method = list(itertools.product(["index", "columns"], ["backfill", "pad"]))[(seed // 2) % 4]

    while True:
        try:
            df = generate_random_dataframe(DfConfig(min_width=MIN_COLS,
                                                    max_width=MAX_COLS,
                                                    min_height=MIN_ROWS, max_height=MAX_ROWS,
                                                    max_index_levels=1, max_column_levels=1,
                                                    nan_prob=0.5))

            if mode == "value":
                df.fillna(0)
                return [df], [0], {
                    "fillna_mode": ["value"]
                }

            else:
                return [df], [], {
                    "fillna_method": [method],
                    "fillna_axis": [axis],
                }

        except:
            pass


def datagen_dropna(seed: int):
    df = generate_random_dataframe(DfConfig(min_width=MIN_COLS,
                                            max_width=MAX_COLS,
                                            min_height=MIN_ROWS, max_height=MAX_ROWS,
                                            max_index_levels=1, max_column_levels=1,
                                            nan_prob=0.5))

    return [df], [], {}


def datagen_merge(seed: int):
    while True:
        df1 = generate_random_dataframe(DfConfig(min_width=MIN_COLS,
                                                 max_width=MAX_COLS,
                                                 min_height=MIN_ROWS, max_height=MAX_ROWS,
                                                 max_index_levels=1,
                                                 max_column_levels=1))
        on_columns = random.sample(list(df1.columns), random.randint(1, df1.shape[1]))
        df2_width = random.randint(len(on_columns), MAX_COLS)
        df2 = generate_random_dataframe(DfConfig(num_cols=df2_width,
                                                 min_height=MIN_ROWS, max_height=MAX_ROWS,
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
            res = df1.merge(df2)
            if res.shape[0] == 0 or res.shape[1] == 0:
                continue

        except:
            continue

        return [df1, df2], [], {}


def datagen_combine_first(seed: int):
    while True:
        df1 = generate_random_dataframe(DfConfig(min_width=MIN_COLS,
                                                 max_width=MAX_COLS,
                                                 min_height=MIN_ROWS, max_height=MAX_ROWS,
                                                 max_index_levels=1,
                                                 nan_prob=0.5,
                                                 max_column_levels=1))
        on_columns = random.sample(list(df1.columns), random.randint(1, df1.shape[1]))
        df2_width = random.randint(len(on_columns), MAX_COLS)
        df2 = generate_random_dataframe(DfConfig(num_cols=df2_width,
                                                 min_height=MIN_ROWS, max_height=MAX_ROWS,
                                                 max_index_levels=1,
                                                 max_column_levels=1,
                                                 nan_prob=0.5,
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
            df1 = df1.T
            df2 = df2.T
            res = df1.combine_first(df2)
            if res.shape[0] == 0 or res.shape[1] == 0:
                continue

        except:
            continue

        return [df1, df2], [], {}


def datagen_separate(seed: int):
    while True:
        try:
            df = generate_random_dataframe(DfConfig(min_width=2, max_width=MAX_COLS,
                                                    min_height=MIN_ROWS, max_height=MAX_ROWS,
                                                    max_index_levels=1, max_column_levels=1))

            unite_cols = random.sample(list(df.columns), random.randint(2, df.shape[1]))
            df = df.drop(columns=unite_cols).assign(MY_NEW_COL=df[unite_cols[0]].str.cat(df[unite_cols[1:]], sep='_'))
            return [df], [], {"separate_col": ["MY_NEW_COL"]}

        except:
            pass


datagen_dict = {
    "pd.groupby_agg": datagen_groupby_agg,
    "pd.groupby_transform": datagen_groupby_transform,
    "pd.pivot_table": datagen_pivot_table,
    "pd.filtering_expr": datagen_filtering_expr,
    "pd.filtering_contains": datagen_filtering_contains,
    'pd.melt': datagen_default,
    'pd.drop_columns': datagen_default,
    'pd.fillna': datagen_fillna,
    'pd.dropna': datagen_dropna,
    'pd.merge': datagen_merge,
    'pd.combine_first': datagen_combine_first,
    'pd.unite': datagen_default,
    'pd.separate': datagen_separate,
    'pd.mutate': datagen_default,
}
