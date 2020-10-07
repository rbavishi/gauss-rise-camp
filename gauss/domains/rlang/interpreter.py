import re

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union, Optional

pd_dfgroupby = pd.core.groupby.DataFrameGroupBy


class RInterpreter:
    # Re-implementations of tidyr/dplyr R functions for reshaping that were used in examples from the Morpheus paper.

    # "Melts" the table, creating a "var" and "value" column where for each (column_name, [column_values]),
    # a set of rows is created with var = column_name and value = column_values[i]. Optionally, only melt the columns
    # in value_vars and keep the columns in `id_vars` as indices, i.e. unmelted.
    # https://www.rdocumentation.org/packages/tidyr/versions/0.8.2/topics/gather
    @staticmethod
    def gather(input_frame: pd.DataFrame, var_name: str = 'var', value_name: str = 'value',
               value_vars: List[str] = None, id_vars: List[str] = None) -> pd.DataFrame:
        assert(id_vars or value_vars)
        if id_vars is None:
            id_vars = [i for i in input_frame.columns if i not in value_vars]
        elif value_vars is None:
            value_vars = [i for i in input_frame.columns if i not in id_vars]

        ret_frame: pd.DataFrame = input_frame.melt(id_vars, value_vars, var_name, value_name)
        # print(ret_frame, value_vars, id_vars)
        return ret_frame

    # Creates a new column that concatenates the values in `cols` with `sep`, and drops the original columns.
    # https://www.rdocumentation.org/packages/tidyr/versions/0.8.2/topics/unite
    @staticmethod
    def unite(input_frame: pd.DataFrame, cols: List[str], new_col_name: str = "Default_XYSZ", sep: str = "_"):
        new_col = None
        for i in range(len(cols)):
            col = cols[i]
            if i == 0:
                new_col = input_frame[col].astype(str)
            else:
                new_col = new_col + sep + input_frame[col].astype(str)
        new_frame = input_frame.drop(columns=cols)
        new_frame[new_col_name] = new_col
        return new_frame

    # "Spreads" the table so that the values in the column `columns` become column names with the corresponding row
    # values in `values` becoming the values in the new columns. Columns other than `columns` and `values` are kept
    # as indices. https://www.rdocumentation.org/packages/tidyr/versions/0.8.2/topics/spread
    @staticmethod
    def spread(input_frame: pd.DataFrame, columns: str, values: str):
        index_cols = list(input_frame.columns)
        index_cols.remove(columns)
        index_cols.remove(values)
        # TODO: should I get rid of agg_func first or
        ret_frame: pd.DataFrame = input_frame.pivot_table(index=list(index_cols) or input_frame.index, columns=columns,
                                                          values=values, aggfunc='first')
        # print(ret_frame, ret_frame.index, list(index_cols) or None)
        if len(index_cols) > 0:
            ret_frame = ret_frame.reset_index()

        ret_frame.columns.name = None
        return ret_frame

    # Keeps (and possibly reorders) the columns in `columns_keep`, then drops any columns in `columns_remove`.
    # Slightly less expressive than the original R function, but Morpheus only synthesizes a restricted subset of the
    # use, it looks like. Removed support for numerical indexing since this is identical to column-name indexing (
    # albeit less easy to read a reordering).
    # https://www.rdocumentation.org/packages/dplyr/versions/0.7.8/topics/select
    @staticmethod
    def select(input_frame: pd.DataFrame, columns_keep: List[str] = None, columns_remove: List[str] = None):
        if (columns_keep is None) and (columns_remove is None):
            raise ValueError("One of columns_keep or columns_remove must not be none.")
        ret_frame = input_frame
        if columns_keep is not None:
            ret_frame = input_frame[columns_keep]
        else:
            cols_remaining = list(ret_frame.columns)
            for remove_col in columns_remove:
                cols_remaining.remove(remove_col)
            ret_frame = ret_frame[cols_remaining]
        return ret_frame

    # Turns a single string column into multiple columns, splitting at non alpha-numeric characters in the columns
    # values. https://www.rdocumentation.org/packages/tidyr/versions/0.8.2/topics/separate
    @staticmethod
    def separate(input_frame: pd.DataFrame, split_col: str, into: List[str]):
        orig_col = input_frame[split_col]
        ret_frame = input_frame.drop(columns=split_col)
        split_values = [re.compile("[^a-zA-Z0-9]+").split(str(x)) for x in orig_col]
        longest_item_len = max([len(val) for val in split_values])
        split_values_filled = [val + [np.nan] * (longest_item_len - len(val)) for val in split_values]
        split_series = list(zip(*split_values_filled))
        for i in range(len(into)):
            if i >= len(split_series):
                split_series.append((np.NaN,) * len(orig_col))
            # Turn things into numbers if we can.
            ret_frame[into[i]] = pd.to_numeric(split_series[i], errors='ignore')
        return ret_frame

    # Returns a group by object where the table is grouped into the sets of unique
    # values in the columns `group_cols`s
    @staticmethod
    def group_by(input_frame: pd.DataFrame, group_cols: List[str]) -> pd_dfgroupby:
        if not group_cols:
            raise ValueError("Need a non-empty list of columns to group by.")
        return input_frame.groupby(group_cols, as_index=False)

    # Applies `summaries` to the group_by object, where `summaries` is a dict of:
    #    - keys: new name for the column with the summaries
    #    - values: a tuple of (summary_col, function), where summary_col is the column to apply
    #              the summary function `function` to.
    #              OR
    #              a function to apply over everything (always count?)
    # https://www.rdocumentation.org/packages/dplyr/versions/0.7.8/topics/summarise
    @staticmethod
    def summarise(input_obj: Union[pd_dfgroupby, pd.DataFrame],
                  summaries: Dict[str, Tuple[Optional[str], str]]):
        if len(summaries) > 1:
            raise NotImplementedError('Cannot handle more than one summarization')

        new_col = list(summaries.keys())[0]
        col_id, operation = list(summaries.values())[0]

        if operation == 'count':
            if isinstance(input_obj, pd.DataFrame):
                raise NotImplementedError("count does not make sense for dataframe inputs")

            summarized_frame = input_obj.size().reset_index(name=new_col)
            return summarized_frame

        elif operation in ['sum', 'mean']:
            if isinstance(input_obj, pd.DataFrame):
                #  Transpose required to bring the column name back to being a column name
                #  instead of an index in the resulting series after agg
                summarized_frame = pd.DataFrame(input_obj.agg({col_id: operation})).T
                summarized_frame.rename(columns={col_id: new_col}, inplace=True)
                return summarized_frame

            elif isinstance(input_obj, pd_dfgroupby):
                summarized_frame = input_obj.agg({col_id: operation})
                summarized_frame.rename(columns={col_id: new_col}, inplace=True)
                return summarized_frame

        else:
            raise NotImplementedError("Cannot handle operation {} right now".format(operation))

    # Add a new column named `new_col_name` with the values from `new_col_values`
    # https://www.rdocumentation.org/packages/dplyr/versions/0.7.8/topics/mutate
    @staticmethod
    def mutate(input_frame: pd.DataFrame, new_col_name: str, operation: str, col_args: Union[str, List[str]]):
        if operation == 'normalize':
            new_col_values = input_frame[col_args] / sum(input_frame[col_args])
        elif operation in ['div']:
            new_col_values = input_frame[col_args[0]] / input_frame[col_args[1]]
        else:
            raise NotImplementedError("Operation {} not implemented".format(operation))

        return input_frame.assign(**{new_col_name: new_col_values})

    # Filters the input frame to only have rows which satisfy `filter_expr`. Renamed to filter_ to avoid overloading the
    # native python filter function.
    # https://www.rdocumentation.org/packages/dplyr/versions/0.7.8/topics/filter
    # TODO for the generator, should only be == or >
    @staticmethod
    def filter_(input_frame: pd.DataFrame, filter_expr: str, reset_index=True):
        result = input_frame.query(filter_expr)
        #  Reset the index as it carries the old index values.
        #  For example, it may contain [1, 5, 6] as the indices, whereas we want [0, 1, 2]
        if reset_index:
            return result.reset_index(drop=True)
        else:
            return result

    # Performs an inner join on `input_frame_1` and `input_frame_2`.
    # https://www.rdocumentation.org/packages/dplyr/versions/0.7.8/topics/join
    @staticmethod
    def inner_join(input_frame_1: pd.DataFrame, input_frame_2: pd.DataFrame):
        return pd.DataFrame.merge(input_frame_1, input_frame_2, how='inner')
