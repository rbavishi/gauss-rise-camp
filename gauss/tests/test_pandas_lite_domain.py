import unittest

import pandas as pd
import numpy as np

from gauss.domains.pandas_lite.definition import PandasLiteSynthesisDomain


class TestDataFrameGraph(unittest.TestCase):
    def test_df_graph(self):
        from gauss.domains.pandas_lite.graphs import DataFrameGraph
        df = pd.DataFrame([['a', 'b', 'e'], ['c', 'd', 'f']], columns=['C1', 'C2', 'C3'])
        df_graph = DataFrameGraph(df)

        #  Check if all the nodes have been created.
        self.assertListEqual(list(df.columns), [c.value for c in df_graph.columns])
        self.assertListEqual(list(df.index), [c.value for c in df_graph.index])
        for row_df, row_df_graph in zip(df.values, df_graph.values):
            self.assertListEqual(list(row_df), [v.value for v in row_df_graph])

        #  Check indexers
        self.assertEqual('d', df_graph.iloc[1, 1].value)
        self.assertEqual('d', df_graph.loc[1, 'C2'].value)


class TestDataGen(unittest.TestCase):
    def test_basic_gen(self):
        domain = PandasLiteSynthesisDomain()
        for name in domain.get_available_components():
            for i in range(30):
                entry = domain.generate_witness_entry(name, seed=i)
                print(entry.program)


class TestGenerators(unittest.TestCase):
    def test_groupby(self):
        from gauss.domains.pandas_lite.generators import gen_groupby_agg, gen_groupby_transform, DataFrameGraph

        df = pd.DataFrame([['A', 100], ['A', 200], ['B', 300]], columns=['C1', 'C2'])
        g_df = DataFrameGraph(df)
        result, call_str, graph, res_graph = gen_groupby_agg.with_env(ignore_exceptions=False).call(df, g_df, [])

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')))

        df = pd.DataFrame([['A', 100], ['A', 200], ['B', 300]], columns=['C1', 'C2'])
        g_df = DataFrameGraph(df)
        result, call_str, graph, res_graph = gen_groupby_transform.with_env(ignore_exceptions=False).call(df, g_df, [])

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')))

    def test_pivot_table(self):
        from gauss.domains.pandas_lite.generators import gen_pivot_table, DataFrameGraph

        df = pd.DataFrame([['c', 'C2', 'd'], ['a', 'C2', 'b'], ['a', 'C3', 'e']],
                          columns=['C1', 'var', 'value'])

        g_df = DataFrameGraph(df)
        result, call_str, graph, res_graph = gen_pivot_table.with_env(ignore_exceptions=False).call(df, g_df, [])

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result, check_names=False)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')), check_names=False)

    def test_fillna(self):
        from gauss.domains.pandas_lite.generators import gen_fillna, DataFrameGraph

        df = pd.DataFrame({'col1': {0: 'A1', 1: 'A1', 2: 'A2', 3: 'A2'},
                           'col2': {0: 'B1', 1: 'B2', 2: 'B1', 3: 'B2'},
                           'after': {0: 13.0, 1: 21.0, 2: np.nan, 3: 22.0},
                           'before': {0: 20.0, 1: 11.0, 2: 18.0, 3: np.nan}})

        g_df = DataFrameGraph(df)
        result, call_str, graph, res_graph = gen_fillna.with_env(ignore_exceptions=False,
                                                                 replay={
                                                                     "fillna_mode": ["method"],
                                                                     "fillna_method": ["backfill"]
                                                                 }).call(df, g_df, [])

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result, check_names=False)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')), check_names=False)

        result, call_str, graph, res_graph = gen_fillna.with_env(ignore_exceptions=False,
                                                                 replay={
                                                                     "fillna_mode": ["value"],
                                                                 }).call(df, g_df, [0])

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result, check_names=False)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')), check_names=False)

    def test_merge(self):
        from gauss.domains.pandas_lite.generators import gen_merge, DataFrameGraph

        df1 = pd.DataFrame([['a', 'b', 'c'], ['d', 'g', 'c'], ['f', 'b', 'h']], columns=['c1', 'c2', 'c3'])
        df2 = pd.DataFrame([['x', 'g', 'z'], ['w', 'b', 'u'], ['y', 'g', 'j']], columns=['c4', 'c2', 'c5'])

        g_df1 = DataFrameGraph(df1)
        g_df2 = DataFrameGraph(df2)

        result, call_str, graph, res_graph = gen_merge.with_env(ignore_exceptions=True).call(df1, df2, g_df1, g_df2,
                                                                                             [])

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df1, result, check_names=False)
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df2, result, check_names=False)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df1', inp2='df2')), check_names=False)

    def test_unite(self):
        from gauss.domains.pandas_lite.generators import gen_unite, DataFrameGraph

        df = pd.DataFrame([['c', 'C2', 'd'], ['a', 'C2', 'b'], ['a', 'C3', 'e']],
                          columns=['C1', 'var', 'value'])

        g_df = DataFrameGraph(df)
        result, call_str, graph, res_graph = gen_unite.with_env(ignore_exceptions=False).call(df, g_df, [])

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result, check_names=False)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')), check_names=False)

    def test_separate(self):
        from gauss.domains.pandas_lite.generators import gen_separate, DataFrameGraph

        df = pd.DataFrame([['c', 'C2', 'C4_d'], ['a', 'C2', 'C5_b'], ['a', 'C3', 'C6_e']],
                          columns=['C1', 'var', 'value'])

        g_df = DataFrameGraph(df)
        result, call_str, graph, res_graph = gen_separate.with_env(ignore_exceptions=False,
                                                                   replay={
                                                                       "separate_col": ["value"],
                                                                   }).call(df, g_df, [])

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result, check_names=False)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')), check_names=False)

    def test_mutate(self):
        from gauss.domains.pandas_lite.generators import gen_mutate, DataFrameGraph

        df = pd.DataFrame({'col1': {0: 'A1', 1: 'A1', 2: 'A2', 3: 'A2'},
                           'col2': {0: 'B1', 1: 'B2', 2: 'B1', 3: 'B2'},
                           'after': {0: 13.0, 1: 21.0, 2: 23.30, 3: 22.0},
                           'before': {0: 20.0, 1: 11.0, 2: 18.0, 3: 34.0}})

        g_df = DataFrameGraph(df)
        result, call_str, graph, res_graph = gen_mutate.with_env(ignore_exceptions=False,
                                                                 replay={
                                                                     "mutate_cols": [["after", "before"]],
                                                                     "mutate_op": ["div"]
                                                                 }).call(df, g_df, [])

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result, check_names=False)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')), check_names=False)
