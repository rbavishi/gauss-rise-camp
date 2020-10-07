import unittest

import pandas as pd

from gauss.domains.rlang.definition import RLangSynthesisDomain


class TestDataFrameGraph(unittest.TestCase):
    def test_df_graph(self):
        from gauss.domains.rlang.graphs import DataFrameGraph
        df = pd.DataFrame([['a', 'b', 'e'], ['c', 'd', 'f']], columns=['C1', 'C2', 'C3'])
        df_graph = DataFrameGraph(df)

        #  Check if all the nodes have been created.
        self.assertListEqual(list(df.columns), [c.value for c in df_graph.columns])
        self.assertListEqual(list(df.index), [c.value for c in df_graph.index])
        for row_df, row_df_graph in zip(df.values, df_graph.values):
            self.assertListEqual(list(row_df), [v.value for v in row_df_graph])


class TestDataGen(unittest.TestCase):
    def test_basic_gen(self):
        domain = RLangSynthesisDomain()
        for name in domain.get_available_components():
            domain.generate_witness_entry(name, seed=0)


class TestGenerators(unittest.TestCase):
    def test_gather(self):
        from gauss.domains.rlang.generators import gen_gather, DataFrameGraph, RInterpreter
        gather = RInterpreter.gather
        df = pd.DataFrame([['a', 'b', 'e'], ['c', 'd', 'f']], columns=['C1', 'C2', 'C3'])
        result, call_str, graph, res_graph = gen_gather.call(df, DataFrameGraph(df))

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')))

    def test_groupby(self):
        from gauss.domains.rlang.generators import gen_group_by_summarise, DataFrameGraph, RInterpreter
        group_by = RInterpreter.group_by
        summarise = RInterpreter.summarise

        df = pd.DataFrame([['A', 100], ['A', 200], ['B', 300]], columns=['C1', 'C2'])
        g_df = DataFrameGraph(df)
        result, call_str, graph, res_graph = gen_group_by_summarise.with_env(ignore_exceptions=False).call(df, g_df)

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df, result)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df')))

    def test_inner_join(self):
        from gauss.domains.rlang.generators import gen_inner_join, DataFrameGraph, RInterpreter
        inner_join = RInterpreter.inner_join

        df1 = pd.DataFrame([['a', 'b', 'c'], ['d', 'g', 'c'], ['f', 'b', 'h']], columns=['c1', 'c2', 'c3'])
        df2 = pd.DataFrame([['x', 'g', 'z'], ['w', 'b', 'u'], ['y', 'g', 'j']], columns=['c4', 'c2', 'c5'])

        result, call_str, graph, res_graph = gen_inner_join.call(df1, df2, DataFrameGraph(df1), DataFrameGraph(df2))

        #  Result should not be equal to the input
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df1, result)
        self.assertRaises(AssertionError, pd.testing.assert_frame_equal, df2, result)

        #  Call str should evaluate to the result
        pd.testing.assert_frame_equal(result, eval(call_str.format(inp1='df1', inp2='df2')))
