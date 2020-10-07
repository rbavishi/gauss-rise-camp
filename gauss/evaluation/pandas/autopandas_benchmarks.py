from io import StringIO
from typing import Any, Dict, Collection, Optional, List

import attr
import pandas as pd
import numpy as np

from gauss.domains.pandas_lite.definition import PandasLiteSynthesisDomain
from gauss.domains.pandas_lite.graphs import DataFrameGraph
from gauss.evaluation.benchmark import Benchmark
from gauss.graphs import Graph
from gauss.synthesis.problem import SynthesisProblem
from gauss.synthesis.skeleton import Skeleton
from gauss.utilities.debug import debug_iter
from gauss.utilities.logutils import logger


@attr.s
class AutoPandasBenchmark(Benchmark):
    replay_map: Dict[str, Any] = attr.ib()
    constants: List[Any] = attr.ib(factory=list)
    stats: Dict[str, Any] = attr.ib(factory=dict)

    _graph: Graph = attr.ib(init=False, default=None)
    _g_inputs: List[Graph] = attr.ib(init=False, default=None)

    def _convert_inp_to_graph(self, inp):
        if isinstance(inp, pd.DataFrame):
            return DataFrameGraph(inp)

        raise AssertionError(f"Input of type {type(inp)} not recognized")

    def init(self):
        domain = PandasLiteSynthesisDomain()
        replay = {k: iter(v) for k, v in self.replay_map.items()}
        graph = Graph()

        g_inputs = self._g_inputs = [self._convert_inp_to_graph(inp) for inp in self.inputs]
        int_to_val = {-idx: inp for idx, inp in enumerate(self.inputs, 1)}
        int_to_graph = {-idx: g_inp for idx, g_inp in enumerate(g_inputs, 1)}

        #  Run the generators to extract the programs and graphs for each component call.
        #  Merge the individual graphs into the master graph.
        call_strs: List[str] = []
        for idx, (component_name, arg_ints) in enumerate(self.skeleton, 1):
            c_inputs = [int_to_val[i] for i in arg_ints]
            g_c_inputs = [int_to_graph[i] for i in arg_ints]
            output, program, c_graph, output_graph = next(domain.enumerate(component_name, c_inputs, g_c_inputs,
                                                                           replay=replay))
            int_to_val[idx] = output
            int_to_graph[idx] = output_graph
            call_strs.append(program)
            graph.merge(c_graph)

        #  Check that the final output is equivalent to the original output specified in the benchmark.
        assert domain.check_equivalent(self.output, int_to_val[self.skeleton.length]), \
            f"Generated output inconsistent with specified output in Pandas benchmark {self.b_id}"

        #  Retrofit the value of the output entity to the original output
        cur_out_entity = next(ent for ent in graph.iter_entities() if ent.value is int_to_val[self.skeleton.length])
        cur_out_entity.value = self.output

        #  Perform transitive closure w.r.t the nodes corresponding to the intermediate outputs
        #  and take the induced subgraph containing all nodes except those
        if self.skeleton.length > 1:
            join_nodes = set.union(*(set(int_to_graph[i].iter_nodes()) for i in range(1, self.skeleton.length)))
            domain.perform_transitive_closure(graph, join_nodes=join_nodes)
            intent_graph = graph.induced_subgraph(keep_nodes=set(graph.iter_nodes()) - join_nodes)
        else:
            intent_graph = graph

        self._graph = intent_graph

        #  Also construct the string representation of the ground-truth program.
        program_list: List[str] = []
        for depth, (call_str, (component_name, arg_ints)) in enumerate(zip(call_strs, self.skeleton), 1):
            arg_strs = [f"inp{-i}" if i < 0 else f"v{i}" for i in arg_ints]
            call_str = call_str.format(**{f"inp{idx}": arg_str for idx, arg_str in enumerate(arg_strs, 1)})
            if depth == self.skeleton.length:
                program_list.append(call_str)
            else:
                program_list.append(f"v{depth} = {call_str}")

        self.program = "\n".join(program_list)

    def construct_synthesis_problem(self) -> SynthesisProblem:
        if self._graph is None:
            self.init()

        return SynthesisProblem(inputs=self.inputs,
                                output=self.output,
                                constants=self.constants,
                                graph=self._graph,
                                graph_inputs=self._g_inputs)


class AutoPandasBenchmarks:
    _benchmarks_map: Dict[str, AutoPandasBenchmark]

    def __init__(self,
                 b_ids: Optional[Collection[str]] = None,
                 min_depth: Optional[int] = None,
                 max_depth: Optional[int] = None):

        if b_ids is not None:
            b_ids = set(b_ids)

        logger.debug("Collecting Pandas benchmarks")
        domain = PandasLiteSynthesisDomain()
        known_components = set(domain.get_available_components())

        benchmarks: Dict[str, AutoPandasBenchmark] = {}
        logger.debug(f"Found {sum(1 for k, v in self.__class__.__dict__.items() if k.startswith('test_'))} "
                     f"total benchmarks.")

        for benchmark in (v(self) for k, v in self.__class__.__dict__.items() if k.startswith("test_")):
            used_components = set(benchmark.skeleton.components)

            if not used_components.issubset(known_components):
                continue

            if min_depth is not None and min_depth > benchmark.skeleton.length:
                continue

            if max_depth is not None and max_depth < benchmark.skeleton.length:
                continue

            if b_ids is None or benchmark.b_id in b_ids:
                benchmarks[benchmark.b_id] = benchmark

        logger.debug(f"Collected {len(benchmarks)} Pandas benchmarks after filtering.")
        self._benchmarks_map = {}

        logger.debug("Building intent graphs")
        for b_id, benchmark in debug_iter(benchmarks.items(), desc='Building Graphs', total=len(benchmarks)):
            #  Initialize the graph and the ground-truth program.
            benchmark.init()
            self._benchmarks_map[b_id] = benchmark

    def __getitem__(self, item):
        return self._benchmarks_map[item]

    def __iter__(self):
        yield from self._benchmarks_map.values()

    def test_SO_13659881_depth2(self):
        b_id = "b_1"
        inputs = [pd.DataFrame(
            columns=['ip', 'useragent'],
            index=[0, 1, 2, 3],
            data=[['192.168.0.1', 'a'], ['192.168.0.1', 'a'], ['192.168.0.1', 'b'], ['192.168.0.2', 'b']]
        )]

        output = inputs[0].groupby(['ip', 'useragent'], as_index=False).size().reset_index(name='size')
        intermediates = [output]
        skeleton = Skeleton([('pd.groupby_agg', [(-1)])])

        replay_map = {
            'groupby_agg_by_cols': [['ip', 'useragent']],
            'groupby_agg_op': ['size'],
            'groupby_agg_size_new_col': ['size']
        }

        stats = {'autopandas_time': 1.38}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map, stats=stats)

    def test_SO_13647222_depth1(self):
        b_id = "b_2"
        inputs = [pd.DataFrame({'series': {0: 'A', 1: 'B', 2: 'C', 3: 'A', 4: 'B', 5: 'C', 6: 'A', 7: 'B',
                                           8: 'C', 9: 'A', 10: 'B', 11: 'C', 12: 'A', 13: 'B', 14: 'C'},
                                'step': {0: '100', 1: '100', 2: '100', 3: '101', 4: '101', 5: '101', 6: '102', 7: '102',
                                         8: '102', 9: '103', 10: '103', 11: '103', 12: '104', 13: '104', 14: '104'},
                                'value': {0: '1000', 1: '1001', 2: '1002', 3: '1003', 4: '1004', 5: '1005', 6: '1006',
                                          7: '1007',
                                          8: '1008', 9: '1009', 10: '1010', 11: '1011', 12: '1012', 13: '1013',
                                          14: '1014'}})]

        output = inputs[0].pivot(columns='series', values='value', index='step').reset_index()
        intermediates = [output]
        skeleton = Skeleton([('pd.pivot_table', [(-1)])])

        replay_map = {
            'pivot_columns': ['series'],
            'pivot_values': ['value'],
            'pivot_index': [['step']],
        }

        stats = {'autopandas_time': 3.32}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map, stats=stats)

    def test_SO_12860421_depth1(self):
        b_id = "b_3"
        inputs = [pd.DataFrame(columns=['X', 'Y', 'Z'],
                               data=[['X1', 'Y2', 'Z3'], ['X1', 'Y1', 'Z1'], ['X1', 'Y1', 'Z1'], ['X1', 'Y1', 'Z2']]
                               )]

        output = inputs[0].pivot_table(values='X', index='Y', columns='Z', aggfunc=pd.Series.nunique).reset_index()
        intermediates = [output]
        skeleton = Skeleton([('pd.groupby_agg', [(-1)]), ('pd.pivot_table', [1])])

        replay_map = {
            'groupby_agg_by_cols': [['Z', 'Y']],
            'groupby_agg_op': ['AGG_nunique'],
            'pivot_columns': ['Z'],
            'pivot_values': ['X'],
            'pivot_index': [['Y']],
        }

        stats = {'autopandas_time': 3.3}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map, stats=stats)

    def test_SO_23321300_depth3(self):
        b_id = "b_4"
        inputs = [pd.DataFrame({"a": [1, 1, 1, 1, 1, 1, 1, 1, 1], "b": [1, 1, 1, 1, 1, 2, 2, 2, 3],
                                "d": [0, 200, 300, 0, 600, 0, 100, 200, 0]})]
        output = inputs[0].query('d > 0').groupby(['a', 'b'], as_index=False).mean()
        intermediates = [output]
        skeleton = Skeleton([('pd.filtering_expr', [(-1)]), ('pd.groupby_agg', [1])])

        replay_map = {
            'filtering_expr_expression': ['d > 0'],
            'groupby_agg_by_cols': [['a', 'b']],
            'groupby_agg_op': ['mean'],
        }

        stats = {'autopandas_time': 0}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   constants=['d > 0'], stats=stats)

    def test_SO_39656670_depth3(self):
        b_id = "b_5"
        inputs = [pd.DataFrame({
            "Player": ["Abdoun", "Abe", "Abidal", "Abreu"],
            "Team": ["Algeria", "Japan", "France", "Uruguay"],
            "Shots": [0, 3, 0, 5],
            "Passes": [6, 101, 91, 15],
            "Tackles": [0, 14, 6, 0]
        })]

        output = inputs[0].melt(value_vars=["Passes", "Tackles"],
                                var_name="Var",
                                value_name="Mean").groupby("Var", as_index=False).mean()

        intermediates = [output]
        skeleton = Skeleton([('pd.melt', [(-1)]), ('pd.groupby_agg', [1])])

        replay_map = {
            'melt_id_vars': [[]],
            'melt_value_vars': [['Passes', 'Tackles']],
            'melt_var_name': ['Var'],
            'melt_value_name': ['Mean'],
            'groupby_agg_by_cols': [['Var']],
            'groupby_agg_op': ['mean'],
        }

        stats = {'autopandas_time': 0}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_21982987_depth3(self):
        b_id = "b_6"
        inputs = [pd.DataFrame({"Name": ["Aira", "Aira", "Ben", "Ben", "Cat", "Cat"], "Month": [1, 2, 1, 2, 1, 2],
                                "Rate1": [12, 18, 53, 22, 22, 27], "Rate2": [23, 73, 19, 87, 87, 43]})]

        output = pd.DataFrame({'Name': {0: 'Aira', 1: 'Ben', 2: 'Cat'}, 'Rate1': {0: 15.0, 1: 37.5, 2: 24.5},
                               'Rate2': {0: 48.0, 1: 53.0, 2: 65.0}})

        intermediates = [output]
        skeleton = Skeleton([('pd.groupby_agg', [(-1)]), ('pd.drop_columns', [1])])

        replay_map = {
            'groupby_agg_by_cols': [['Name']],
            'groupby_agg_op': ['mean'],
            'drop_columns_cols': [['Month']],
        }

        stats = {'autopandas_time': 30.80}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_53762029_depth3(self):
        b_id = "b_7"
        data = """
doc_created_month   doc_created_year    speciality      doc_id
8                   2016                Acupuncturist   1           
2                   2017                Acupuncturist   1           
4                   2017                Acupuncturist   1           
4                   2017                Allergist       1           
5                   2018                Allergist       1           
10                  2018                Allergist       2   
"""

        df = pd.read_csv(StringIO(data), sep=r'\s+')
        inputs = [df]
        output = df.assign(doc_id_count=df.groupby(['speciality'],
                                                   as_index=False)['doc_id'].transform('cumsum')
                           ).drop(columns=['doc_id'])

        intermediates = [output]
        skeleton = Skeleton([('pd.groupby_transform', [(-1)]), ('pd.drop_columns', [1])])

        replay_map = {
            'groupby_transform_by_cols': [['speciality']],
            'groupby_transform_op': ['cumsum'],
            'groupby_transform_op_col': ['doc_id'],
            'groupby_transform_new_col': ['doc_id_count'],
            'drop_columns_cols': [['doc_id']],
        }

        stats = {'autopandas_time': 1.90}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_14023037_depth3_part1(self):
        b_id = "b_8_1"
        inputs = [pd.DataFrame(
            {'id': [1, 2, 3, 4, 5, 6],
             'col1': ['A1', 'A1', 'A1', 'A1', 'A2', 'A2'],
             'col2': ['B1', 'B1', 'B2', 'B2', 'B1', 'B2'],
             'col3': ['before', 'after', 'before', 'after', 'before', 'after'],
             'value': [20, 13, 11, 21, 18, 22]},
            columns=['id', 'col1', 'col2', 'col3', 'value'])]

        output = inputs[0].pivot_table(values='value',
                                       index=['col1', 'col2'],
                                       columns=['col3']).reset_index()
        intermediates = [output]
        skeleton = Skeleton([('pd.pivot_table', [(-1)])])

        replay_map = {
            'pivot_columns': ['col3'],
            'pivot_values': ['value'],
            'pivot_index': [['col1', 'col2']],
        }

        stats = {'autopandas_time': 0}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_14023037_depth3_part2(self):
        b_id = "b_8_2"
        df = pd.DataFrame(
            {'id': [1, 2, 3, 4, 5, 6],
             'col1': ['A1', 'A1', 'A1', 'A1', 'A2', 'A2'],
             'col2': ['B1', 'B1', 'B2', 'B2', 'B1', 'B2'],
             'col3': ['before', 'after', 'before', 'after', 'before', 'after'],
             'value': [20, 13, 11, 21, 18, 22]},
            columns=['id', 'col1', 'col2', 'col3', 'value'])

        inputs = [df.pivot_table(values='value',
                                 index=['col1', 'col2'],
                                 columns=['col3']).reset_index()]
        output = inputs[0].fillna(method='backfill').dropna()
        intermediates = [output]
        skeleton = Skeleton([('pd.fillna', [(-1)]), ('pd.dropna', [1])])

        replay_map = {
            'fillna_mode': ['method'],
            'fillna_axis': ['index'],
            'fillna_method': ['backfill'],
            'dropna_how': ['any'],
            'dropna_inspect_cols': [['before']],
        }

        stats = {'autopandas_time': 0}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_13576164_depth3_simplified(self):
        b_id = "b_9"

        inputs = [pd.DataFrame(columns=['col1', 'to_merge_on'],
                               index=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']],
                                                               names=['id1', 'id2']),
                               data=[[1, 2], [3, 4], [1, 2], [3, 4]]).reset_index(),
                  pd.DataFrame(columns=['col2', 'to_merge_on'],
                               index=[0, 1, 2],
                               data=[[1, 1], [2, 3], [3, 4]])]

        output = inputs[0].merge(inputs[1], how='inner')
        intermediates = [output]
        skeleton = Skeleton([('pd.merge', [(-1), (-2)])])

        replay_map = {}

        stats = {'autopandas_time': 339.25}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_12065885_depth3(self):
        b_id = "b_10"

        inputs = [pd.DataFrame({'RPT_Date': {0: '1980-01-01', 1: '1980-01-02', 2: '1980-01-03', 3: '1980-01-04',
                                             4: '1980-01-05', 5: '1980-01-06', 6: '1980-01-07', 7: '1980-01-08',
                                             8: '1980-01-09', 9: '1980-01-10'},
                                'STK_ID': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
                                'STK_Name': {0: 'Arthur', 1: 'Beate', 2: 'Cecil', 3: 'Dana', 4: 'Eric', 5: 'Fidel',
                                             6: 'George', 7: 'Hans', 8: 'Ingrid', 9: 'Jones'},
                                'sales': {0: 0, 1: 4, 2: 2, 3: 8, 4: 4, 5: 5, 6: 4, 7: 7, 8: 7, 9: 4}})]
        constants = [[4, 2, 6]]

        output = inputs[0][inputs[0].STK_ID.isin(constants[0])]
        intermediates = [output]
        skeleton = Skeleton([('pd.filtering_contains', [(-1)])])

        replay_map = {
            'filtering_contains_filter_col': ['STK_ID'],
            'filtering_contains_collection': [[4, 2, 6]],
        }

        stats = {'autopandas_time': 0.9}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   constants=constants, stats=stats)

    def test_SO_10982266_depth3(self):
        b_id = "b_11"

        inputs = [pd.DataFrame(
            [['08:01:08', 'C', 'PXA', 20100101, 4000, 'A', 57.8, 60],
             ['08:01:11', 'C', 'PXA', 20100101, 4000, 'A', 58.4, 60],
             ['08:01:12', 'C', 'PXA', 20100101, 4000, 'A', 58.0, 60],
             ['08:01:16', 'C', 'PXA', 20100101, 4000, 'A', 58.4, 60],
             ['08:01:16', 'C', 'PXA', 20100101, 4000, 'A', 58.0, 60],
             ['08:01:21', 'C', 'PXA', 20100101, 4000, 'A', 58.4, 60],
             ['08:01:21', 'C', 'PXA', 20100101, 4000, 'A', 58.0, 60]],
            columns=['time', 'contract', 'ticker', 'expiry', 'strike', 'quote', 'price', 'volume'],
            index=[0, 1, 2, 3, 4, 5, 6]
        )]

        output = pd.DataFrame(
            [['08:01:08', 57.8, 60], ['08:01:11', 58.4, 60], ['08:01:12', 58.0, 60], ['08:01:16', 58.2, 60],
             ['08:01:21', 58.2, 60]],
            columns=['time', 'price', 'volume'],
            index=[0, 1, 2, 3, 4])

        intermediates = [output]
        skeleton = Skeleton([('pd.groupby_agg', [(-1)]), ('pd.drop_columns', [1])])

        replay_map = {
            'groupby_agg_by_cols': [['time', 'volume']],
            'groupby_agg_op': ['mean'],
            'drop_columns_cols': [['expiry', 'strike']],
        }

        stats = {'autopandas_time': 0}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map, stats=stats)

    def test_SO_34365578_depth2(self):
        b_id = "b_12"

        inputs = [pd.DataFrame({'Group': {0: 'A', 1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'B'},
                                'Id': {0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16},
                                'Var1': {0: 'good', 1: 'good', 2: 'bad', 3: 'good', 4: 'good', 5: 'bad'},
                                'Var2': {0: 20, 1: 26, 2: 29, 3: 23, 4: 23, 5: 28}})]

        constants = ["`Group` == 'A'"]
        output = inputs[0].query('Group == "A"').pivot_table(index='Group', columns='Var1', values='Var2',
                                                             aggfunc='sum').reset_index()

        intermediates = [output]
        skeleton = Skeleton([('pd.filtering_expr', [(-1)]), ('pd.groupby_agg', [1]), ('pd.pivot_table', [2])])

        replay_map = {
            'filtering_expr_expression': ['`Group` == "A"'],
            'groupby_agg_by_cols': [['Group', 'Var1']],
            'groupby_agg_op': ['sum'],
            'pivot_columns': ['Var1'],
            'pivot_values': ['Var2'],
            'pivot_index': [['Group']],
        }

        stats = {'autopandas_time': 0}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   constants=constants, stats=stats)

    def test_SO_13807758_depth2(self):
        b_id = "b_13"

        df1 = pd.DataFrame([[10], [11], [12], [14], [16], [18]], columns=['A'])
        df1[::3] = np.nan
        inputs = [df1]

        output = inputs[0].dropna()

        intermediates = [output]
        skeleton = Skeleton([('pd.dropna', [(-1)])])

        replay_map = {
            'dropna_how': ['any'],
            'dropna_inspect_cols': [['A']],
        }

        stats = {'autopandas_time': 7.21}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_49987108_depth2(self):
        b_id = "b_14"

        inputs = [pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                'COL': [23, np.nan, np.nan, np.nan, np.nan, 21, np.nan, np.nan, np.nan, 25, np.nan,
                                        np.nan]}).set_index('ID')]

        output = inputs[0].fillna(method='pad')

        intermediates = [output]
        skeleton = Skeleton([('pd.fillna', [(-1)])])

        replay_map = {
            'fillna_mode': ['method'],
            'fillna_axis': ['index'],
            'fillna_method': ['pad'],
        }

        stats = {'autopandas_time': 0}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_49567723_depth2(self):
        b_id = "b_15"

        inputs = [pd.DataFrame({'id': {0: 255, 1: 91, 2: 347, 3: 30, 4: 68, 5: 159, 6: 32, 7: 110, 8: 225, 9: 257},
                                'valueA': {0: 1141, 1: 1130, 2: 830, 3: 757, 4: 736, 5: 715, 6: 713, 7: 683, 8: 638,
                                           9: 616}}),
                  pd.DataFrame({'id': {0: 255, 1: 91, 2: 5247, 3: 347, 4: 30, 5: 68,
                                       6: 159, 7: 32, 8: 110, 9: 225, 10: 257,
                                       11: 917, 12: 211, 13: 25},
                                'valueB': {0: 1231, 1: 1170, 2: 954, 3: 870, 4: 757,
                                           5: 736, 6: 734, 7: 713, 8: 683, 9: 644,
                                           10: 616, 11: 585, 12: 575, 13: 530}})]

        constants = ['`valueA` != `valueB`']
        output = inputs[0].merge(inputs[1]).query('`valueA` != `valueB`')

        intermediates = [output]
        skeleton = Skeleton([('pd.merge', [(-1), (-2)]), ('pd.filtering_expr', [1])])

        replay_map = {
            'filtering_expr_expression': [constants[0]]
        }
        stats = {'autopandas_time': 753.10}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map, constants=constants,
                                   stats=stats)

    def test_SO_11418192_depth2(self):
        b_id = "b_16"

        inputs = [pd.DataFrame(data=[[5, 7], [6, 8], [-1, 9], [-2, 10]], columns=['a', 'b'])]

        constants = ['`a` > 1']

        output = inputs[0].query("`a` > 1")
        intermediates = [output]

        skeleton = Skeleton([('pd.filtering_expr', [-1])])

        replay_map = {
            'filtering_expr_expression': [constants[0]]
        }
        stats = {'autopandas_time': 0.71}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map, constants=constants,
                                   stats=stats)

    def test_SO_13793321_depth1(self):
        b_id = "b_17"

        inputs = [pd.DataFrame([[11, 12, 13]], columns=[10, 1, 2]),
                  pd.DataFrame([[11, 37, 38], [34, 19, 39]], columns=[10, 3, 4])]

        output = inputs[0].merge(inputs[1], on=10)
        intermediates = [output]

        skeleton = Skeleton([('pd.merge', [(-1), (-2)])])

        replay_map = {
        }

        stats = {'autopandas_time': 4.16}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_13261175_depth1_simplified(self):
        b_id = "b_18"

        inputs = [pd.DataFrame({'name': ['A', 'B', 'A', 'B'], 'type': [11, 11, 12, 12],
                                'date': ['2012-01-01', '2012-01-01', '2012-02-01', '2012-02-01'],
                                'value': [4, 5, 6, 7]})]

        output = inputs[0].pivot_table(values='value', index='name', columns='date').reset_index()
        intermediates = [output]

        skeleton = Skeleton([('pd.pivot_table', [(-1)])])

        replay_map = {
            'pivot_columns': ['date'],
            'pivot_values': ['value'],
            'pivot_index': [['name']],
        }

        stats = {'autopandas_time': 300.20}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   stats=stats)

    def test_SO_18172851_depth1(self):
        b_id = "b_19"

        inputs = [pd.DataFrame({'daysago': {'2007-03-31': 62, '2007-03-10': 83, '2007-02-10': 111, '2007-01-13': 139,
                                            '2006-12-23': 160, '2006-11-09': 204, '2006-10-22': 222, '2006-09-29': 245,
                                            '2006-09-16': 258, '2006-08-30': 275, '2006-02-11': 475, '2006-01-13': 504,
                                            '2006-01-02': 515, '2005-12-06': 542, '2005-11-29': 549, '2005-11-22': 556,
                                            '2005-11-01': 577, '2005-10-20': 589, '2005-09-27': 612, '2005-09-07': 632,
                                            '2005-06-12': 719, '2005-05-29': 733, '2005-05-02': 760, '2005-04-02': 790,
                                            '2005-03-13': 810, '2004-11-09': 934},
                                'line_race': {'2007-03-31': 111, '2007-03-10': 211, '2007-02-10': 29, '2007-01-13': 110,
                                              '2006-12-23': 210, '2006-11-09': 39, '2006-10-22': 28, '2006-09-29': 49,
                                              '2006-09-16': 311, '2006-08-30': 48, '2006-02-11': 45, '2006-01-13': 0,
                                              '2006-01-02': 0, '2005-12-06': 0, '2005-11-29': 0, '2005-11-22': 0,
                                              '2005-11-01': 0, '2005-10-20': 0, '2005-09-27': 0, '2005-09-07': 0,
                                              '2005-06-12': 0, '2005-05-29': 0, '2005-05-02': 0, '2005-04-02': 0,
                                              '2005-03-13': 0, '2004-11-09': 0},
                                'rating': {'2007-03-31': 2, '2007-03-10': 3, '2007-02-10': 4, '2007-01-13': 5,
                                           '2006-12-23': 6, '2006-11-09': 7, '2006-10-22': 8, '2006-09-29': 9,
                                           '2006-09-16': 10, '2006-08-30': 11, '2006-02-11': 12, '2006-01-13': 13,
                                           '2006-01-02': 14, '2005-12-06': 15, '2005-11-29': 16, '2005-11-22': 17,
                                           '2005-11-01': 18, '2005-10-20': 19, '2005-09-27': 20, '2005-09-07': 21,
                                           '2005-06-12': 22, '2005-05-29': 23, '2005-05-02': 24, '2005-04-02': 25,
                                           '2005-03-13': 26, '2004-11-09': 27},
                                'rw': {'2007-03-31': 0.99999, '2007-03-10': 0.97, '2007-02-10': 0.9,
                                       '2007-01-13': 0.8806780000000001, '2006-12-23': 0.793033, '2006-11-09': 0.636655,
                                       '2006-10-22': 0.581946, '2006-09-29': 0.518825,
                                       '2006-09-16': 0.48622600000000005, '2006-08-30': 0.446667,
                                       '2006-02-11': 0.16459100000000002, '2006-01-13': 0.14240899999999998,
                                       '2006-01-02': 0.1348, '2005-12-06': 0.11780299999999999,
                                       '2005-11-29': 0.113758, '2005-11-22': 0.10985199999999999,
                                       '2005-11-01': 0.098919, '2005-10-20': 0.093168, '2005-09-27': 0.083063,
                                       '2005-09-07': 0.075171, '2005-06-12': 0.04869, '2005-05-29': 0.045404,
                                       '2005-05-02': 0.039679, '2005-04-02': 0.03416,
                                       '2005-03-13': 0.030914999999999998, '2004-11-09': 0.016647},
                                'wrating': {'2007-03-31': 1.99998, '2007-03-10': 2.91, '2007-02-10': 3.6,
                                            '2007-01-13': 4.40339, '2006-12-23': 4.758198, '2006-11-09': 4.456585,
                                            '2006-10-22': 4.655568, '2006-09-29': 4.6694249999999995,
                                            '2006-09-16': 4.862260000000001, '2006-08-30': 4.913336999999999,
                                            '2006-02-11': 1.975092, '2006-01-13': 1.8513169999999997,
                                            '2006-01-02': 1.8872, '2005-12-06': 1.767045, '2005-11-29': 1.820128,
                                            '2005-11-22': 1.867484, '2005-11-01': 1.780542, '2005-10-20': 1.770192,
                                            '2005-09-27': 1.66126, '2005-09-07': 1.578591, '2005-06-12': 1.07118,
                                            '2005-05-29': 1.044292, '2005-05-02': 0.952296,
                                            '2005-04-02': 0.8540000000000001, '2005-03-13': 0.80379,
                                            '2004-11-09': 0.44946899999999995}})]

        constants = ["`line_race` != 0"]
        output = inputs[0].query("`line_race` != 0")
        intermediates = [output]

        skeleton = Skeleton([('pd.filtering_expr', [-1])])

        replay_map = {
            'filtering_expr_expression': [constants[0]]
        }
        stats = {'autopandas_time': 0}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   constants=constants, stats=stats)

    def test_SO_11941492_depth1(self):
        b_id = "b_20"

        df = pd.DataFrame({'group1': ['a', 'a', 'a', 'b', 'b', 'b'],
                           'group2': ['c', 'c', 'd', 'd', 'd', 'e'],
                           'value1': [1.1, 2, 3, 4, 5, 6],
                           'value2': [7.1, 8, 9, 10, 11, 12]
                           })

        constants = ['`group1` == "a"']
        inputs = [df]
        output = df.set_index(['group1', 'group2']).xs('a', level=0).reset_index()
        intermediates = [output]

        skeleton = Skeleton([('pd.filtering_expr', [-1]), ('pd.drop_columns', [1])])
        replay_map = {
            'filtering_expr_expression': [constants[0]],
            "drop_columns_cols": [['group1']]
        }
        stats = {'autopandas_time': 12.55}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map,
                                   constants=constants, stats=stats)

    def test_SO_49583055_depth1(self):
        b_id = "b_21"

        inputs = [pd.DataFrame({'value': {pd.Timestamp('2014-05-21 09:30:00'): 0.0,
                                          pd.Timestamp('2014-05-21 10:00:00'): 10.0,
                                          pd.Timestamp('2014-05-21 10:30:00'): 3.0,
                                          pd.Timestamp('2017-07-10 22:30:00'): 18.3,
                                          pd.Timestamp('2017-07-10 23:00:00'): 7.6,
                                          pd.Timestamp('2017-07-10 23:30:00'): 2.0}}),
                  pd.DataFrame({'value': {pd.Timestamp('2014-05-21 09:00:00'): 1.0,
                                          pd.Timestamp('2014-05-21 10:00:00'): 13.0,
                                          pd.Timestamp('2017-07-10 21:00:00'): 1.6,
                                          pd.Timestamp('2017-07-10 22:00:00'): 32.1,
                                          pd.Timestamp('2017-07-10 23:00:00'): 7.7}})
                  ]

        output = inputs[0].combine_first(inputs[1])
        intermediates = [output]

        skeleton = Skeleton([('pd.combine_first', [(-1), (-2)])])
        replay_map = {
        }
        stats = {'autopandas_time': 0}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map, stats=stats)

    def test_SO_49572546_depth1(self):
        b_id = "b_22"

        inputs = [pd.DataFrame(
            {'C1': {1: 100, 2: 102, 3: 103, 4: 104, 5: 105, 6: 106, 7: 107},
             'C2': {1: 201, 2: 202, 3: 203, 4: 204, 5: 205, 6: 206, 7: 207},
             'C3': {1: 301, 2: 302, 3: 303, 4: 304, 5: 305, 6: 306, 7: 307}}),
            pd.DataFrame(
                {'C1': {2: '1002', 3: 'v1', 4: 'v4', 7: '1007'}, 'C2': {2: '2002', 3: 'v2', 4: 'v5', 7: '2007'},
                 'C3': {2: '3002', 3: 'v3', 4: 'v6', 7: '3007'}})]

        output = inputs[1].combine_first(inputs[0])
        intermediates = [output]

        skeleton = Skeleton([('pd.combine_first', [(-2), (-1)])])
        replay_map = {
        }
        stats = {'autopandas_time': 1.1}

        return AutoPandasBenchmark(b_id, inputs, intermediates, output, "", skeleton, replay_map, stats=stats)
