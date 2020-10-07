from typing import Any, Dict, Collection, Optional, List

import attr
import pandas as pd

from gauss.domains.rlang.definition import RLangSynthesisDomain, RLangEnumerationItem
from gauss.domains.rlang.graphs import DataFrameGraph
from gauss.domains.rlang.interpreter import RInterpreter as rf
from gauss.evaluation.benchmark import Benchmark
from gauss.graphs import Graph
from gauss.synthesis.problem import SynthesisProblem
from gauss.synthesis.skeleton import Skeleton
from gauss.utilities.debug import debug_iter
from gauss.utilities.logutils import logger


@attr.s
class MorpheusBenchmark(Benchmark):
    morpheus_statistics: Dict[str, Any] = attr.ib()
    replay_map: Dict[str, Any] = attr.ib()
    constants: List[Any] = attr.ib(factory=list)

    _graph: Graph = attr.ib(init=False, default=None)
    _g_inputs: List[Graph] = attr.ib(init=False, default=None)

    def init(self):
        domain = RLangSynthesisDomain()
        replay = {k: iter(v) for k, v in self.replay_map.items()}
        graph = Graph()

        g_inputs = self._g_inputs = [DataFrameGraph(inp) for inp in self.inputs]
        int_to_val = {-idx: inp for idx, inp in enumerate(self.inputs, 1)}
        int_to_graph = {-idx: g_inp for idx, g_inp in enumerate(g_inputs, 1)}

        #  Run the generators to extract the programs and graphs for each component call.
        #  Merge the individual graphs into the master graph.
        call_strs: List[str] = []
        for idx, (component_name, arg_ints) in enumerate(self.skeleton, 1):
            c_inputs = [int_to_val[i] for i in arg_ints]
            g_c_inputs = [int_to_graph[i] for i in arg_ints]
            result: RLangEnumerationItem = next(domain.enumerate(component_name, c_inputs, g_c_inputs, replay=replay))
            output = result.output
            call_str = result.call_str
            c_graph = result.graph
            output_graph = result.o_graph
            int_to_val[idx] = output
            int_to_graph[idx] = output_graph
            call_strs.append(call_str)
            graph.merge(c_graph)

        #  Check that the final output is equivalent to the original output specified in the benchmark.
        print(self.output)
        print(int_to_val[self.skeleton.length])
        assert domain.check_equivalent(self.output, int_to_val[self.skeleton.length]), \
            f"Generated output inconsistent with specified output in Morpheus benchmark {self.b_id}"

        #  Retrofit the value of the output entity to the original output
        cur_out_entity = next(ent for ent in graph.iter_entities() if ent.value is int_to_val[self.skeleton.length])
        cur_out_entity.value = self.output

        #  Perform transitive closure w.r.t the nodes corresponding to the intermediate outputs
        #  and take the induced subgraph containing all nodes except those
        join_nodes = set.union(*(set(int_to_graph[i].iter_nodes()) for i in range(1, self.skeleton.length)))
        domain.perform_transitive_closure(graph, join_nodes=join_nodes)
        intent_graph = graph.induced_subgraph(keep_nodes=set(graph.iter_nodes())-join_nodes)
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
                                graph=self._graph,
                                graph_inputs=self._g_inputs)


class MorpheusBenchmarks:
    _benchmarks_map: Dict[str, MorpheusBenchmark]

    def __init__(self,
                 b_ids: Optional[Collection[str]] = None,
                 min_depth: Optional[int] = None,
                 max_depth: Optional[int] = None):

        if b_ids is not None:
            b_ids = set(b_ids)

        logger.debug("Collecting Morpheus benchmarks")
        domain = RLangSynthesisDomain()
        benchmarks: Dict[str, MorpheusBenchmark] = {}
        known_components = set(domain.get_available_components())
        logger.debug(f"Found {sum(1 for k, v in self.__class__.__dict__.items() if k.startswith('test_'))} "
                     f"total benchmarks.")

        for benchmark in (v(self) for k, v in self.__class__.__dict__.items() if k.startswith("test_")):
            used_components = set(benchmark.skeleton.components)

            #  rf.summarise is fused with rf.group_by
            if not (used_components - {"rf.summarise"}).issubset(known_components):
                continue

            if min_depth is not None and min_depth > benchmark.skeleton.length:
                continue

            if max_depth is not None and max_depth < benchmark.skeleton.length:
                continue

            if b_ids is None or benchmark.b_id in b_ids:
                benchmarks[benchmark.b_id] = benchmark

        logger.debug(f"Collected {len(benchmarks)} Morpheus benchmarks after filtering.")
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

    def test_1(self):
        b_id = "b_1"
        inputs = [pd.DataFrame({
            'round': {
                1: 'round1', 2: 'round2', 3: 'round1', 4: 'round2', }, 'var1': {
                1: 22, 2: 11, 3: 22, 4: 11, }, 'var2': {
                1: 33, 2: 44, 3: 33, 4: 44, }, 'nam': {
                1: 'foo', 2: 'foo', 3: 'bar', 4: 'bar', }, 'val': {
                1: 0.16912200977094502, 2: 0.18570826458744696, 3: 0.124105813913047, 4: 0.0325823465827852, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH2', value_name='MORPH1', value_vars=None, id_vars=['round', 'nam'])
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH159', cols=['MORPH2', 'round'])
        morpheus = rf.spread(tbl_1, columns='MORPH159', values='MORPH1')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 1, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 158, '#partial programs without partial evaluation': 303,
            'Total time': 1.35, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH2'],
            'gather_value_name': ['MORPH1'], 'gather_id_vars': [['round', 'nam']],
            'gather_value_vars': [['var1', 'var2', 'val']], 'gather_df': ['inputs[0]'],
            'unite_new_col_name': ['MORPH159'], 'unite_cols': [['MORPH2', 'round']], 'spread_columns': ['MORPH159'],
            'spread_values': ['MORPH1'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_2(self):
        b_id = "b_2"
        inputs = [pd.DataFrame({
            'month': {
                1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, }, 'student': {
                1: 'Amy', 2: 'Amy', 3: 'Amy', 4: 'Bob', 5: 'Bob', 6: 'Bob', }, 'A': {
                1: 9, 2: 7, 3: 6, 4: 8, 5: 6, 6: 9, }, 'B': {
                1: 6, 2: 7, 3: 8, 4: 5, 5: 6, 6: 7, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH102', value_name='MORPH101', value_vars=['A', 'B'], id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH110', cols=['student', 'MORPH102'])
        morpheus = rf.spread(tbl_1, columns='MORPH110', values='MORPH101')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 26, '#partial programs without partial evaluation': 131,
            'Total time': 0.29, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH102'],
            'gather_value_name': ['MORPH101'], 'gather_id_vars': [['month', 'student']],
            'gather_value_vars': [['A', 'B']], 'gather_df': ['inputs[0]'], 'unite_new_col_name': ['MORPH110'],
            'unite_cols': [['student', 'MORPH102']], 'spread_columns': ['MORPH110'], 'spread_values': ['MORPH101'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_3(self):
        b_id = "b_3"
        inputs = [pd.DataFrame({
            'Person': {
                1: 'greg', 2: 'greg', 3: 'sally', 4: 'sally', }, 'Time': {
                1: 'Pre', 2: 'Post', 3: 'Pre', 4: 'Post', }, 'Score1': {
                1: 88, 2: 78, 3: 76, 4: 78, }, 'Score2': {
                1: 84, 2: 82, 3: 72, 4: 79, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH72', value_name='MORPH71', value_vars=['Score1', 'Score2'],
                          id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH80', cols=['Time', 'MORPH72'])
        morpheus = rf.spread(tbl_1, columns='MORPH80', values='MORPH71')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 22, '#partial programs without partial evaluation': 93,
            'Total time': 0.22, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH72'],
            'gather_value_name': ['MORPH71'], 'gather_id_vars': [['Person', 'Time']],
            'gather_value_vars': [['Score1', 'Score2']], 'gather_df': ['inputs[0]'], 'unite_new_col_name': ['MORPH80'],
            'unite_cols': [['Time', 'MORPH72']], 'spread_columns': ['MORPH80'], 'spread_values': ['MORPH71'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_4(self):
        b_id = "b_4"
        inputs = [pd.DataFrame({
            'id': {
                1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, }, 'Year': {
                1: 2007, 2: 2008, 3: 2009, 4: 2007, 5: 2008, 6: 2009, }, 'A': {
                1: 5, 2: 2, 3: 3, 4: 7, 5: 5, 6: 6, }, 'B': {
                1: 10, 2: 0, 3: 50, 4: 13, 5: 17, 6: 17, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH132', value_name='MORPH131', value_vars=['A', 'B'], id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH139', cols=['MORPH132', 'Year'])
        morpheus = rf.spread(tbl_1, columns='MORPH139', values='MORPH131')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 20, '#partial programs without partial evaluation': 150,
            'Total time': 0.22, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH132'],
            'gather_value_name': ['MORPH131'], 'gather_id_vars': [['id', 'Year']], 'gather_value_vars': [['A', 'B']],
            'gather_df': ['inputs[0]'], 'unite_new_col_name': ['MORPH139'], 'unite_cols': [['MORPH132', 'Year']],
            'spread_columns': ['MORPH139'], 'spread_values': ['MORPH131'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_5(self):
        b_id = "b_5"
        inputs = [pd.DataFrame({
            'ID': {
                1: 1, 2: 2, }, 'T': {
                1: 24.3, 2: 23.4, }, 'P.1': {
                1: 10.2, 2: 10.4, }, 'P.2': {
                1: 5.5, 2: 5.7, }, 'Q.1': {
                1: 4.5, 2: 3.2, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH394', value_name='P', value_vars=None, id_vars=['ID', 'T'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH394', into=['MORPH469', 'Channel'])
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['MORPH469'])
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.select', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 12, 'timeout': False, '#sketches without SMT-based deduction': 45,
            '#partial programs with partial evaluation': 452, '#partial programs without partial evaluation': 5100,
            'Total time': 5.01, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.select'], 'gather_var_name': ['MORPH394'],
            'gather_value_name': ['P'], 'gather_id_vars': [['ID', 'T']], 'gather_value_vars': [['P.1', 'P.2', 'Q.1']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH394'], 'separate_into': ['MORPH469', 'Channel'],
            'select_keep_or_remove': [False], 'select_columns_keep': [None], 'select_columns_remove': [['MORPH469']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_6(self):
        b_id = "b_6"
        inputs = [pd.DataFrame({
            'GeneID': {
                1: 8876.5, 2: 2120.8, 3: 1266.6, }, 'D.1': {
                1: 510.5, 2: 480.3, 3: 213.8, }, 'T.1': {
                1: 4318.3, 2: 1694.6, 3: 1337.9, }, 'D.8': {
                1: 8957.7, 2: 2471.0, 3: 831.5, }, 'T.8': {
                1: 4092.4, 2: 1784.1, 3: 814.1, }, })]
        tbl_7 = rf.gather(inputs[0], var_name='MORPH36162', value_name='MORPH36161', value_vars=None,
                          id_vars=['GeneID', 'T.8'])
        tbl_3 = rf.separate(tbl_7, split_col='MORPH36162', into=['type', 'MORPH42956'])
        tbl_1 = rf.group_by(tbl_3, group_cols=['GeneID', 'type'])
        morpheus = rf.summarise(tbl_1, summaries={
            'sum': ('MORPH36161', 'sum'), })
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.group_by', [2])])
        intermediates = [tbl_7, tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 15,
            '#partial programs with partial evaluation': 4197, '#partial programs without partial evaluation': 47177,
            'Total time': 48.89, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.group_by', 'rf.summarise'], 'gather_var_name': ['MORPH36162'],
            'gather_value_name': ['MORPH36161'], 'gather_id_vars': [['GeneID', 'T.8']],
            'gather_value_vars': [['D.1', 'T.1', 'D.8']], 'gather_df': ['inputs[0]'],
            'separate_split_col': ['MORPH36162'], 'separate_into': ['type', 'MORPH42956'],
            'group_by_group_cols': [['GeneID', 'type']], 'summarise_new_col': ['sum'], 'summarise_agg': ['sum'],
            'summarise_col': ['MORPH36161'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_7(self):
        b_id = "b_7"
        inputs = [pd.DataFrame({
            'GeneID': {
                1: 'A2M', 2: 'ABL1', }, 'D.1': {
                1: 18, 2: 20, }, 'T.1': {
                1: 50, 2: 48, }, 'D.2': {
                1: 2, 2: 4, }, 'T.2': {
                1: 6, 2: 8, }, 'D.8': {
                1: 'A1', 2: 'C1', }, })]
        tbl_15 = rf.gather(inputs[0], var_name='MORPH72960', value_name='MORPH72959', value_vars=None,
                           id_vars=['GeneID', 'D.8'])
        tbl_7 = rf.separate(tbl_15, split_col='MORPH72960', into=['MORPH73167', 'pt.num'])
        tbl_3 = rf.spread(tbl_7, columns='MORPH73167', values='MORPH72959')
        tbl_1 = rf.mutate(tbl_3, new_col_name='Ratio', operation='div', col_args=['D', 'T'])
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['D.8'])
        output = morpheus
        skeleton = Skeleton(
            [('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2]), ('rf.mutate', [3]), ('rf.select', [4])])
        intermediates = [tbl_15, tbl_7, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 14, 'timeout': False, '#sketches without SMT-based deduction': 146,
            '#partial programs with partial evaluation': 21140, '#partial programs without partial evaluation': 80051,
            'Total time': 160.32, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread', 'rf.mutate', 'rf.select'],
            'gather_var_name': ['MORPH72960'], 'gather_value_name': ['MORPH72959'],
            'gather_id_vars': [['GeneID', 'D.8']], 'gather_value_vars': [['D.1', 'T.1', 'D.2', 'T.2']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH72960'], 'separate_into': ['MORPH73167', 'pt.num'],
            'spread_columns': ['MORPH73167'], 'spread_values': ['MORPH72959'], 'mutate_new_col_name': ['Ratio'],
            'mutate_operation': ['div'], 'mutate_col_args_normalize': [None], 'mutate_col_args_div': [['D', 'T']],
            'select_keep_or_remove': [False], 'select_columns_keep': [None], 'select_columns_remove': [['D.8']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_8(self):
        b_id = "b_8"
        inputs = [pd.DataFrame({
            'Name': {
                1: 'Aira', 2: 'Aira', 3: 'Ben', 4: 'Ben', 5: 'Cat', 6: 'Cat', }, 'Month': {
                1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, }, 'Rate1': {
                1: 12, 2: 18, 3: 53, 4: 22, 5: 22, 6: 67, }, 'Rate2': {
                1: 23, 2: 73, 3: 19, 4: 87, 5: 87, 6: 43, }, })]
        tbl_15 = rf.group_by(inputs[0], group_cols=['Name'])
        tbl_7 = rf.summarise(tbl_15, summaries={
            'avg2': ('Rate2', 'mean'), })
        tbl_3 = rf.inner_join(tbl_7, inputs[0])
        tbl_1 = rf.group_by(tbl_3, group_cols=['Name', 'avg2'])
        morpheus = rf.summarise(tbl_1, summaries={
            'avg1': ('Rate1', 'mean'), })
        output = morpheus
        skeleton = Skeleton([('rf.group_by', [(- 1)]), ('rf.inner_join', [1, (- 1)]), ('rf.group_by', [2])])
        intermediates = [tbl_7, tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 12, 'timeout': False, '#sketches without SMT-based deduction': 14,
            '#partial programs with partial evaluation': 2718, '#partial programs without partial evaluation': 25829,
            'Total time': 27.53, }
        replay_map = {
            'func_seq': ['rf.group_by', 'rf.summarise', 'rf.inner_join', 'rf.group_by', 'rf.summarise'],
            'group_by_group_cols': [['Name'], ['Name', 'avg2']], 'summarise_new_col': ['avg2', 'avg1'],
            'summarise_agg': ['mean', 'mean'], 'summarise_col': ['Rate2', 'Rate1'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_9(self):
        b_id = "b_9"
        inputs = [pd.DataFrame({
            'V1': {
                1: 'a', 2: 'a', 3: 'a', 4: 'a', 5: 'a', 6: 'a', 7: 'a', 8: 'a', 9: 'b', 10: 'b', 11: 'b', 12: 'b',
                13: 'b', 14: 'b', 15: 'b', 16: 'b', }, 'V2': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 3, 12: 4, 13: 1, 14: 2, 15: 3,
                16: 4, }, 'V3': {
                1: 'High', 2: 'High', 3: 'High', 4: 'High', 5: 'Low', 6: 'Low', 7: 'Low', 8: 'Low', 9: 'High',
                10: 'High', 11: 'High', 12: 'High', 13: 'Low', 14: 'Low', 15: 'Low', 16: 'Low', }, 'V4': {
                1: (- 0.62645381), 2: 0.18364332, 3: (- 0.83562861), 4: 1.5952808, 5: 0.32950777, 6: (- 0.82046838),
                7: 0.48742905, 8: 0.73832471, 9: 0.57578135, 10: (- 0.30538839), 11: 1.51178117, 12: 0.38984324,
                13: (- 0.62124058), 14: (- 2.21469989), 15: 1.12493092, 16: (- 0.04493361), }, })]
        tbl_3 = rf.spread(inputs[0], columns='V3', values='V4')
        tbl_1 = rf.mutate(tbl_3, new_col_name='Ratio', operation='div', col_args=['High', 'Low'])
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['High', 'Low'])
        output = morpheus
        skeleton = Skeleton([('rf.spread', [(- 1)]), ('rf.mutate', [1]), ('rf.select', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 33, 'timeout': False, '#sketches without SMT-based deduction': 73,
            '#partial programs with partial evaluation': 1248, '#partial programs without partial evaluation': 7362,
            'Total time': 8.59, }
        replay_map = {
            'func_seq': ['rf.spread', 'rf.mutate', 'rf.select'], 'spread_columns': ['V3'], 'spread_values': ['V4'],
            'mutate_new_col_name': ['Ratio'], 'mutate_operation': ['div'], 'mutate_col_args_normalize': [None],
            'mutate_col_args_div': [['High', 'Low']], 'select_keep_or_remove': [False], 'select_columns_keep': [None],
            'select_columns_remove': [['High', 'Low']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_10(self):
        b_id = "b_10"
        inputs = [pd.DataFrame({
            'date': {
                1: '2012-06-12', 2: '2012-07-13', 3: '2012-08-04', }, 'days': {
                1: 1.0, 2: 6.0, 3: 0.5, }, 'name': {
                1: 'Intro to stats', 2: 'Stats Winter school', 3: 'TidyR tools', }, 'topics': {
                1: 'probability|R', 2: 'R|regression', 3: 'tidyR|dplyr', }, })]
        tbl_3 = rf.separate(inputs[0], split_col='topics', into=['MORPH338', 'MORPH339'])
        tbl_1 = rf.gather(tbl_3, var_name='MORPH341', value_name='value2', value_vars=['MORPH338', 'MORPH339'],
                          id_vars=None)
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['MORPH341'])
        output = morpheus
        skeleton = Skeleton([('rf.separate', [(- 1)]), ('rf.gather', [1]), ('rf.select', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 26, 'timeout': False, '#sketches without SMT-based deduction': 118,
            '#partial programs with partial evaluation': 174, '#partial programs without partial evaluation': 1542,
            'Total time': 2.27, }
        replay_map = {
            'func_seq': ['rf.separate', 'rf.gather', 'rf.select'], 'separate_split_col': ['topics'],
            'separate_into': ['MORPH338', 'MORPH339'], 'gather_var_name': ['MORPH341'], 'gather_value_name': ['value2'],
            'gather_id_vars': [['date', 'days', 'name']], 'gather_value_vars': [['MORPH338', 'MORPH339']],
            'gather_df': ['tbl_3'], 'select_keep_or_remove': [False], 'select_columns_keep': [None],
            'select_columns_remove': [['MORPH341']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_11(self):
        b_id = "b_11"
        inputs = [pd.DataFrame({
            'expr': {
                1: 'base__1d4', 2: 'base__1d4', 3: 'base__1d5', 4: 'base__1d5', 5: 'dplyr_1d4', 6: 'dplyr_1d4',
                7: 'dplyr_1d5', 8: 'dplyr_1d5', }, 'time': {
                1: 4203379, 2: 4219165, 3: 59249811, 4: 59249833, 5: 4911550, 6: 4911533, 7: 72271322,
                8: 63373463, }, })]
        tbl_15 = rf.group_by(inputs[0], group_cols=['expr'])
        tbl_7 = rf.summarise(tbl_15, summaries={
            'MORPH6832': ('time', 'mean'), })
        tbl_3 = rf.separate(tbl_7, split_col='expr', into=['MORPH6835', 'size'])
        tbl_1 = rf.spread(tbl_3, columns='MORPH6835', values='MORPH6832')
        morpheus = rf.mutate(tbl_1, new_col_name='ratio', operation='div', col_args=['base', 'dplyr'])
        output = morpheus
        skeleton = Skeleton([('rf.group_by', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2]), ('rf.mutate', [3])])
        intermediates = [tbl_7, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 69, 'timeout': False, '#sketches without SMT-based deduction': 274,
            '#partial programs with partial evaluation': 2379, '#partial programs without partial evaluation': 8543,
            'Total time': 14.29, }
        replay_map = {
            'func_seq': ['rf.group_by', 'rf.summarise', 'rf.separate', 'rf.spread', 'rf.mutate'],
            'group_by_group_cols': [['expr']], 'summarise_new_col': ['MORPH6832'], 'summarise_agg': ['mean'],
            'summarise_col': ['time'], 'separate_split_col': ['expr'], 'separate_into': ['MORPH6835', 'size'],
            'spread_columns': ['MORPH6835'], 'spread_values': ['MORPH6832'], 'mutate_new_col_name': ['ratio'],
            'mutate_operation': ['div'], 'mutate_col_args_normalize': [None],
            'mutate_col_args_div': [['base', 'dplyr']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_12(self):
        b_id = "b_12"
        inputs = [pd.DataFrame({
            'flight': {
                1: 1141, 2: 725, 3: 461, 4: 1696, 5: 507, 6: 5708, 7: 79, 8: 301, 9: 11, 10: 495, 11: 1670, },
            'origin': {
                1: 'JFK', 2: 'JFK', 3: 'LGA', 4: 'EWR', 5: 'EWR', 6: 'LGA', 7: 'JFK', 8: 'LGA', 9: 'EWR', 10: 'JFK',
                11: 'EWR', }, 'dest': {
                1: 'MIA', 2: 'BQN', 3: 'ATL', 4: 'ORD', 5: 'FLL', 6: 'IAD', 7: 'MCO', 8: 'ORD', 9: 'SEA', 10: 'SEA',
                11: 'SEA', }, })]
        tbl_7 = rf.filter_(inputs[0], filter_expr="dest == 'SEA'")
        tbl_3 = rf.group_by(tbl_7, group_cols=['origin'])
        tbl_1 = rf.summarise(tbl_3, summaries={
            'n': (None, 'count'), })
        morpheus = rf.mutate(tbl_1, new_col_name='freq', operation='normalize', col_args='n')
        output = morpheus
        skeleton = Skeleton([('rf.filter', [(- 1)]), ('rf.group_by', [1]), ('rf.mutate', [2])])
        intermediates = [tbl_7, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 25, 'timeout': False, '#sketches without SMT-based deduction': 33,
            '#partial programs with partial evaluation': 1621, '#partial programs without partial evaluation': 10446,
            'Total time': 10.65, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.group_by', 'rf.summarise', 'rf.mutate'],
            'filter_mode': ['equality-inequality'], 'filter_column_eq': ['dest'], 'filter_column_relop': [],
            'filter_eq_op': ['=='], 'filter_relop': ['>'], 'filter_value_eq': ['SEA'], 'filter_value_relop': [],
            'group_by_group_cols': [['origin']], 'summarise_new_col': ['n'], 'summarise_agg': ['count'],
            'summarise_col': [], 'mutate_new_col_name': ['freq'], 'mutate_operation': ['normalize'],
            'mutate_col_args_normalize': ['n'], 'mutate_col_args_div': [None], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_13(self):
        b_id = "b_13"
        inputs = [pd.DataFrame({
            'id': {
                1: 20, 2: 20, 3: 30, 4: 30, }, 'type': {
                1: 'income', 2: 'expense', 3: 'income', 4: 'expense', }, 'transactions': {
                1: 20, 2: 25, 3: 50, 4: 45, }, 'amount': {
                1: 100, 2: 95, 3: 300, 4: 250, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH102', value_name='MORPH101', value_vars=['transactions', 'amount'],
                          id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH110', cols=['type', 'MORPH102'])
        morpheus = rf.spread(tbl_1, columns='MORPH110', values='MORPH101')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 22, '#partial programs without partial evaluation': 123,
            'Total time': 0.33, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH102'],
            'gather_value_name': ['MORPH101'], 'gather_id_vars': [['id', 'type']],
            'gather_value_vars': [['transactions', 'amount']], 'gather_df': ['inputs[0]'],
            'unite_new_col_name': ['MORPH110'], 'unite_cols': [['type', 'MORPH102']], 'spread_columns': ['MORPH110'],
            'spread_values': ['MORPH101'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_14(self):
        b_id = "b_14"
        inputs = [pd.DataFrame({
            'ID': {
                1: 'A', 2: 'A', 3: 'A', 4: 'B', 5: 'C', 6: 'C', 7: 'D', 8: 'E', 9: 'E', 10: 'E', }, 'Diagnosis_1': {
                1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, }, 'Diagnosis_2': {
                1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 0, 9: 1, 10: 0, }, 'Diagnosis_3': {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0, 10: 1, }, 'Diagnosis_4': {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 0, 10: 0, }, })]
        tbl_7 = rf.gather(inputs[0], var_name='MORPH144074', value_name='MORPH144073', value_vars=None, id_vars=['ID'])
        tbl_3 = rf.separate(tbl_7, split_col='MORPH144074', into=['MORPH144281', 'value'])
        tbl_1 = rf.filter_(tbl_3, filter_expr='MORPH144073 > 0.0')
        morpheus = rf.select(tbl_1, columns_keep=['ID', 'value'], columns_remove=None)
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.filter', [2]), ('rf.select', [3])])
        intermediates = [tbl_7, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 74, 'timeout': False, '#sketches without SMT-based deduction': 111,
            '#partial programs with partial evaluation': 21501, '#partial programs without partial evaluation': 181516,
            'Total time': 204.83, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.filter', 'rf.select'], 'gather_var_name': ['MORPH144074'],
            'gather_value_name': ['MORPH144073'], 'gather_id_vars': [['ID']],
            'gather_value_vars': [['Diagnosis_1', 'Diagnosis_2', 'Diagnosis_3', 'Diagnosis_4']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH144074'],
            'separate_into': ['MORPH144281', 'value'], 'filter_mode': ['relop'], 'filter_column_eq': [],
            'filter_column_relop': ['MORPH144073'], 'filter_eq_op': ['!='], 'filter_relop': ['>'],
            'filter_value_eq': [], 'filter_value_relop': [0.0], 'select_keep_or_remove': [True],
            'select_columns_keep': [['ID', 'value']], 'select_columns_remove': [None], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_15(self):
        b_id = "b_15"
        inputs = [pd.DataFrame({
            'Timepoint': {
                1: 7, 2: 14, }, 'Group1': {
                1: 60, 2: 66, }, 'Error1_Group1': {
                1: 4, 2: 6, }, 'Group2': {
                1: 60, 2: 90, }, 'Error2_Group1': {
                1: 14, 2: 16, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH596', value_name='MORPH595',
                          value_vars=['Error1_Group1', 'Error2_Group1'], id_vars=None)
        tbl_1 = rf.separate(tbl_3, split_col='MORPH596', into=['MORPH629', 'mGroup'])
        morpheus = rf.spread(tbl_1, columns='MORPH629', values='MORPH595')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 400, '#partial programs without partial evaluation': 905,
            'Total time': 3.63, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH596'],
            'gather_value_name': ['MORPH595'], 'gather_id_vars': [['Timepoint', 'Group1', 'Group2']],
            'gather_value_vars': [['Error1_Group1', 'Error2_Group1']], 'gather_df': ['inputs[0]'],
            'separate_split_col': ['MORPH596'], 'separate_into': ['MORPH629', 'mGroup'], 'spread_columns': ['MORPH629'],
            'spread_values': ['MORPH595'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_16(self):
        b_id = "b_16"
        inputs = [pd.DataFrame({
            'ID': {
                1: 1, 2: 2, 3: 7, 4: 8, 5: 9, }, 'Color': {
                1: 'red', 2: 'red', 3: 'red', 4: 'red', 5: 'yellow', }, 'Type': {
                1: 'Outdoor', 2: 'Indoor', 3: 'Indoor', 4: 'Outdoor', 5: 'Outdoor', }, 'W1': {
                1: 74.22, 2: 78.59, 3: 38.41, 4: 140.68, 5: 65.95, }, 'W2': {
                1: 26.86, 2: 138.8, 3: 84.81, 4: 93.33, 5: 104.31, }, })]
        tbl_15 = rf.filter_(inputs[0], filter_expr='W2 > 26.86')
        tbl_7 = rf.group_by(tbl_15, group_cols=['Color'])
        tbl_3 = rf.summarise(tbl_7, summaries={
            'sumCount': (None, 'count'), })
        tbl_1 = rf.group_by(tbl_3, group_cols=['Color'])
        morpheus = rf.summarise(tbl_1, summaries={
            'sumMean': ('sumCount', 'mean'), })
        output = morpheus
        skeleton = Skeleton([('rf.filter', [(- 1)]), ('rf.group_by', [1]), ('rf.group_by', [2])])
        intermediates = [tbl_15, tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 8, 'timeout': False, '#sketches without SMT-based deduction': 9,
            '#partial programs with partial evaluation': 4833, '#partial programs without partial evaluation': 24686,
            'Total time': 28.09, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.group_by', 'rf.summarise', 'rf.group_by', 'rf.summarise'],
            'filter_mode': ['relop'], 'filter_column_eq': [], 'filter_column_relop': ['W2'], 'filter_eq_op': ['!='],
            'filter_relop': ['>'], 'filter_value_eq': [], 'filter_value_relop': [26.86],
            'group_by_group_cols': [['Color'], ['Color']], 'summarise_new_col': ['sumCount', 'sumMean'],
            'summarise_agg': ['count', 'mean'], 'summarise_col': ['sumCount'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_17(self):
        b_id = "b_17"
        inputs = [pd.DataFrame({
            'Id': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, }, 'Group': {
                1: 'A', 2: 'A', 3: 'A', 4: 'B', 5: 'B', 6: 'B', }, 'Var1': {
                1: 'good', 2: 'good', 3: 'bad', 4: 'good', 5: 'good', 6: 'bad', }, 'Var2': {
                1: 10, 2: 6, 3: 9, 4: 3, 5: 3, 6: 8, }, })]
        tbl_7 = rf.filter_(inputs[0], filter_expr="Group == 'A'")
        tbl_3 = rf.group_by(tbl_7, group_cols=['Group', 'Var1'])
        tbl_1 = rf.summarise(tbl_3, summaries={
            'MORPH3405': ('Var2', 'sum'), })
        morpheus = rf.spread(tbl_1, columns='Var1', values='MORPH3405')
        output = morpheus
        skeleton = Skeleton([('rf.filter', [(- 1)]), ('rf.group_by', [1]), ('rf.spread', [2])])
        intermediates = [tbl_7, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 14, 'timeout': False, '#sketches without SMT-based deduction': 36,
            '#partial programs with partial evaluation': 3491, '#partial programs without partial evaluation': 9332,
            'Total time': 31.63, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.group_by', 'rf.summarise', 'rf.spread'],
            'filter_mode': ['equality-inequality'], 'filter_column_eq': ['Group'], 'filter_column_relop': [],
            'filter_eq_op': ['=='], 'filter_relop': ['>'], 'filter_value_eq': ['A'], 'filter_value_relop': [],
            'group_by_group_cols': [['Group', 'Var1']], 'summarise_new_col': ['MORPH3405'], 'summarise_agg': ['sum'],
            'summarise_col': ['Var2'], 'spread_columns': ['Var1'], 'spread_values': ['MORPH3405'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_18(self):
        b_id = "b_18"
        inputs = [pd.DataFrame({
            'message.id': {
                1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, }, 'sender': {
                1: 'A', 2: 'A', 3: 'A', 4: 'B', 5: 'C', 6: 'D', }, 'recipient': {
                1: 'A', 2: 'C', 3: 'B', 4: 'C', 5: 'D', 6: 'B', }, })]
        tbl_7 = rf.gather(inputs[0], var_name='MORPH4133', value_name='address', value_vars=['sender', 'recipient'],
                          id_vars=None)
        tbl_3 = rf.group_by(tbl_7, group_cols=['MORPH4133', 'address'])
        tbl_1 = rf.summarise(tbl_3, summaries={
            'MORPH4140': (None, 'count'), })
        morpheus = rf.spread(tbl_1, columns='MORPH4133', values='MORPH4140')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.group_by', [1]), ('rf.spread', [2])])
        intermediates = [tbl_7, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 20, 'timeout': False, '#sketches without SMT-based deduction': 26,
            '#partial programs with partial evaluation': 2078, '#partial programs without partial evaluation': 6637,
            'Total time': 9.05, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.group_by', 'rf.summarise', 'rf.spread'], 'gather_var_name': ['MORPH4133'],
            'gather_value_name': ['address'], 'gather_id_vars': [['message.id']],
            'gather_value_vars': [['sender', 'recipient']], 'gather_df': ['inputs[0]'],
            'group_by_group_cols': [['MORPH4133', 'address']], 'summarise_new_col': ['MORPH4140'],
            'summarise_agg': ['count'], 'summarise_col': [], 'spread_columns': ['MORPH4133'],
            'spread_values': ['MORPH4140'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_19(self):
        b_id = "b_19"
        inputs = [pd.DataFrame({
            '12:10': {
                1: 'nuclear', 2: 'nuclear', 3: 'child', 4: 'child', 5: 'child', 6: 'nuclear and acquaintance',
                7: 'nuclear and acquaintance', 8: 'notnotnot', 9: 'nuclear', }, '12:20': {
                1: 'nuclear', 2: 'nuclear', 3: 'child', 4: 'child', 5: 'child', 6: 'nuclear and acquaintance',
                7: 'nuclear and acquaintance', 8: 'notnotnot', 9: 'nuclear', }, '12:30': {
                1: 'nuclear', 2: 'child', 3: 'child', 4: 'child', 5: 'child', 6: 'nuclear and acquaintance',
                7: 'nuclear and acquaintance', 8: 'notnotnot', 9: 'nuclear', }, })]
        tbl_7 = rf.gather(inputs[0], var_name='MORPH38', value_name='value', value_vars=['12:10', '12:20', '12:30'],
                          id_vars=None)
        tbl_3 = rf.group_by(tbl_7, group_cols=['MORPH38', 'value'])
        tbl_1 = rf.summarise(tbl_3, summaries={
            'MORPH39': (None, 'count'), })
        morpheus = rf.spread(tbl_1, columns='MORPH38', values='MORPH39')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.group_by', [1]), ('rf.spread', [2])])
        intermediates = [tbl_7, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 7, 'timeout': False, '#sketches without SMT-based deduction': 26,
            '#partial programs with partial evaluation': 60, '#partial programs without partial evaluation': 85,
            'Total time': 0.56, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.group_by', 'rf.summarise', 'rf.spread'], 'gather_var_name': ['MORPH38'],
            'gather_value_name': ['value'], 'gather_id_vars': [[]], 'gather_value_vars': [['12:10', '12:20', '12:30']],
            'gather_df': ['inputs[0]'], 'group_by_group_cols': [['MORPH38', 'value']], 'summarise_new_col': ['MORPH39'],
            'summarise_agg': ['count'], 'summarise_col': [], 'spread_columns': ['MORPH38'],
            'spread_values': ['MORPH39'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_20(self):
        b_id = "b_20"
        inputs = [pd.DataFrame({
            'group': {
                1: 'a', 2: 'a', 3: 'b', 4: 'b', }, 'times': {
                1: 'before', 2: 'after', 3: 'before', 4: 'after', }, 'action_rate': {
                1: 0.1, 2: 0.15, 3: 0.2, 4: 0.18, }, 'action_rate2': {
                1: 0.2, 2: 0.25, 3: 0.3, 4: 0.28, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH72', value_name='MORPH71',
                          value_vars=['action_rate', 'action_rate2'], id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH77', cols=['MORPH72', 'times'])
        morpheus = rf.spread(tbl_1, columns='MORPH77', values='MORPH71')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 17, '#partial programs without partial evaluation': 88,
            'Total time': 0.21, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH72'],
            'gather_value_name': ['MORPH71'], 'gather_id_vars': [['group', 'times']],
            'gather_value_vars': [['action_rate', 'action_rate2']], 'gather_df': ['inputs[0]'],
            'unite_new_col_name': ['MORPH77'], 'unite_cols': [['MORPH72', 'times']], 'spread_columns': ['MORPH77'],
            'spread_values': ['MORPH71'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_21(self):
        b_id = "b_21"
        inputs = [pd.DataFrame({
            'obs': {
                1: 1, 2: 2, 3: 3, 4: 4, }, 'pre.data1': {
                1: 0.4, 2: 0.21, 3: 0.48, 4: 0.66, }, 'post.data1': {
                1: 0.12, 2: 0.05, 3: 0.85, 4: 0.29, }, 'pre.data2': {
                1: 0.61, 2: 0.18, 3: 0.0, 4: 0.88, }, 'post.data2': {
                1: 0.15, 2: 0.49, 3: 0.62, 4: 0.56, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH356', value_name='MORPH355', value_vars=None, id_vars=['obs'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH356', into=['key', 'MORPH388'])
        morpheus = rf.spread(tbl_1, columns='MORPH388', values='MORPH355')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 389, '#partial programs without partial evaluation': 704,
            'Total time': 3.64, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH356'],
            'gather_value_name': ['MORPH355'], 'gather_id_vars': [['obs']],
            'gather_value_vars': [['pre.data1', 'post.data1', 'pre.data2', 'post.data2']], 'gather_df': ['inputs[0]'],
            'separate_split_col': ['MORPH356'], 'separate_into': ['key', 'MORPH388'], 'spread_columns': ['MORPH388'],
            'spread_values': ['MORPH355'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_22(self):
        b_id = "b_22"
        inputs = [pd.DataFrame({
            'Player': {
                1: 'Abdoun', 2: 'Abe', 3: 'Abidal', 4: 'Abreu', }, 'Team': {
                1: 'Algeria', 2: 'Japan', 3: 'France', 4: 'Uruguay', }, 'Shots': {
                1: 0, 2: 3, 3: 0, 4: 5, }, 'Passes': {
                1: 6, 2: 101, 3: 91, 4: 15, }, 'Tackles': {
                1: 0, 2: 14, 3: 6, 4: 0, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='Var', value_name='MORPH2221', value_vars=['Passes', 'Tackles'],
                          id_vars=None)
        tbl_1 = rf.group_by(tbl_3, group_cols=['Var'])
        morpheus = rf.summarise(tbl_1, summaries={
            'Mean': ('MORPH2221', 'mean'), })
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.group_by', [1])])
        intermediates = [tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 5, 'timeout': False, '#sketches without SMT-based deduction': 6,
            '#partial programs with partial evaluation': 150, '#partial programs without partial evaluation': 1522,
            'Total time': 1.46, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.group_by', 'rf.summarise'], 'gather_var_name': ['Var'],
            'gather_value_name': ['MORPH2221'], 'gather_id_vars': [['Player', 'Team', 'Shots']],
            'gather_value_vars': [['Passes', 'Tackles']], 'gather_df': ['inputs[0]'], 'group_by_group_cols': [['Var']],
            'summarise_new_col': ['Mean'], 'summarise_agg': ['mean'], 'summarise_col': ['MORPH2221'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_23(self):
        b_id = "b_23"
        inputs = [pd.DataFrame({
            'custno': {
                1: 100, 2: 100, 3: 100, }, 'X1': {
                1: 29.85, 2: 122.7, 3: 0.0, }, 'X2': {
                1: 49.75, 2: 49.75, 3: 9.95, }, 'X3': {
                1: 146.7, 2: 39.8, 3: 44.95, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH3988', value_name='MORPH3987', value_vars=None, id_vars=['custno'])
        tbl_1 = rf.group_by(tbl_3, group_cols=['custno'])
        morpheus = rf.summarise(tbl_1, summaries={
            'totalspent': ('MORPH3987', 'sum'), })
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.group_by', [1])])
        intermediates = [tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 4,
            '#partial programs with partial evaluation': 573, '#partial programs without partial evaluation': 5141,
            'Total time': 5.03, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.group_by', 'rf.summarise'], 'gather_var_name': ['MORPH3988'],
            'gather_value_name': ['MORPH3987'], 'gather_id_vars': [['custno']],
            'gather_value_vars': [['X1', 'X2', 'X3']], 'gather_df': ['inputs[0]'], 'group_by_group_cols': [['custno']],
            'summarise_new_col': ['totalspent'], 'summarise_agg': ['sum'], 'summarise_col': ['MORPH3987'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_24(self):
        b_id = "b_24"
        inputs = [pd.DataFrame({
            'id': {
                1: 1, 2: 2, 3: 3, 4: 4, }, 'yr1': {
                1: 1090, 2: 1026, 3: 1036, 4: 1056, }, 'yr2': {
                1: 2066, 2: 2062, 3: 2006, 4: 2020, }, 'yr3': {
                1: 3050, 2: 3071, 3: 3098, 4: 3037, }, 'yr4': {
                1: 4012, 2: 4026, 3: 4038, 4: 4001, }, 'var': {
                1: 'yr3', 2: 'yr2', 3: 'yr1', 4: 'yr4', }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH1429', value_name='value', value_vars=None, id_vars=['id', 'var'])
        tbl_1 = rf.filter_(tbl_3, filter_expr='value > 1090.0')
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['MORPH1429'])
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.filter', [1]), ('rf.select', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 5, 'timeout': False, '#sketches without SMT-based deduction': 17,
            '#partial programs with partial evaluation': 946, '#partial programs without partial evaluation': 5059,
            'Total time': 6.25, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.filter', 'rf.select'], 'gather_var_name': ['MORPH1429'],
            'gather_value_name': ['value'], 'gather_id_vars': [['id', 'var']],
            'gather_value_vars': [['yr1', 'yr2', 'yr3', 'yr4']], 'gather_df': ['inputs[0]'], 'filter_mode': ['relop'],
            'filter_column_eq': [], 'filter_column_relop': ['value'], 'filter_eq_op': ['!='], 'filter_relop': ['>'],
            'filter_value_eq': [], 'filter_value_relop': [1090.0], 'select_keep_or_remove': [False],
            'select_columns_keep': [None], 'select_columns_remove': [['MORPH1429']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_25(self):
        b_id = "b_25"
        inputs = [pd.DataFrame({
            'a': {
                1: 1, 2: 1, 3: 4, 4: 4, 5: 1, 6: 1, }, 'b': {
                1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 2, }, })]
        tbl_7 = rf.filter_(inputs[0], filter_expr='b > 1.0')
        tbl_3 = rf.unite(tbl_7, new_col_name='key_ab', cols=['a', 'b'])
        tbl_1 = rf.group_by(tbl_3, group_cols=['key_ab'])
        morpheus = rf.summarise(tbl_1, summaries={
            'e': (None, 'count'), })
        output = morpheus
        skeleton = Skeleton([('rf.filter', [(- 1)]), ('rf.unite', [1]), ('rf.group_by', [2])])
        intermediates = [tbl_7, tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 263, 'timeout': False, '#sketches without SMT-based deduction': 537,
            '#partial programs with partial evaluation': 5346, '#partial programs without partial evaluation': 25487,
            'Total time': 27.94, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.unite', 'rf.group_by', 'rf.summarise'], 'filter_mode': ['relop'],
            'filter_column_eq': [], 'filter_column_relop': ['b'], 'filter_eq_op': ['!='], 'filter_relop': ['>'],
            'filter_value_eq': [], 'filter_value_relop': [1.0], 'unite_new_col_name': ['key_ab'],
            'unite_cols': [['a', 'b']], 'group_by_group_cols': [['key_ab']], 'summarise_new_col': ['e'],
            'summarise_agg': ['count'], 'summarise_col': [], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_26(self):
        b_id = "b_26"
        inputs = [pd.DataFrame({
            'frame': {
                1: 1, 2: 2, 3: 3, 4: 4, }, 'X1': {
                1: 0, 2: 10, 3: 15, 4: 15, }, 'X2': {
                1: 0, 2: 15, 3: 10, 4: 10, }, 'X3': {
                1: 0, 2: 0, 3: 0, 4: 0, }, }), pd.DataFrame({
            'frame': {
                1: 1, 2: 2, 3: 3, 4: 4, }, 'X1': {
                1: 0.0, 2: 14.53, 3: 13.9, 4: 14.1, }, 'X2': {
                1: 0.0, 2: 12.57, 3: 14.65, 4: 14.7, }, 'X3': {
                1: 0, 2: 0, 3: 0, 4: 0, }, })]
        tbl_4 = rf.gather(inputs[0], var_name='pos', value_name='carid', value_vars=None, id_vars=['frame'])
        tbl_3 = rf.gather(inputs[1], var_name='pos', value_name='speed', value_vars=None, id_vars=['frame'])
        tbl_1 = rf.inner_join(tbl_3, tbl_4)
        morpheus = rf.filter_(tbl_1, filter_expr='carid > 0.0')
        output = morpheus
        skeleton = Skeleton(
            [('rf.gather', [(- 1)]), ('rf.gather', [(- 2)]), ('rf.inner_join', [2, 1]), ('rf.filter', [3])])
        intermediates = [tbl_4, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 131, 'timeout': False, '#sketches without SMT-based deduction': 462,
            '#partial programs with partial evaluation': 45567, '#partial programs without partial evaluation': 87394,
            'Total time': 130.92, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.gather', 'rf.inner_join', 'rf.filter'], 'gather_var_name': ['pos', 'pos'],
            'gather_value_name': ['carid', 'speed'], 'gather_id_vars': [['frame'], ['frame']],
            'gather_value_vars': [['X1', 'X2', 'X3'], ['X1', 'X2', 'X3']], 'gather_df': ['inputs[0]', 'inputs[1]'],
            'filter_mode': ['relop'], 'filter_column_eq': [], 'filter_column_relop': ['carid'], 'filter_eq_op': ['!='],
            'filter_relop': ['>'], 'filter_value_eq': [], 'filter_value_relop': [0.0], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_27(self):
        b_id = "b_27"
        inputs = [pd.DataFrame({
            'id': {
                1: 1, 2: 3, 3: 4, 4: 2, 5: 5, }, 'prod': {
                1: 8, 2: 2, 3: 7, 4: 8, 5: 8, }, 'clnt': {
                1: 5, 2: 6, 3: 1, 4: 4, 5: 5, }, 'order': {
                1: 6.9129309999999995, 2: 5.119676, 3: 7.472010000000001, 4: 7.345583, 5: 9.41205, }, }), pd.DataFrame({
            'id': {
                1: 3, 2: 5, }, 'prod': {
                1: 2, 2: 8, }, 'clnt': {
                1: 6, 2: 5, }, 'order': {
                1: 5.119676, 2: 9.41205, }, })]
        tbl_9 = rf.group_by(inputs[0], group_cols=['prod', 'clnt'])
        tbl_4 = rf.summarise(tbl_9, summaries={
            'mean_order': ('order', 'mean'), })
        tbl_1 = rf.inner_join(inputs[1], tbl_4)
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['id', 'order'])
        output = morpheus
        skeleton = Skeleton([('rf.group_by', [(- 1)]), ('rf.inner_join', [(- 2), 1]), ('rf.select', [2])])
        intermediates = [tbl_4, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 3,
            '#partial programs with partial evaluation': 777, '#partial programs without partial evaluation': 1959,
            'Total time': 2.76, }
        replay_map = {
            'func_seq': ['rf.group_by', 'rf.summarise', 'rf.inner_join', 'rf.select'],
            'group_by_group_cols': [['prod', 'clnt']], 'summarise_new_col': ['mean_order'], 'summarise_agg': ['mean'],
            'summarise_col': ['order'], 'select_keep_or_remove': [False], 'select_columns_keep': [None],
            'select_columns_remove': [['id', 'order']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_28(self):
        b_id = "b_28"
        inputs = [pd.DataFrame({
            'ID': {
                1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3, }, 'Species': {
                1: 'Callvulg', 2: 'Callvulg', 3: 'Callvulg', 4: 'Empenigr', 5: 'Empenigr', 6: 'Empenigr', 7: 'Rhodtome',
                8: 'Rhodtome', 9: 'Rhodtome', }, 'Value': {
                1: 0.55, 2: 0.67, 3: 0.1, 4: 11.13, 5: 0.17, 6: 1.55, 7: 0.17, 8: 1.55, 9: 3.0, }, }), pd.DataFrame({
            'Species': {
                1: 'Callvulg', 2: 'Empenigr', 3: 'Rhodtome', }, 'Attribute': {
                1: 'MI', 2: 'MI', 3: 'PI', }, })]
        tbl_7 = rf.inner_join(inputs[1], inputs[0])
        tbl_3 = rf.filter_(tbl_7, filter_expr="Attribute == 'MI'")
        tbl_1 = rf.group_by(tbl_3, group_cols=['ID'])
        morpheus = rf.summarise(tbl_1, summaries={
            'Total': ('Value', 'sum'), })
        output = morpheus
        skeleton = Skeleton([('rf.inner_join', [(- 2), (- 1)]), ('rf.filter', [1]), ('rf.group_by', [2])])
        intermediates = [tbl_7, tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 8, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 949, '#partial programs without partial evaluation': 3642,
            'Total time': 4.55, }
        replay_map = {
            'func_seq': ['rf.inner_join', 'rf.filter', 'rf.group_by', 'rf.summarise'],
            'filter_mode': ['equality-inequality'], 'filter_column_eq': ['Attribute'], 'filter_column_relop': [],
            'filter_eq_op': ['=='], 'filter_relop': ['>'], 'filter_value_eq': ['MI'], 'filter_value_relop': [],
            'group_by_group_cols': [['ID']], 'summarise_new_col': ['Total'], 'summarise_agg': ['sum'],
            'summarise_col': ['Value'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_29(self):
        b_id = "b_29"
        inputs = [pd.DataFrame({
            'sym': {
                1: 'a', 2: 'a', 3: 'a', 4: 'b', 5: 'b', 6: 'b', }, 'a1': {
                1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, }, 'a2': {
                1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, }, 'b1': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, }, 'b2': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, }, })]
        tbl_5 = rf.group_by(inputs[0], group_cols=['sym', 'a2'])
        tbl_3 = rf.group_by(inputs[0], group_cols=['a1', 'a2'])
        tbl_2 = rf.summarise(tbl_5, summaries={
            'b2_mean': ('b2', 'mean'), })
        tbl_1 = rf.summarise(tbl_3, summaries={
            'b1_mean': ('b2', 'mean'), })
        morpheus = rf.inner_join(tbl_1, tbl_2)
        output = morpheus
        skeleton = Skeleton([('rf.group_by', [(- 1)]), ('rf.group_by', [(- 1)]), ('rf.inner_join', [2, 1])])
        intermediates = [tbl_2, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 4, 'timeout': False, '#sketches without SMT-based deduction': 22,
            '#partial programs with partial evaluation': 8354, '#partial programs without partial evaluation': 28497,
            'Total time': 53.12, }
        replay_map = {
            'func_seq': ['rf.group_by', 'rf.group_by', 'rf.summarise', 'rf.summarise', 'rf.inner_join'],
            'group_by_group_cols': [['sym', 'a2'], ['a1', 'a2']], 'summarise_new_col': ['b2_mean', 'b1_mean'],
            'summarise_agg': ['mean', 'mean'], 'summarise_col': ['b2', 'b2'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_30(self):
        b_id = "b_30"
        inputs = [pd.DataFrame({
            'Factor': {
                1: 'K', 2: 'L', 3: 'M', }, 'A.measure': {
                1: 52127803, 2: 63410326, 3: 76455662, }, 'A.SD': {
                1: 9124563, 2: 21975533, 3: 9864019, }, 'B.measure': {
                1: 63752981, 2: 68303447, 3: 73250794, }, 'B.SD': {
                1: 34800000, 2: 22600000, 3: 6090000, }, 'C.measure': {
                1: 103512032, 2: 65074191, 3: 92686983, }, 'C.SD': {
                1: 23900000, 2: 20800000, 3: 38300000, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH300', value_name='MORPH299', value_vars=None, id_vars=['Factor'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH300', into=['measure_letter', 'MORPH344'])
        morpheus = rf.spread(tbl_1, columns='MORPH344', values='MORPH299')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 335, '#partial programs without partial evaluation': 706,
            'Total time': 2.03, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH300'],
            'gather_value_name': ['MORPH299'], 'gather_id_vars': [['Factor']],
            'gather_value_vars': [['A.measure', 'A.SD', 'B.measure', 'B.SD', 'C.measure', 'C.SD']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH300'],
            'separate_into': ['measure_letter', 'MORPH344'], 'spread_columns': ['MORPH344'],
            'spread_values': ['MORPH299'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_31(self):
        b_id = "b_31"
        inputs = [pd.DataFrame({
            'id': {
                1: 101, 2: 102, 3: 103, }, 'a': {
                1: 1, 2: 2, 3: 3, }, 'b': {
                1: 2, 2: 2, 3: 2, }, 'c': {
                1: 3, 2: 3, 3: 3, }, })]
        tbl_7 = rf.gather(inputs[0], var_name='MORPH26762', value_name='MORPH26761', value_vars=None, id_vars=['id'])
        tbl_3 = rf.group_by(tbl_7, group_cols=['id'])
        tbl_1 = rf.summarise(tbl_3, summaries={
            'mean': ('MORPH26761', 'mean'), })
        morpheus = rf.inner_join(tbl_1, inputs[0])
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.group_by', [1]), ('rf.inner_join', [2, (- 1)])])
        intermediates = [tbl_7, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 46, 'timeout': False, '#sketches without SMT-based deduction': 125,
            '#partial programs with partial evaluation': 16082, '#partial programs without partial evaluation': 51417,
            'Total time': 97.3, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.group_by', 'rf.summarise', 'rf.inner_join'],
            'gather_var_name': ['MORPH26762'], 'gather_value_name': ['MORPH26761'], 'gather_id_vars': [['id']],
            'gather_value_vars': [['a', 'b', 'c']], 'gather_df': ['inputs[0]'], 'group_by_group_cols': [['id']],
            'summarise_new_col': ['mean'], 'summarise_agg': ['mean'], 'summarise_col': ['MORPH26761'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_32(self):
        b_id = "b_32"
        inputs = [pd.DataFrame({
            'rowname': {
                1: 1, 2: 2, 3: 3, }, 'CA': {
                1: 'A002', 2: 'A002', 3: 'A002', }, 'DATE_1': {
                1: '07-27-13', 2: '07-28-13', 3: '07-29-13', }, 'TIME_1': {
                1: '00:00:00', 2: '08:00:00', 3: '16:00:00', }, 'ENTRIES_1': {
                1: 4209603, 2: 4210490, 3: 4211586, }, 'DATE_2': {
                1: '07-27-13', 2: '07-28-13', 3: '07-30-13', }, 'TIME_2': {
                1: '08:00:00', 2: '16:00:00', 3: '00:00:00', }, 'ENTRIES_2': {
                1: 4209663, 2: 4210775, 3: 4212845, }, })]
        tbl_7 = rf.gather(inputs[0], var_name='MORPH26186', value_name='MORPH26185', value_vars=None,
                          id_vars=['rowname', 'CA'])
        tbl_3 = rf.separate(tbl_7, split_col='MORPH26186', into=['MORPH26841', 'MORPH26842'])
        tbl_1 = rf.spread(tbl_3, columns='MORPH26841', values='MORPH26185')
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['rowname', 'MORPH26842'])
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2]), ('rf.select', [3])])
        intermediates = [tbl_7, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 6, 'timeout': False, '#sketches without SMT-based deduction': 51,
            '#partial programs with partial evaluation': 8227, '#partial programs without partial evaluation': 35013,
            'Total time': 75.14, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread', 'rf.select'], 'gather_var_name': ['MORPH26186'],
            'gather_value_name': ['MORPH26185'], 'gather_id_vars': [['rowname', 'CA']],
            'gather_value_vars': [['DATE_1', 'TIME_1', 'ENTRIES_1', 'DATE_2', 'TIME_2', 'ENTRIES_2']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH26186'],
            'separate_into': ['MORPH26841', 'MORPH26842'], 'spread_columns': ['MORPH26841'],
            'spread_values': ['MORPH26185'], 'select_keep_or_remove': [False], 'select_columns_keep': [None],
            'select_columns_remove': [['rowname', 'MORPH26842']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_33(self):
        b_id = "b_33"
        inputs = [pd.DataFrame({
            'mpg_min': {
                1: 10.4, }, 'cyl_min': {
                1: 4, }, 'vs_min': {
                1: 0, }, 'am_min': {
                1: 0, }, 'gear_min': {
                1: 3, }, 'carb_min': {
                1: 1, }, 'mpg_q25': {
                1: 15.425, }, 'cyl_q25': {
                1: 4, }, 'vs_q25': {
                1: 0, }, 'am_q25': {
                1: 0, }, 'gear_q25': {
                1: 3, }, 'carb_q25': {
                1: 2, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH7988', value_name='MORPH7987',
                          value_vars=['mpg_min', 'cyl_min', 'vs_min', 'am_min', 'gear_min', 'carb_min', 'mpg_q25',
                                      'cyl_q25', 'vs_q25', 'am_q25', 'gear_q25', 'carb_q25'], id_vars=None)
        tbl_1 = rf.separate(tbl_3, split_col='MORPH7988', into=['var', 'MORPH7992'])
        morpheus = rf.spread(tbl_1, columns='MORPH7992', values='MORPH7987')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 1462, '#partial programs without partial evaluation': 10338,
            'Total time': 9.96, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH7988'],
            'gather_value_name': ['MORPH7987'], 'gather_id_vars': [[]], 'gather_value_vars': [
                ['mpg_min', 'cyl_min', 'vs_min', 'am_min', 'gear_min', 'carb_min', 'mpg_q25', 'cyl_q25', 'vs_q25',
                 'am_q25', 'gear_q25', 'carb_q25']], 'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH7988'],
            'separate_into': ['var', 'MORPH7992'], 'spread_columns': ['MORPH7992'], 'spread_values': ['MORPH7987'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_34(self):
        b_id = "b_34"
        inputs = [pd.DataFrame({
            'size': {
                1: 1, 2: 2, 3: 3, }, 'mult': {
                1: 'K', 2: 'M', 3: 'G', }, }), pd.DataFrame({
            'value': {
                1: 230, 2: 128, 3: 420, }, 'mult': {
                1: 'K', 2: 'M', 3: 'G', }, })]
        tbl_3 = rf.inner_join(inputs[1], inputs[0])
        tbl_1 = rf.mutate(tbl_3, new_col_name='total', operation='div', col_args=['value', 'size'])
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['value'])
        output = morpheus
        skeleton = Skeleton([('rf.inner_join', [(- 2), (- 1)]), ('rf.mutate', [1]), ('rf.select', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 7, 'timeout': False, '#sketches without SMT-based deduction': 7,
            '#partial programs with partial evaluation': 35, '#partial programs without partial evaluation': 148,
            'Total time': 0.24, }
        replay_map = {
            'func_seq': ['rf.inner_join', 'rf.mutate', 'rf.select'], 'mutate_new_col_name': ['total'],
            'mutate_operation': ['div'], 'mutate_col_args_normalize': [None],
            'mutate_col_args_div': [['value', 'size']], 'select_keep_or_remove': [False], 'select_columns_keep': [None],
            'select_columns_remove': [['value']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_35(self):
        b_id = "b_35"
        inputs = [pd.DataFrame({
            'name1': {
                1: 'a', 2: 'b', 3: 'c', }, 'con1_1': {
                1: 23, 2: 25, 3: 28, }, 'con1_2': {
                1: 33, 2: 34, 3: 29, }, 'con2_1': {
                1: 23, 2: 22, 3: 30, }, 'con2_2': {
                1: 40, 2: 50, 3: 60, }, })]
        tbl_1 = rf.gather(inputs[0], var_name='MORPH1', value_name='MORPH2',
                          value_vars=['con1_1', 'con1_2', 'con2_1', 'con2_2'],
                          id_vars=['name1'])
        tbl_2 = rf.separate(tbl_1, split_col='MORPH1', into=['MORPH3', 'MORPH4'])
        tbl_3 = rf.group_by(tbl_2, group_cols=['name1', 'MORPH3'])
        tbl_4 = rf.summarise(tbl_3, summaries={
            'mean_val': ('MORPH2', 'mean'), })
        output = rf.spread(tbl_4, columns='MORPH3', values='mean_val')
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.separate', [1]), ('rf.group_by', [2]), ('rf.spread', [3])])
        intermediates = [tbl_1, tbl_2, tbl_3, tbl_4]
        stats = {
            'timeout': True,
            '#sketches with SMT-based deduction': 0, '#sketches without SMT-based deduction': 0,
            '#partial programs with partial evaluation': 0, '#partial programs without partial evaluation': 0,
            'Total time': 0,
        }
        replay_map = {
            'gather_var_name': ['MORPH1'], 'gather_value_name': ['MORPH2'], 'gather_id_vars': [['name1']],
            'gather_value_vars': [['con1_1', 'con1_2', 'con2_1', 'con2_2']],
            'separate_split_col': ['MORPH1'], 'separate_into': ['MORPH3', 'MORPH4'], 'spread_columns': ['MORPH3'],
            'spread_values': ['mean_val'], 'group_by_group_cols': [['name1', 'MORPH3']],
            'summarise_new_col': ['mean_val'], 'summarise_agg': ['mean'], 'summarise_col': ['MORPH2']
        }

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_36(self):
        b_id = "b_36"
        inputs = [pd.DataFrame({
            'gear': {
                1: 3, 2: 4, 3: 4, 4: 3, }, 'am': {
                1: 0, 2: 0, 3: 1, 4: 1, }, 'n': {
                1: 15, 2: 4, 3: 8, 4: 5, }, })]
        tbl_7 = rf.mutate(inputs[0], new_col_name='MORPH2776', operation='normalize', col_args='n')
        tbl_3 = rf.gather(tbl_7, var_name='MORPH2778', value_name='MORPH2777', value_vars=['n', 'MORPH2776'],
                          id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH2786', cols=['am', 'MORPH2778'])
        morpheus = rf.spread(tbl_1, columns='MORPH2786', values='MORPH2777')
        output = morpheus
        skeleton = Skeleton([('rf.mutate', [(- 1)]), ('rf.gather', [1]), ('rf.unite', [2]), ('rf.spread', [3])])
        intermediates = [tbl_7, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 10, 'timeout': False, '#sketches without SMT-based deduction': 65,
            '#partial programs with partial evaluation': 924, '#partial programs without partial evaluation': 2819,
            'Total time': 4.89, }
        replay_map = {
            'func_seq': ['rf.mutate', 'rf.gather', 'rf.unite', 'rf.spread'], 'mutate_new_col_name': ['MORPH2776'],
            'mutate_operation': ['normalize'], 'mutate_col_args_normalize': ['n'], 'mutate_col_args_div': [None],
            'gather_var_name': ['MORPH2778'], 'gather_value_name': ['MORPH2777'], 'gather_id_vars': [['gear', 'am']],
            'gather_value_vars': [['n', 'MORPH2776']], 'gather_df': ['tbl_7'], 'unite_new_col_name': ['MORPH2786'],
            'unite_cols': [['am', 'MORPH2778']], 'spread_columns': ['MORPH2786'], 'spread_values': ['MORPH2777'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_37(self):
        b_id = "b_37"
        inputs = [pd.DataFrame({
            'id': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, }, 'sex': {
                1: 'M', 2: 'M', 3: 'F', 4: 'M', 5: 'F', 6: 'M', }, 'trt.1': {
                1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'A', 6: 'A', }, 'response.1': {
                1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, }, 'trt.2': {
                1: 'B', 2: 'B', 3: 'B', 4: 'B', 5: 'B', 6: 'B', }, 'response.2': {
                1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH672', value_name='MORPH671', value_vars=None, id_vars=['id', 'sex'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH672', into=['MORPH791', 'number'])
        morpheus = rf.spread(tbl_1, columns='MORPH791', values='MORPH671')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 723, '#partial programs without partial evaluation': 3054,
            'Total time': 6.13, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH672'],
            'gather_value_name': ['MORPH671'], 'gather_id_vars': [['id', 'sex']],
            'gather_value_vars': [['trt.1', 'response.1', 'trt.2', 'response.2']], 'gather_df': ['inputs[0]'],
            'separate_split_col': ['MORPH672'], 'separate_into': ['MORPH791', 'number'], 'spread_columns': ['MORPH791'],
            'spread_values': ['MORPH671'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_38(self):
        b_id = "b_38"
        inputs = [pd.DataFrame({
            'x': {
                1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', }, }), pd.DataFrame({
            'x': {
                1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'a', 12: 'b',
                13: 'c', 14: 'd', 15: 'e', 16: 'f', 17: 'g', 18: 'h', 19: 'i', 20: 'j', }, 'z': {
                1: 93.7354618925767, 2: 101.83643324222099, 3: 91.64371387589951, 4: 115.952808021378,
                5: 103.29507771815399, 6: 91.79531615881979, 7: 104.87429052428499, 8: 107.383247051292,
                9: 105.757813516535, 10: 96.94611612843642, 11: 115.117811684508, 12: 103.898432364114,
                13: 93.787594194582, 14: 77.853001128225, 15: 111.249309181431, 16: 99.5506639098477,
                17: 99.83809736901051, 18: 109.438362106853, 19: 108.212211950981, 20: 105.93901321217501, }, })]
        tbl_3 = rf.inner_join(inputs[1], inputs[0])
        tbl_1 = rf.group_by(tbl_3, group_cols=['x'])
        morpheus = rf.summarise(tbl_1, summaries={
            'z': ('z', 'sum'), })
        output = morpheus
        skeleton = Skeleton([('rf.inner_join', [(- 2), (- 1)]), ('rf.group_by', [1])])
        intermediates = [tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 3,
            '#partial programs with partial evaluation': 6, '#partial programs without partial evaluation': 22,
            'Total time': 0.07, }
        replay_map = {
            'func_seq': ['rf.inner_join', 'rf.group_by', 'rf.summarise'], 'group_by_group_cols': [['x']],
            'summarise_new_col': ['z'], 'summarise_agg': ['sum'], 'summarise_col': ['z'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_39(self):
        b_id = "b_39"
        inputs = [pd.DataFrame({
            'event_id': {
                1: 'A', 2: 'B', 3: 'A', 4: 'A', 5: 'B', }, 'income': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, }, 'location': {
                1: 'PlaceX', 2: 'PlaceY', 3: 'PlaceX', 4: 'PlaceX', 5: 'PlaceY', }, })]
        tbl_5 = rf.group_by(inputs[0], group_cols=['event_id', 'location'])
        tbl_2 = rf.summarise(tbl_5, summaries={
            'mean_inc': ('income', 'mean'), })
        morpheus = rf.inner_join(inputs[0], tbl_2)
        output = morpheus
        skeleton = Skeleton([('rf.group_by', [(- 1)]), ('rf.inner_join', [(- 1), 1])])
        intermediates = [tbl_2, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 10, '#partial programs without partial evaluation': 38,
            'Total time': 0.14, }
        replay_map = {
            'func_seq': ['rf.group_by', 'rf.summarise', 'rf.inner_join'],
            'group_by_group_cols': [['event_id', 'location']], 'summarise_new_col': ['mean_inc'],
            'summarise_agg': ['mean'], 'summarise_col': ['income'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_40(self):
        b_id = "b_40"
        inputs = [pd.DataFrame({
            'ID': {
                1: 'ID1', 2: 'ID1', 3: 'ID1', 4: 'ID1', 5: 'ID1', 6: 'ID1', 7: 'ID1', 8: 'ID1', 9: 'ID2', 10: 'ID2',
                11: 'ID2', 12: 'ID2', 13: 'ID2', 14: 'ID2', 15: 'ID2', 16: 'ID2', 17: 'ID2', 18: 'ID24', 19: 'ID24',
                20: 'ID24', 21: 'ID25', 22: 'ID25', }, 'miRNA': {
                1: 'hsa-miR-512-1', 2: 'hsa-miR-512-2', 3: 'hsa-miR-1323', 4: 'hsa-miR-498', 5: 'hsa-miR-520e',
                6: 'hsa-miR-515-1', 7: 'hsa-miR-519e', 8: 'hsa-miR-520f', 9: 'hsa-miR-495', 10: 'hsa-miR-376c',
                11: 'hsa-miR-376a-2', 12: 'hsa-miR-654', 13: 'hsa-miR-376b', 14: 'hsa-miR-376a-1', 15: 'hsa-miR-300',
                16: 'hsa-miR-1185-1', 17: 'hsa-miR-1185-2', 18: 'hsa-miR-1179', 19: 'hsa-miR-7-2', 20: 'hsa-miR-3677',
                21: 'hsa-miR-940', 22: 'hsa-miR-4717', }, }), pd.DataFrame({
            'miRNA': {
                1: 'hsa-miR-512-1', 2: 'hsa-miR-512-2', 3: 'hsa-miR-1323', 4: 'hsa-miR-498', 5: 'hsa-miR-520e',
                6: 'hsa-miR-515-1', 7: 'hsa-miR-519e', 8: 'hsa-miR-520f', 9: 'hsa-miR-495', 10: 'hsa-miR-376c',
                11: 'hsa-miR-376a-2', 12: 'hsa-miR-654', 13: 'hsa-miR-376b', 14: 'hsa-miR-376a-1', 15: 'hsa-miR-300',
                16: 'hsa-miR-1185-1', 17: 'hsa-miR-1185-2', 18: 'hsa-miR-1179', 19: 'hsa-miR-7-2', 20: 'hsa-miR-3677',
                21: 'hsa-miR-940', 22: 'hsa-miR-4717', }, 'logFC': {
                1: 13.0, 2: 123.0, 3: 53.0, 4: 4.2, 5: 12.0, 6: 1.0, 7: 56.0, 8: 113.0, 9: 11.0, 10: 11.0, 11: 113.0,
                12: 13.0, 13: 123.0, 14: 567.0, 15: 757.0, 16: 6.0, 17: 35.0, 18: 2.0, 19: 2.0, 20: 1.0, 21: 134.0,
                22: 566.0, }, })]
        tbl_3 = rf.inner_join(inputs[1], inputs[0])
        tbl_1 = rf.group_by(tbl_3, group_cols=['ID'])
        morpheus = rf.summarise(tbl_1, summaries={
            'AvgLogFC': ('logFC', 'mean'), })
        output = morpheus
        skeleton = Skeleton([('rf.inner_join', [(- 2), (- 1)]), ('rf.group_by', [1])])
        intermediates = [tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 3,
            '#partial programs with partial evaluation': 6, '#partial programs without partial evaluation': 30,
            'Total time': 0.07, }
        replay_map = {
            'func_seq': ['rf.inner_join', 'rf.group_by', 'rf.summarise'], 'group_by_group_cols': [['ID']],
            'summarise_new_col': ['AvgLogFC'], 'summarise_agg': ['mean'], 'summarise_col': ['logFC'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_41(self):
        b_id = "b_41"
        inputs = [pd.DataFrame({
            'test1_rater1': {
                1: 1, 2: 3, 3: 2, 4: 3, 5: 4, 6: 3, }, 'test2_rater1': {
                1: 1, 2: 3, 3: 3, 4: 2, 5: 3, 6: 1, }, 'test1_rater2': {
                1: 2, 2: 3, 3: 4, 4: 1, 5: 2, 6: 1, }, 'test2_rater2': {
                1: 1, 2: 3, 3: 4, 4: 3, 5: 4, 6: 3, }, 'row': {
                1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 10, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH356', value_name='MORPH355', value_vars=None, id_vars=['row'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH356', into=['test', 'MORPH364'])
        morpheus = rf.spread(tbl_1, columns='MORPH364', values='MORPH355')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 382, '#partial programs without partial evaluation': 532,
            'Total time': 2.1, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH356'],
            'gather_value_name': ['MORPH355'], 'gather_id_vars': [['row']],
            'gather_value_vars': [['test1_rater1', 'test2_rater1', 'test1_rater2', 'test2_rater2']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH356'], 'separate_into': ['test', 'MORPH364'],
            'spread_columns': ['MORPH364'], 'spread_values': ['MORPH355'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_42(self):
        b_id = "b_42"
        inputs = [pd.DataFrame({
            'Exposure': {
                1: 0.01, 2: 0.03, 3: 0.01, 4: 0.03, 5: 0.01, 6: 0.03, }, 'Signal': {
                1: 185.0, 2: 210.2, 3: 218.2, 4: 249.5, 5: 258.4, 6: 292.7, }, 'Noise': {
                1: 0.6744, 2: 0.7683, 3: 0.8356, 4: 0.8609, 5: 0.8988, 6: 0.8326, }, 'ill': {
                1: 1, 2: 4, 3: 1, 4: 4, 5: 1, 6: 4, }, 'ADC': {
                1: 12, 2: 12, 3: 10, 4: 10, 5: 9, 6: 9, }, }), pd.DataFrame({
            'ill': {
                1: 1, 2: 4, 3: 10, }, 'factor': {
                1: 1.0, 2: 3.0, 3: 11.5, }, })]
        tbl_3 = rf.inner_join(inputs[0], inputs[1])
        tbl_1 = rf.mutate(tbl_3, new_col_name='ExposureNew', operation='div', col_args=['Exposure', 'factor'])
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['Exposure', 'factor'])
        output = morpheus
        skeleton = Skeleton([('rf.inner_join', [(- 1), (- 2)]), ('rf.mutate', [1]), ('rf.select', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 6, 'timeout': False, '#sketches without SMT-based deduction': 7,
            '#partial programs with partial evaluation': 287, '#partial programs without partial evaluation': 1603,
            'Total time': 3.17, }
        replay_map = {
            'func_seq': ['rf.inner_join', 'rf.mutate', 'rf.select'], 'mutate_new_col_name': ['ExposureNew'],
            'mutate_operation': ['div'], 'mutate_col_args_normalize': [None],
            'mutate_col_args_div': [['Exposure', 'factor']], 'select_keep_or_remove': [False],
            'select_columns_keep': [None], 'select_columns_remove': [['Exposure', 'factor']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_43(self):
        b_id = "b_43"
        inputs = [pd.DataFrame({
            'Day': {
                1: 0.0, 2: 1.90625, 3: 3.0, 4: 4.0, }, 'HL.Average': {
                1: 8760, 2: 13300, 3: 14500, 4: 16200, }, 'D.Average': {
                1: 8900, 2: 11900, 3: 7320, 4: 9160, }, 'LL.Average': {
                1: 10000, 2: 12100, 3: 12300, 4: 15100, }, 'noHKB.Average': {
                1: 8030, 2: 3860, 3: 1750, 4: 2710, }, 'HL.SD': {
                1: 2337.844, 2: 1016.291, 3: 2945.098, 4: 1006.893, }, 'D.SD': {
                1: 924.2742, 2: 2308.2661, 3: 1308.0389, 4: 514.2177, }, 'LL.SD': {
                1: 1120.785, 2: 3581.763, 3: 4338.897, 4: 4362.2609999999995, }, 'noHKB.SD': {
                1: 1592.646, 2: 1031.057, 3: 1793.5829999999999, 4: 2691.6479999999997, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH492', value_name='MORPH491', value_vars=None, id_vars=['Day'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH492', into=['Group', 'MORPH548'])
        morpheus = rf.spread(tbl_1, columns='MORPH548', values='MORPH491')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 550, '#partial programs without partial evaluation': 1094,
            'Total time': 3.43, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH492'],
            'gather_value_name': ['MORPH491'], 'gather_id_vars': [['Day']], 'gather_value_vars': [
                ['HL.Average', 'D.Average', 'LL.Average', 'noHKB.Average', 'HL.SD', 'D.SD', 'LL.SD', 'noHKB.SD']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH492'], 'separate_into': ['Group', 'MORPH548'],
            'spread_columns': ['MORPH548'], 'spread_values': ['MORPH491'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_44(self):
        b_id = "b_44"
        inputs = [pd.DataFrame({
            'sbj': {
                1: 'A', 2: 'B', 3: 'C', 4: 'D', }, 'f1.avg': {
                1: 10, 2: 12, 3: 20, 4: 22, }, 'f1.sd': {
                1: 6, 2: 5, 3: 7, 4: 8, }, 'f2.avg': {
                1: 50, 2: 70, 3: 20, 4: 22, }, 'f2.sd': {
                1: 10, 2: 11, 3: 8, 4: 9, }, 'blabla': {
                1: 'bA', 2: 'bB', 3: 'bC', 4: 'bD', }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH672', value_name='MORPH671', value_vars=None,
                          id_vars=['sbj', 'blabla'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH672', into=['var', 'MORPH760'])
        morpheus = rf.spread(tbl_1, columns='MORPH760', values='MORPH671')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 719, '#partial programs without partial evaluation': 1235,
            'Total time': 10.2, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH672'],
            'gather_value_name': ['MORPH671'], 'gather_id_vars': [['sbj', 'blabla']],
            'gather_value_vars': [['f1.avg', 'f1.sd', 'f2.avg', 'f2.sd']], 'gather_df': ['inputs[0]'],
            'separate_split_col': ['MORPH672'], 'separate_into': ['var', 'MORPH760'], 'spread_columns': ['MORPH760'],
            'spread_values': ['MORPH671'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_45(self):
        b_id = "b_45"
        inputs = [pd.DataFrame({
            'V1': {
                1: 'a', 2: 'a', 3: 'a', 4: 'x', 5: 'x', }, 'V2': {
                1: 'b', 2: 'b', 3: 'b', 4: 'y', 5: 'y', }, 'V3': {
                1: 'a', 2: 'EMP', 3: 'EMP', 4: 'h', 5: 'EMP', }, 'V4': {
                1: 'EMP', 2: 'c', 3: 'EMP', 4: 'EMP', 5: 'k', }, 'V5': {
                1: 'EMP', 2: 'EMP', 3: 'd', 4: 'EMP', 5: 'e', }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH1029', value_name='MORPH1028', value_vars=None,
                          id_vars=['V1', 'V2'])
        tbl_1 = rf.filter_(tbl_3, filter_expr="MORPH1028 != 'EMP'")
        morpheus = rf.spread(tbl_1, columns='MORPH1029', values='MORPH1028')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.filter', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 46, 'timeout': False, '#sketches without SMT-based deduction': 129,
            '#partial programs with partial evaluation': 4331, '#partial programs without partial evaluation': 9229,
            'Total time': 19.15, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.filter', 'rf.spread'], 'gather_var_name': ['MORPH1029'],
            'gather_value_name': ['MORPH1028'], 'gather_id_vars': [['V1', 'V2']],
            'gather_value_vars': [['V3', 'V4', 'V5']], 'gather_df': ['inputs[0]'],
            'filter_mode': ['equality-inequality'], 'filter_column_eq': ['MORPH1028'], 'filter_column_relop': [],
            'filter_eq_op': ['!='], 'filter_relop': ['>'], 'filter_value_eq': ['EMP'], 'filter_value_relop': [],
            'spread_columns': ['MORPH1029'], 'spread_values': ['MORPH1028'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_46(self):
        b_id = "b_46"
        inputs = [pd.DataFrame({
            'name': {
                1: 'A', 2: 'A', 3: 'B', 4: 'B', }, 'group': {
                1: 'g1', 2: 'g2', 3: 'g1', 4: 'g2', }, 'V1': {
                1: 10, 2: 40, 3: 20, 4: 30, }, 'V2': {
                1: 6, 2: 3, 3: 1, 4: 7, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH72', value_name='MORPH71', value_vars=['V1', 'V2'], id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH79', cols=['MORPH72', 'group'])
        morpheus = rf.spread(tbl_1, columns='MORPH79', values='MORPH71')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 20, '#partial programs without partial evaluation': 90,
            'Total time': 0.16, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH72'],
            'gather_value_name': ['MORPH71'], 'gather_id_vars': [['name', 'group']],
            'gather_value_vars': [['V1', 'V2']], 'gather_df': ['inputs[0]'], 'unite_new_col_name': ['MORPH79'],
            'unite_cols': [['MORPH72', 'group']], 'spread_columns': ['MORPH79'], 'spread_values': ['MORPH71'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_47(self):
        b_id = "b_47"
        inputs = [pd.DataFrame({
            'id': {
                1: 'user1', 2: 'user2', 3: 'user3', }, 'age_1': {
                1: 20, 2: 25, 3: 32, }, 'age_2': {
                1: 21, 2: 34, 3: 33, }, 'favCol_1': {
                1: 'blue', 2: 'red', 3: 'blue', }, 'favCol_2': {
                1: 'red', 2: 'blue', 3: 'red', }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH2', value_name='MORPH1', value_vars=None, id_vars=['id'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH2', into=['MORPH33', 'panel'])
        morpheus = rf.spread(tbl_1, columns='MORPH33', values='MORPH1')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 1, 'timeout': False, '#sketches without SMT-based deduction': 10,
            '#partial programs with partial evaluation': 9, '#partial programs without partial evaluation': 29,
            'Total time': 0.13, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH2'],
            'gather_value_name': ['MORPH1'], 'gather_id_vars': [['id']],
            'gather_value_vars': [['age_1', 'age_2', 'favCol_1', 'favCol_2']], 'gather_df': ['inputs[0]'],
            'separate_split_col': ['MORPH2'], 'separate_into': ['MORPH33', 'panel'], 'spread_columns': ['MORPH33'],
            'spread_values': ['MORPH1'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_48(self):
        b_id = "b_48"
        inputs = [pd.DataFrame({
            'day': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, }, 'site': {
                1: 'a', 2: 'a', 3: 'a', 4: 'a', 5: 'a', 6: 'a', 7: 'b', 8: 'b', 9: 'b', 10: 'b', 11: 'b', 12: 'b', },
            'value.1': {
                1: 1, 2: 2, 3: 5, 4: 7, 5: 5, 6: 3, 7: 9, 8: 4, 9: 2, 10: 8, 11: 1, 12: 8, }, 'value.2': {
                1: 5, 2: 4, 3: 7, 4: 6, 5: 2, 6: 4, 7: 6, 8: 9, 9: 4, 10: 2, 11: 5, 12: 6, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH102', value_name='MORPH101', value_vars=['value.1', 'value.2'],
                          id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH108', cols=['site', 'MORPH102'])
        morpheus = rf.spread(tbl_1, columns='MORPH108', values='MORPH101')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 24, '#partial programs without partial evaluation': 126,
            'Total time': 0.27, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH102'],
            'gather_value_name': ['MORPH101'], 'gather_id_vars': [['day', 'site']],
            'gather_value_vars': [['value.1', 'value.2']], 'gather_df': ['inputs[0]'],
            'unite_new_col_name': ['MORPH108'], 'unite_cols': [['site', 'MORPH102']], 'spread_columns': ['MORPH108'],
            'spread_values': ['MORPH101'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_49(self):
        b_id = "b_49"
        inputs = [pd.DataFrame({
            'Scenario': {
                1: 'base', 2: 'stress', 3: 'extreme', }, 'x_min': {
                1: (- 3.0), 2: (- 2.0), 3: (- 2.5), }, 'x_mean': {
                1: 0.0, 2: 0.25, 3: 1.0, }, 'x_max': {
                1: 2, 2: 1, 3: 3, }, 'y_min': {
                1: (- 1.5), 2: (- 2.0), 3: (- 3.0), }, 'y_mean': {
                1: 1, 2: 2, 3: 3, }, 'y_max': {
                1: 5.0, 2: 3.0, 3: 3.5, }, 'z_min': {
                1: 0, 2: 1, 3: 3, }, 'z_mean': {
                1: 0.25, 2: 2.0, 3: 5.0, }, 'z_max': {
                1: 2, 2: 4, 3: 7, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH118', value_name='MORPH117', value_vars=None, id_vars=['Scenario'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH118', into=['varNew', 'MORPH194'])
        morpheus = rf.spread(tbl_1, columns='MORPH194', values='MORPH117')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 127, '#partial programs without partial evaluation': 401,
            'Total time': 1.0, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH118'],
            'gather_value_name': ['MORPH117'], 'gather_id_vars': [['Scenario']],
            'gather_value_vars': [['x_min', 'x_mean', 'x_max', 'y_min', 'y_mean', 'y_max', 'z_min', 'z_mean', 'z_max']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH118'], 'separate_into': ['varNew', 'MORPH194'],
            'spread_columns': ['MORPH194'], 'spread_values': ['MORPH117'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_50(self):
        b_id = "b_50"
        inputs = [pd.DataFrame({
            'MemberID': {
                1: 123, 2: 123, 3: 234, 4: 234, }, 'years': {
                1: 'Y1', 2: 'Y2', 3: 'Y1', 4: 'Y2', }, 'a': {
                1: 1, 2: 1, 3: 1, 4: 1, }, 'b': {
                1: 0, 2: 0, 3: 0, 4: 0, }, 'c': {
                1: 0, 2: 0, 3: 0, 4: 1, }, 'd': {
                1: 0, 2: 1, 3: 0, 4: 1, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH2', value_name='MORPH1', value_vars=None,
                          id_vars=['MemberID', 'years'])
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH598', cols=['years', 'MORPH2'])
        morpheus = rf.spread(tbl_1, columns='MORPH598', values='MORPH1')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 1, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 230, '#partial programs without partial evaluation': 649,
            'Total time': 1.52, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH2'],
            'gather_value_name': ['MORPH1'], 'gather_id_vars': [['MemberID', 'years']],
            'gather_value_vars': [['a', 'b', 'c', 'd']], 'gather_df': ['inputs[0]'], 'unite_new_col_name': ['MORPH598'],
            'unite_cols': [['years', 'MORPH2']], 'spread_columns': ['MORPH598'], 'spread_values': ['MORPH1'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_51(self):
        b_id = "b_51"
        inputs = [pd.DataFrame({
            'Geotype': {
                1: 'A', 2: 'A', 3: 'A', 4: 'B', 5: 'B', 6: 'B', }, 'Strategy': {
                1: 'Demand', 2: 'Strategy_1', 3: 'Strategy_2', 4: 'Demand', 5: 'Strategy_1', 6: 'Strategy_2', },
            'Year.1': {
                1: 1, 2: 2, 3: 3, 4: 8, 5: 9, 6: 10, }, 'Year.2': {
                1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, }, })]
        tbl_7 = rf.gather(inputs[0], var_name='key', value_name='MORPH16493', value_vars=['Year.1', 'Year.2'],
                          id_vars=None)
        tbl_3 = rf.filter_(tbl_7, filter_expr="Strategy != 'Demand'")
        tbl_1 = rf.group_by(tbl_3, group_cols=['Geotype', 'key'])
        morpheus = rf.summarise(tbl_1, summaries={
            'sumVal': ('MORPH16493', 'sum'), })
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.filter', [1]), ('rf.group_by', [2])])
        intermediates = [tbl_7, tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 10, 'timeout': False, '#sketches without SMT-based deduction': 13,
            '#partial programs with partial evaluation': 2492, '#partial programs without partial evaluation': 22109,
            'Total time': 21.49, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.filter', 'rf.group_by', 'rf.summarise'], 'gather_var_name': ['key'],
            'gather_value_name': ['MORPH16493'], 'gather_id_vars': [['Geotype', 'Strategy']],
            'gather_value_vars': [['Year.1', 'Year.2']], 'gather_df': ['inputs[0]'],
            'filter_mode': ['equality-inequality'], 'filter_column_eq': ['Strategy'], 'filter_column_relop': [],
            'filter_eq_op': ['!='], 'filter_relop': ['>'], 'filter_value_eq': ['Demand'], 'filter_value_relop': [],
            'group_by_group_cols': [['Geotype', 'key']], 'summarise_new_col': ['sumVal'], 'summarise_agg': ['sum'],
            'summarise_col': ['MORPH16493'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_52(self):
        b_id = "b_52"
        inputs = [pd.DataFrame({
            'Geotype': {
                1: 'A', 2: 'A', 3: 'A', 4: 'B', 5: 'B', 6: 'B', }, 'Strategy': {
                1: 'Demand', 2: 'Strategy_1', 3: 'Strategy_2', 4: 'Demand', 5: 'Strategy_1', 6: 'Strategy_2', },
            'Year.1': {
                1: 1, 2: 2, 3: 3, 4: 8, 5: 9, 6: 10, }, 'Year.2': {
                1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, }, })]
        tbl_15 = rf.gather(inputs[0], var_name='MORPH49865', value_name='MORPH49864', value_vars=['Year.1', 'Year.2'],
                           id_vars=None)
        tbl_7 = rf.filter_(tbl_15, filter_expr="Strategy != 'Demand'")
        tbl_3 = rf.group_by(tbl_7, group_cols=['Geotype', 'MORPH49865'])
        tbl_1 = rf.summarise(tbl_3, summaries={
            'MORPH51377': ('MORPH49864', 'sum'), })
        morpheus = rf.spread(tbl_1, columns='MORPH49865', values='MORPH51377')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.filter', [1]), ('rf.group_by', [2]), ('rf.spread', [3])])
        intermediates = [tbl_15, tbl_7, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 59, 'timeout': False, '#sketches without SMT-based deduction': 142,
            '#partial programs with partial evaluation': 30512, '#partial programs without partial evaluation': 155920,
            'Total time': 291.96, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.filter', 'rf.group_by', 'rf.summarise', 'rf.spread'],
            'gather_var_name': ['MORPH49865'], 'gather_value_name': ['MORPH49864'],
            'gather_id_vars': [['Geotype', 'Strategy']], 'gather_value_vars': [['Year.1', 'Year.2']],
            'gather_df': ['inputs[0]'], 'filter_mode': ['equality-inequality'], 'filter_column_eq': ['Strategy'],
            'filter_column_relop': [], 'filter_eq_op': ['!='], 'filter_relop': ['>'], 'filter_value_eq': ['Demand'],
            'filter_value_relop': [], 'group_by_group_cols': [['Geotype', 'MORPH49865']],
            'summarise_new_col': ['MORPH51377'], 'summarise_agg': ['sum'], 'summarise_col': ['MORPH49864'],
            'spread_columns': ['MORPH49865'], 'spread_values': ['MORPH51377'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_53(self):
        b_id = "b_53"
        inputs = [pd.DataFrame({
            'posture': {
                1: 'cycling', 2: 'standing', 3: 'sitting', 4: 'walking', 5: 'cycling', 6: 'standing', 7: 'sitting',
                8: 'walking', }, 'code': {
                1: 'A03', 2: 'A03', 3: 'A03', 4: 'A03', 5: 'B01', 6: 'B01', 7: 'B01', 8: 'B01', }, 'HR': {
                1: 102, 2: 99, 3: 98, 4: 97, 5: 111, 6: 100, 7: 78, 8: 99, }, 'EE': {
                1: 100, 2: 99, 3: 67, 4: 78, 5: 76, 6: 88, 7: 34, 8: 99, }, 'a': {
                1: 3, 2: 4, 3: 5, 4: 3, 5: 5, 6: 4, 7: 4, 8: 2, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH2', value_name='MORPH1', value_vars=None,
                          id_vars=['posture', 'code'])
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH302', cols=['posture', 'MORPH2'])
        morpheus = rf.spread(tbl_1, columns='MORPH302', values='MORPH1')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 1, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 198, '#partial programs without partial evaluation': 374,
            'Total time': 2.31, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH2'],
            'gather_value_name': ['MORPH1'], 'gather_id_vars': [['posture', 'code']],
            'gather_value_vars': [['HR', 'EE', 'a']], 'gather_df': ['inputs[0]'], 'unite_new_col_name': ['MORPH302'],
            'unite_cols': [['posture', 'MORPH2']], 'spread_columns': ['MORPH302'], 'spread_values': ['MORPH1'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_54(self):
        b_id = "b_54"
        inputs = [pd.DataFrame({
            'sample_ID': {
                1: 382870, 2: 487405, 3: 487405, 4: 487405, 5: 382899, 6: 382899, 7: 382899, 8: 382899, 9: 382899, },
            'site': {
                1: 'site_1', 2: 'site_2', 3: 'site_2', 4: 'site_2', 5: 'site_1', 6: 'site_1', 7: 'site_2', 8: 'site_1',
                9: 'site_2', }, 'species': {
                1: 'Species_B', 2: 'Species_A', 3: 'Species_B', 4: 'Species_A', 5: 'Species_A', 6: 'Species_C',
                7: 'Species_C', 8: 'Species_D', 9: 'Species_D', }, 'TOT': {
                1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 10, 8: 1, 9: 20, }, 'inf_status': {
                1: 'positive', 2: 'positive', 3: 'positive', 4: 'positive', 5: 'positive', 6: 'positive', 7: 'positive',
                8: 'positive', 9: 'positive', }, })]
        tbl_7 = rf.unite(inputs[0], new_col_name='MORPH42009', cols=['species', 'inf_status'])
        tbl_3 = rf.group_by(tbl_7, group_cols=['site', 'MORPH42009'])
        tbl_1 = rf.summarise(tbl_3, summaries={
            'MORPH42025': ('TOT', 'sum'), })
        morpheus = rf.spread(tbl_1, columns='MORPH42009', values='MORPH42025')
        output = morpheus
        skeleton = Skeleton([('rf.unite', [(- 1)]), ('rf.group_by', [1]), ('rf.spread', [2])])
        intermediates = [tbl_7, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 120, 'timeout': False, '#sketches without SMT-based deduction': 479,
            '#partial programs with partial evaluation': 29097, '#partial programs without partial evaluation': 77427,
            'Total time': 138.0, }
        replay_map = {
            'func_seq': ['rf.unite', 'rf.group_by', 'rf.summarise', 'rf.spread'], 'unite_new_col_name': ['MORPH42009'],
            'unite_cols': [['species', 'inf_status']], 'group_by_group_cols': [['site', 'MORPH42009']],
            'summarise_new_col': ['MORPH42025'], 'summarise_agg': ['sum'], 'summarise_col': ['TOT'],
            'spread_columns': ['MORPH42009'], 'spread_values': ['MORPH42025'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_55(self):
        b_id = "b_55"
        inputs = [pd.DataFrame({
            'ID': {
                1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', }, 'c_Al': {
                1: 0, 2: 1, 3: 0, 4: 0, 5: 1, }, 'c_D': {
                1: 0, 2: 0, 3: 0, 4: 1, 5: 1, }, 'c_Hy': {
                1: 1, 2: 1, 3: 1, 4: 0, 5: 0, }, 'occ': {
                1: 2581, 2: 1917, 3: 2708, 4: 2751, 5: 1522, }, })]
        tbl_7 = rf.gather(inputs[0], var_name='Var', value_name='MORPH116131', value_vars=None, id_vars=['ID', 'occ'])
        tbl_3 = rf.group_by(tbl_7, group_cols=['Var', 'MORPH116131'])
        tbl_1 = rf.summarise(tbl_3, summaries={
            'MORPH117459': ('occ', 'mean'), })
        morpheus = rf.spread(tbl_1, columns='MORPH116131', values='MORPH117459')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.group_by', [1]), ('rf.spread', [2])])
        intermediates = [tbl_7, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 18, 'timeout': False, '#sketches without SMT-based deduction': 26,
            '#partial programs with partial evaluation': 11043, '#partial programs without partial evaluation': 128168,
            'Total time': 131.85, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.group_by', 'rf.summarise', 'rf.spread'], 'gather_var_name': ['Var'],
            'gather_value_name': ['MORPH116131'], 'gather_id_vars': [['ID', 'occ']],
            'gather_value_vars': [['c_Al', 'c_D', 'c_Hy']], 'gather_df': ['inputs[0]'],
            'group_by_group_cols': [['Var', 'MORPH116131']], 'summarise_new_col': ['MORPH117459'],
            'summarise_agg': ['mean'], 'summarise_col': ['occ'], 'spread_columns': ['MORPH116131'],
            'spread_values': ['MORPH117459'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_56(self):
        b_id = "b_56"
        inputs = [pd.DataFrame({
            'sample': {
                1: 'AA', 2: 'BB', }, 'BMI': {
                1: 18.9, 2: 27.1, }, 'var1_LRR': {
                1: 0.27, 2: 0.23, }, 'var1_BAF': {
                1: 0.99, 2: 1.0, }, 'var2_LRR': {
                1: 0.18, 2: 0.13, }, 'var2_BAF': {
                1: 0.99, 2: 0.99, }, 'var3_LRR': {
                1: 0.11, 2: 0.17, }, 'var3_BAF': {
                1: 1, 2: 1, }, 'var200_LRR': {
                1: 0.2, 2: 0.23, }, 'var200_BAF': {
                1: 0.99, 2: 0.99, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH118', value_name='MORPH117', value_vars=None,
                          id_vars=['sample', 'BMI'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH118', into=['varNew', 'MORPH186'])
        morpheus = rf.spread(tbl_1, columns='MORPH186', values='MORPH117')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 121, '#partial programs without partial evaluation': 415,
            'Total time': 0.94, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH118'],
            'gather_value_name': ['MORPH117'], 'gather_id_vars': [['sample', 'BMI']], 'gather_value_vars': [
                ['var1_LRR', 'var1_BAF', 'var2_LRR', 'var2_BAF', 'var3_LRR', 'var3_BAF', 'var200_LRR', 'var200_BAF']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH118'], 'separate_into': ['varNew', 'MORPH186'],
            'spread_columns': ['MORPH186'], 'spread_values': ['MORPH117'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_57(self):
        b_id = "b_57"
        inputs = [pd.DataFrame({
            'Test': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, }, 'temperature_sensor1': {
                1: 30.1008390999259, 2: 37.5587331567327, 3: 27.698755723572102, 4: 23.2898450366646,
                5: 20.451697108168002, 6: 23.7257003357801, }, 'temperature_sensor2': {
                1: 32.930285705825, 2: 21.3353314923698, 3: 24.4626561565937, 4: 33.7237270019053, 5: 28.4518640346818,
                6: 27.0358118265204, }, 'pressure_sensor1': {
                1: 10.8850509116509, 2: 7.2929949615878495, 3: 11.250254096498999, 4: 11.893415216492699,
                5: 11.8588439703521, 6: 11.692289024302198, }, 'pressure_sensor2': {
                1: 9.14442430326963, 2: 10.5098461719334, 3: 10.9207007625597, 4: 9.91898176804089,
                5: 10.226134142154699, 6: 9.94034018885316, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH356', value_name='MORPH355', value_vars=None, id_vars=['Test'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH356', into=['MORPH387', 'sensor'])
        morpheus = rf.spread(tbl_1, columns='MORPH387', values='MORPH355')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 392, '#partial programs without partial evaluation': 656,
            'Total time': 3.55, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH356'],
            'gather_value_name': ['MORPH355'], 'gather_id_vars': [['Test']], 'gather_value_vars': [
                ['temperature_sensor1', 'temperature_sensor2', 'pressure_sensor1', 'pressure_sensor2']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH356'], 'separate_into': ['MORPH387', 'sensor'],
            'spread_columns': ['MORPH387'], 'spread_values': ['MORPH355'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_58(self):
        b_id = "b_58"
        inputs = [pd.DataFrame({
            'ID': {
                1: 1, 2: 2, }, 'p_2012': {
                1: 160, 2: 163, }, 'p_2010': {
                1: 162, 2: 164, }, 'p_2008': {
                1: 163, 2: 164, }, 'p_2006': {
                1: 165, 2: 163, }, 'c_2012': {
                1: 37.3, 2: 2.6, }, 'c_2010': {
                1: 37.3, 2: 2.6, }, 'c_2008': {
                1: 37.1, 2: 2.6, }, 'c_2006': {
                1: 37.1, 2: 2.6, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH2', value_name='MORPH1', value_vars=None, id_vars=['ID'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH2', into=['MORPH57', 'year'])
        morpheus = rf.spread(tbl_1, columns='MORPH57', values='MORPH1')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 1, 'timeout': False, '#sketches without SMT-based deduction': 10,
            '#partial programs with partial evaluation': 14, '#partial programs without partial evaluation': 91,
            'Total time': 0.23, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH2'],
            'gather_value_name': ['MORPH1'], 'gather_id_vars': [['ID']],
            'gather_value_vars': [['p_2012', 'p_2010', 'p_2008', 'p_2006', 'c_2012', 'c_2010', 'c_2008', 'c_2006']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH2'], 'separate_into': ['MORPH57', 'year'],
            'spread_columns': ['MORPH57'], 'spread_values': ['MORPH1'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_59(self):
        b_id = "b_59"
        inputs = [pd.DataFrame({
            'V2': {
                1: 'CCRG10', 2: 'CCRG10', 3: 'CCRG20', 4: 'CCRG20', 5: 'CCRG30', 6: 'CCRG30', 7: 'CCRG40', 8: 'CCRG40',
                9: 'CCRG50', 10: 'CCRG50', }, 'V3': {
                1: 'BranchDBMS', 2: 'CacheDBMS', 3: 'BranchDBMS', 4: 'CacheDBMS', 5: 'BranchDBMS', 6: 'CacheDBMS',
                7: 'BranchDBMS', 8: 'CacheDBMS', 9: 'BranchDBMS', 10: 'CacheDBMS', }, 'V4': {
                1: 2, 2: 3, 3: 7, 4: 2, 5: 15, 6: 5, 7: 62, 8: 7, 9: 58, 10: 11, }, })]
        tbl_3 = rf.spread(inputs[0], columns='V3', values='V4')
        tbl_1 = rf.mutate(tbl_3, new_col_name='MORPH712', operation='div', col_args=['BranchDBMS', 'CacheDBMS'])
        morpheus = rf.gather(tbl_1, var_name='key', value_name='value', value_vars=None, id_vars=['V2'])
        output = morpheus
        skeleton = Skeleton([('rf.spread', [(- 1)]), ('rf.mutate', [1]), ('rf.gather', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 70, 'timeout': False, '#sketches without SMT-based deduction': 338,
            '#partial programs with partial evaluation': 341, '#partial programs without partial evaluation': 1782,
            'Total time': 4.29, }
        replay_map = {
            'func_seq': ['rf.spread', 'rf.mutate', 'rf.gather'], 'spread_columns': ['V3'], 'spread_values': ['V4'],
            'mutate_new_col_name': ['MORPH712'], 'mutate_operation': ['div'], 'mutate_col_args_normalize': [None],
            'mutate_col_args_div': [['BranchDBMS', 'CacheDBMS']], 'gather_var_name': ['key'],
            'gather_value_name': ['value'], 'gather_id_vars': [['V2']],
            'gather_value_vars': [['BranchDBMS', 'CacheDBMS', 'MORPH712']], 'gather_df': ['tbl_1'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_60(self):
        b_id = "b_60"
        inputs = [pd.DataFrame({
            'Market': {
                1: 'market_1', 2: 'market_1', 3: 'market_2', 4: 'market_2', }, 'Variables': {
                1: 'var_1', 2: 'var_2', 3: 'var_1', 4: 'var_2', }, 'median': {
                1: 2.78, 2: 3.21, 3: 2.95, 4: 2.11, }, 'lower.limit': {
                1: 2.71, 2: 2.96, 3: 2.79, 4: 1.91, }, 'upper.limit': {
                1: 2.72, 2: 3.44, 3: 3.11, 4: 2.3, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH2', value_name='MORPH1', value_vars=None,
                          id_vars=['Market', 'Variables'])
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH68', cols=['Variables', 'MORPH2'])
        morpheus = rf.spread(tbl_1, columns='MORPH68', values='MORPH1')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 1, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 72, '#partial programs without partial evaluation': 137,
            'Total time': 1.06, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH2'],
            'gather_value_name': ['MORPH1'], 'gather_id_vars': [['Market', 'Variables']],
            'gather_value_vars': [['median', 'lower.limit', 'upper.limit']], 'gather_df': ['inputs[0]'],
            'unite_new_col_name': ['MORPH68'], 'unite_cols': [['Variables', 'MORPH2']], 'spread_columns': ['MORPH68'],
            'spread_values': ['MORPH1'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_61(self):
        b_id = "b_61"
        inputs = [pd.DataFrame({
            'year': {
                1: 2006, 2: 2009, 3: 2013, 4: 2020, 5: 2004, }, 'roleInDebate': {
                1: 'x', 2: 'y', 3: 'r', 4: 'q', 5: 'b', }, 'Clarity_1': {
                1: 3, 2: 2, 3: 7, 4: 4, 5: 8, }, 'Effort_1': {
                1: 5, 2: 8, 3: 10, 4: 4, 5: 8, }, 'Clarity_2': {
                1: 10, 2: 3, 3: 7, 4: 2, 5: 3, }, 'Effort_2': {
                1: 4, 2: 1, 3: 4, 4: 9, 5: 4, }, 'Clarity_3': {
                1: 5, 2: 6, 3: 5, 4: 2, 5: 9, }, 'Effort_3': {
                1: 7, 2: 8, 3: 2, 4: 8, 5: 5, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH390', value_name='MORPH389', value_vars=None,
                          id_vars=['year', 'roleInDebate'])
        tbl_1 = rf.separate(tbl_3, split_col='MORPH390', into=['MORPH613', 'person'])
        morpheus = rf.spread(tbl_1, columns='MORPH613', values='MORPH389')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 11,
            '#partial programs with partial evaluation': 454, '#partial programs without partial evaluation': 698,
            'Total time': 1.73, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread'], 'gather_var_name': ['MORPH390'],
            'gather_value_name': ['MORPH389'], 'gather_id_vars': [['year', 'roleInDebate']],
            'gather_value_vars': [['Clarity_1', 'Effort_1', 'Clarity_2', 'Effort_2', 'Clarity_3', 'Effort_3']],
            'gather_df': ['inputs[0]'], 'separate_split_col': ['MORPH390'], 'separate_into': ['MORPH613', 'person'],
            'spread_columns': ['MORPH613'], 'spread_values': ['MORPH389'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_62(self):
        b_id = "b_62"
        inputs = [pd.DataFrame({
            'score': {
                1: 10, 2: 8, 3: 9, 4: 1, 5: 5, 6: 8, 7: 2, 8: 8, 9: 5, 10: 6, 11: 9, 12: 4, 13: 7, 14: 9, 15: 9, },
            'group': {
                1: 'a1', 2: 'a1', 3: 'a1', 4: 'a1', 5: 'a1', 6: 'a2', 7: 'a2', 8: 'a2', 9: 'a2', 10: 'a2', 11: 'a3',
                12: 'a3', 13: 'a3', 14: 'a3', 15: 'a3', }, 'category': {
                1: 'big', 2: 'big', 3: 'big', 4: 'big', 5: 'small', 6: 'big', 7: 'big', 8: 'big', 9: 'big', 10: 'small',
                11: 'big', 12: 'big', 13: 'big', 14: 'big', 15: 'small', }, })]
        tbl_3 = rf.filter_(inputs[0], filter_expr="category == 'big'")
        tbl_1 = rf.group_by(tbl_3, group_cols=['group'])
        morpheus = rf.summarise(tbl_1, summaries={
            'mean': ('score', 'mean'), })
        output = morpheus
        skeleton = Skeleton([('rf.filter', [(- 1)]), ('rf.group_by', [1])])
        intermediates = [tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 4, 'timeout': False, '#sketches without SMT-based deduction': 5,
            '#partial programs with partial evaluation': 55, '#partial programs without partial evaluation': 227,
            'Total time': 0.26, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.group_by', 'rf.summarise'], 'filter_mode': ['equality-inequality'],
            'filter_column_eq': ['category'], 'filter_column_relop': [], 'filter_eq_op': ['=='], 'filter_relop': ['>'],
            'filter_value_eq': ['big'], 'filter_value_relop': [], 'group_by_group_cols': [['group']],
            'summarise_new_col': ['mean'], 'summarise_agg': ['mean'], 'summarise_col': ['score'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_63(self):
        b_id = "b_63"
        inputs = [pd.DataFrame({
            'Category': {
                1: 'Cat1', 2: 'Cat2', 3: 'Cat2', 4: 'Cat2', 5: 'Cat1', 6: 'Cat2', 7: 'Cat1', 8: 'Cat1', 9: 'Cat2', },
            'qs': {
                1: 'Q1.a-Some-Text', 2: 'Q1.b-Some-Text', 3: 'Q1.a-Some-Text', 4: 'Q1.a-Some-Text', 5: 'Q1.b-Some-Text',
                6: 'Q1.a-Some-Text', 7: 'Q1.b-Some-Text', 8: 'Q1.a-Some-Text', 9: 'Q1.b-Some-Text', }, 'Ans': {
                1: 1, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 1, }, })]
        tbl_15 = rf.group_by(inputs[0], group_cols=['qs', 'Ans'])
        tbl_7 = rf.summarise(tbl_15, summaries={
            'MORPH1836': (None, 'count'), })
        tbl_3 = rf.mutate(tbl_7, new_col_name='MORPH1839', operation='normalize', col_args='MORPH1836')
        tbl_1 = rf.select(tbl_3, columns_keep=['qs', 'Ans', 'MORPH1839'], columns_remove=None)
        morpheus = rf.spread(tbl_1, columns='Ans', values='MORPH1839')
        output = morpheus
        skeleton = Skeleton([('rf.group_by', [(- 1)]), ('rf.mutate', [1]), ('rf.select', [2]), ('rf.spread', [3])])
        intermediates = [tbl_7, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 8, 'timeout': False, '#sketches without SMT-based deduction': 26,
            '#partial programs with partial evaluation': 1584, '#partial programs without partial evaluation': 8961,
            'Total time': 9.21, }
        replay_map = {
            'func_seq': ['rf.group_by', 'rf.summarise', 'rf.mutate', 'rf.select', 'rf.spread'],
            'group_by_group_cols': [['qs', 'Ans']], 'summarise_new_col': ['MORPH1836'], 'summarise_agg': ['count'],
            'summarise_col': [], 'mutate_new_col_name': ['MORPH1839'], 'mutate_operation': ['normalize'],
            'mutate_col_args_normalize': ['MORPH1836'], 'mutate_col_args_div': [None], 'select_keep_or_remove': [True],
            'select_columns_keep': [['qs', 'Ans', 'MORPH1839']], 'select_columns_remove': [None],
            'spread_columns': ['Ans'], 'spread_values': ['MORPH1839'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_64(self):
        b_id = "b_64"
        inputs = [pd.DataFrame({
            'ST': {
                1: 'PA', 2: 'VA', 3: 'RI', 4: 'NJ', 5: 'PA', 6: 'VT', 7: 'NY', 8: 'PA', 9: 'NC', }, 'Rfips': {
                1: 42107, 2: 51760, 3: 44001, 4: 34001, 5: 51061, 6: 50023, 7: 36029, 8: 42101, 9: 37019, }, 'zip': {
                1: 17972, 2: 23226, 3: 2806, 4: 8330, 5: 20118, 6: 5681, 7: 14072, 8: 19115, 9: 28451, }, 'Year': {
                1: 2010, 2: 2005, 3: 2010, 4: 2008, 5: 2007, 6: 2006, 7: 2005, 8: 2008, 9: 2009, }, 'dist_km': {
                1: 0.0, 2: 42.46894, 3: 28.112340000000003, 4: 36.8547, 5: 0.0, 6: 49.72765, 7: 0.0,
                8: 30.193720000000003, 9: 0.0, }, })]
        tbl_7 = rf.group_by(inputs[0], group_cols=['ST', 'dist_km'])
        tbl_3 = rf.summarise(tbl_7, summaries={
            'total': (None, 'count'), })
        tbl_1 = rf.filter_(tbl_3, filter_expr='dist_km < 28.11234')
        morpheus = rf.select(tbl_1, columns_keep=['ST', 'total'], columns_remove=None)
        output = morpheus
        skeleton = Skeleton([('rf.group_by', [(- 1)]), ('rf.filter', [1]), ('rf.select', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 3,
            '#partial programs with partial evaluation': 40, '#partial programs without partial evaluation': 529,
            'Total time': 0.59, }
        replay_map = {
            'func_seq': ['rf.group_by', 'rf.summarise', 'rf.filter', 'rf.select'],
            'group_by_group_cols': [['ST', 'dist_km']], 'summarise_new_col': ['total'], 'summarise_agg': ['count'],
            'summarise_col': [], 'filter_mode': ['relop'], 'filter_column_eq': [], 'filter_column_relop': ['dist_km'],
            'filter_eq_op': ['!='], 'filter_relop': ['<'], 'filter_value_eq': [], 'filter_value_relop': [28.11234],
            'select_keep_or_remove': [True], 'select_columns_keep': [['ST', 'total']],
            'select_columns_remove': [None], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_65(self):
        b_id = "b_65"
        inputs = [pd.DataFrame({
            'ID': {
                1: 'A', 2: 'B', 3: 'C', }, 'MGW.one': {
                1: 10.0, 2: (- 13.29), 3: (- 6.95), }, 'MGW.two': {
                1: 19, 2: 13, 3: 10, }, 'HEL.one': {
                1: 12, 2: 12, 3: 15, }, 'HEL.two': {
                1: 13.0, 2: (- 0.12), 3: 4.0, }, })]
        tbl_1 = rf.gather(inputs[0], var_name='MORPH1', value_name='MORPH2',
                          value_vars=['MGW.one', 'MGW.two', 'HEL.one', 'HEL.two'],
                          id_vars=['ID'])
        tbl_2 = rf.separate(tbl_1, split_col='MORPH1', into=['MORPH3', 'MORPH4'])
        tbl_3 = rf.group_by(tbl_2, group_cols=['ID', 'MORPH3'])
        tbl_4 = rf.summarise(tbl_3, summaries={
            'mean_val': ('MORPH2', 'mean'), })
        output = rf.spread(tbl_4, columns='MORPH3', values='mean_val')
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.separate', [1]), ('rf.group_by', [2]), ('rf.spread', [3])])
        intermediates = [tbl_1, tbl_2, tbl_3, tbl_4]
        stats = {
            'timeout': True,
            '#sketches with SMT-based deduction': 0, '#sketches without SMT-based deduction': 0,
            '#partial programs with partial evaluation': 0, '#partial programs without partial evaluation': 0,
            'Total time': 0,
        }
        replay_map = {
            'gather_var_name': ['MORPH1'], 'gather_value_name': ['MORPH2'], 'gather_id_vars': [['ID']],
            'gather_value_vars': [['MGW.one', 'MGW.two', 'HEL.one', 'HEL.two']],
            'separate_split_col': ['MORPH1'], 'separate_into': ['MORPH3', 'MORPH4'], 'spread_columns': ['MORPH3'],
            'spread_values': ['mean_val'], 'group_by_group_cols': [['ID', 'MORPH3']],
            'summarise_new_col': ['mean_val'], 'summarise_agg': ['mean'], 'summarise_col': ['MORPH2']
        }

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_66(self):
        b_id = "b_66"
        inputs = [pd.DataFrame({
            'V51': {
                1: 1, 2: 1, 3: 9, 4: 4, 5: 2, 6: 6, 7: 4, 8: 6, }, 'Hour': {
                1: '02:00:00', 2: '08:00:00', 3: '08:00:00', 4: '18:00:00', 5: '06:00:00', 6: '11:00:00', 7: '18:00:00',
                8: '10:00:00', }, 'Group': {
                1: 'SBT', 2: 'SBS', 3: 'SBS', 4: 'SBS', 5: 'SBI', 6: 'SBT', 7: 'SBS', 8: 'SBI', }, })]
        tbl_7 = rf.group_by(inputs[0], group_cols=['Hour'])
        tbl_3 = rf.summarise(tbl_7, summaries={
            'sum': ('V51', 'sum'), })
        tbl_1 = rf.filter_(tbl_3, filter_expr='sum > 6.0')
        morpheus = rf.select(tbl_1, columns_keep=['sum'], columns_remove=None)
        output = morpheus
        skeleton = Skeleton([('rf.group_by', [(- 1)]), ('rf.filter', [1]), ('rf.select', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 3,
            '#partial programs with partial evaluation': 35, '#partial programs without partial evaluation': 160,
            'Total time': 0.19, }
        replay_map = {
            'func_seq': ['rf.group_by', 'rf.summarise', 'rf.filter', 'rf.select'], 'group_by_group_cols': [['Hour']],
            'summarise_new_col': ['sum'], 'summarise_agg': ['sum'], 'summarise_col': ['V51'], 'filter_mode': ['relop'],
            'filter_column_eq': [], 'filter_column_relop': ['sum'], 'filter_eq_op': ['!='], 'filter_relop': ['>'],
            'filter_value_eq': [], 'filter_value_relop': [6.0], 'select_keep_or_remove': [True],
            'select_columns_keep': [['sum']], 'select_columns_remove': [None], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_67(self):
        b_id = "b_67"
        inputs = [pd.DataFrame({
            'year': {
                1: 1955, 2: 1955, 3: 1980, 4: 1980, 5: 1988, 6: 1988, 7: 1980, 8: 1980, }, 'sex': {
                1: 'F', 2: 'M', 3: 'F', 4: 'M', 5: 'F', 6: 'M', 7: 'F', 8: 'M', }, 'name': {
                1: 'Kerry', 2: 'Kerry', 3: 'Kerry', 4: 'Kerry', 5: 'Kerry', 6: 'Kerry', 7: 'Sherry', 8: 'Sherry', },
            'n': {
                1: 615, 2: 1600, 3: 1000, 4: 432, 5: 598, 6: 421, 7: 234, 8: 1200, }, })]
        tbl_3 = rf.filter_(inputs[0], filter_expr="name == 'Kerry'")
        tbl_1 = rf.select(tbl_3, columns_keep=None, columns_remove=['name'])
        morpheus = rf.spread(tbl_1, columns='sex', values='n')
        output = morpheus
        skeleton = Skeleton([('rf.filter', [(- 1)]), ('rf.select', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 14, 'timeout': False, '#sketches without SMT-based deduction': 33,
            '#partial programs with partial evaluation': 1690, '#partial programs without partial evaluation': 6039,
            'Total time': 7.15, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.select', 'rf.spread'], 'filter_mode': ['equality-inequality'],
            'filter_column_eq': ['name'], 'filter_column_relop': [], 'filter_eq_op': ['=='], 'filter_relop': ['>'],
            'filter_value_eq': ['Kerry'], 'filter_value_relop': [], 'select_keep_or_remove': [False],
            'select_columns_keep': [None], 'select_columns_remove': [['name']], 'spread_columns': ['sex'],
            'spread_values': ['n'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_68(self):
        b_id = "b_68"
        inputs = [pd.DataFrame({
            'mpg': {
                1: 21.0, 2: 21.0, 3: 22.8, 4: 21.4, 5: 18.7, 6: 18.1, 7: 14.3, 8: 24.4, }, 'cyl': {
                1: 6, 2: 6, 3: 4, 4: 6, 5: 8, 6: 6, 7: 8, 8: 4, }, 'vs': {
                1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0, 8: 1, }, 'am': {
                1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, }, })]
        tbl_3 = rf.unite(inputs[0], new_col_name='vs_am', cols=['am', 'vs'])
        tbl_1 = rf.group_by(tbl_3, group_cols=['vs_am'])
        morpheus = rf.summarise(tbl_1, summaries={
            'countofvalues': (None, 'count'), })
        output = morpheus
        skeleton = Skeleton([('rf.unite', [(- 1)]), ('rf.group_by', [1])])
        intermediates = [tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 14, 'timeout': False, '#sketches without SMT-based deduction': 52,
            '#partial programs with partial evaluation': 177, '#partial programs without partial evaluation': 2554,
            'Total time': 2.65, }
        replay_map = {
            'func_seq': ['rf.unite', 'rf.group_by', 'rf.summarise'], 'unite_new_col_name': ['vs_am'],
            'unite_cols': [['am', 'vs']], 'group_by_group_cols': [['vs_am']], 'summarise_new_col': ['countofvalues'],
            'summarise_agg': ['count'], 'summarise_col': [], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_69(self):
        b_id = "b_69"
        inputs = [pd.DataFrame({
            'Subject': {
                1: 'A-pre', 2: 'A-post', 3: 'B-pre', 4: 'B-post', }, 'Var1': {
                1: 25, 2: 25, 3: 30, 4: 30, }, 'Var2': {
                1: 27, 2: 26, 3: 28, 4: 28, }, })]
        tbl_7 = rf.gather(inputs[0], var_name='MORPH2127', value_name='MORPH2126', value_vars=['Var1', 'Var2'],
                          id_vars=None)
        tbl_3 = rf.separate(tbl_7, split_col='Subject', into=['SubjectNew', 'MORPH2137'])
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH2142', cols=['MORPH2127', 'MORPH2137'])
        morpheus = rf.spread(tbl_1, columns='MORPH2142', values='MORPH2126')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.unite', [2]), ('rf.spread', [3])])
        intermediates = [tbl_7, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 16, 'timeout': False, '#sketches without SMT-based deduction': 104,
            '#partial programs with partial evaluation': 857, '#partial programs without partial evaluation': 2031,
            'Total time': 4.57, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH2127'],
            'gather_value_name': ['MORPH2126'], 'gather_id_vars': [['Subject']],
            'gather_value_vars': [['Var1', 'Var2']], 'gather_df': ['inputs[0]'], 'separate_split_col': ['Subject'],
            'separate_into': ['SubjectNew', 'MORPH2137'], 'unite_new_col_name': ['MORPH2142'],
            'unite_cols': [['MORPH2127', 'MORPH2137']], 'spread_columns': ['MORPH2142'],
            'spread_values': ['MORPH2126'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_70(self):
        b_id = "b_70"
        inputs = [pd.DataFrame({
            'Title': {
                1: 'Carrie', 2: 'Fried-Green-Tomatoes', 3: 'Amadeus', 4: 'Amityville-Horror', 5: 'Dracula',
                6: 'Speed', }, 'Rating': {
                1: 4, 2: 2, 3: 5, 4: 1, 5: 2, 6: 4, }, 'Action': {
                1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1, }, 'Sci.Fi': {
                1: 1, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, }, })]
        tbl_7 = rf.gather(inputs[0], var_name='genre', value_name='MORPH22554', value_vars=['Action', 'Sci.Fi'],
                          id_vars=None)
        tbl_3 = rf.filter_(tbl_7, filter_expr='MORPH22554 > 0.0')
        tbl_1 = rf.group_by(tbl_3, group_cols=['genre'])
        morpheus = rf.summarise(tbl_1, summaries={
            'average': ('Rating', 'mean'), })
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.filter', [1]), ('rf.group_by', [2])])
        intermediates = [tbl_7, tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 10, 'timeout': False, '#sketches without SMT-based deduction': 13,
            '#partial programs with partial evaluation': 2370, '#partial programs without partial evaluation': 24837,
            'Total time': 23.43, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.filter', 'rf.group_by', 'rf.summarise'], 'gather_var_name': ['genre'],
            'gather_value_name': ['MORPH22554'], 'gather_id_vars': [['Title', 'Rating']],
            'gather_value_vars': [['Action', 'Sci.Fi']], 'gather_df': ['inputs[0]'], 'filter_mode': ['relop'],
            'filter_column_eq': [], 'filter_column_relop': ['MORPH22554'], 'filter_eq_op': ['!='],
            'filter_relop': ['>'], 'filter_value_eq': [], 'filter_value_relop': [0.0],
            'group_by_group_cols': [['genre']], 'summarise_new_col': ['average'], 'summarise_agg': ['mean'],
            'summarise_col': ['Rating'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_71(self):
        b_id = "b_71"
        inputs = [pd.DataFrame({
            'id': {
                1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, }, 'p1': {
                1: (- 0.7833568000000001), 2: (- 0.4073465), 3: 0.2799414, 4: (- 1.3496633), 5: (- 0.10300450000000001),
                6: 0.583907, }, 'p2': {
                1: 0.6383588000000001, 2: 0.348086, 3: (- 0.1938586), 4: (- 0.527108), 5: 0.8642335999999999,
                6: (- 0.9723264000000001), }, 'p3': {
                1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH132', value_name='MORPH131', value_vars=['p1', 'p2'], id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH164', cols=['p3', 'MORPH132'])
        morpheus = rf.spread(tbl_1, columns='MORPH164', values='MORPH131')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 48, '#partial programs without partial evaluation': 195,
            'Total time': 0.49, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH132'],
            'gather_value_name': ['MORPH131'], 'gather_id_vars': [['id', 'p3']], 'gather_value_vars': [['p1', 'p2']],
            'gather_df': ['inputs[0]'], 'unite_new_col_name': ['MORPH164'], 'unite_cols': [['p3', 'MORPH132']],
            'spread_columns': ['MORPH164'], 'spread_values': ['MORPH131'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_72(self):
        b_id = "b_72"
        inputs = [pd.DataFrame({
            'a': {
                1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, }, 'b': {
                1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, }, 'd': {
                1: 0, 2: 200, 3: 300, 4: 0, 5: 600, 6: 0, 7: 100, 8: 200, 9: 0, }, })]
        tbl_3 = rf.filter_(inputs[0], filter_expr='d > 0.0')
        tbl_1 = rf.group_by(tbl_3, group_cols=['a', 'b'])
        morpheus = rf.summarise(tbl_1, summaries={
            'mean_d': ('d', 'mean'), })
        output = morpheus
        skeleton = Skeleton([('rf.filter', [(- 1)]), ('rf.group_by', [1])])
        intermediates = [tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 4, 'timeout': False, '#sketches without SMT-based deduction': 5,
            '#partial programs with partial evaluation': 153, '#partial programs without partial evaluation': 1667,
            'Total time': 1.54, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.group_by', 'rf.summarise'], 'filter_mode': ['relop'], 'filter_column_eq': [],
            'filter_column_relop': ['d'], 'filter_eq_op': ['!='], 'filter_relop': ['>'], 'filter_value_eq': [],
            'filter_value_relop': [0.0], 'group_by_group_cols': [['a', 'b']], 'summarise_new_col': ['mean_d'],
            'summarise_agg': ['mean'], 'summarise_col': ['d'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_73(self):
        b_id = "b_73"
        inputs = [pd.DataFrame({
            'vial_id': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, }, 'band': {
                1: 1, 2: 0, 3: 0, 4: 2, 5: 1, 6: 2, }, 'non_spec': {
                1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, }, 'reads': {
                1: (- 1.7906248505279398), 2: 1.3883797777161202, 3: 0.44903146521550297, 4: 0.9137950252884179,
                5: (- 1.5885562657126298), 6: 0.41834075595853704, }, })]
        tbl_3 = rf.unite(inputs[0], new_col_name='group_id', cols=['band', 'non_spec'])
        tbl_1 = rf.group_by(tbl_3, group_cols=['group_id'])
        morpheus = rf.summarise(tbl_1, summaries={
            'group_mean': ('reads', 'mean'), })
        output = morpheus
        skeleton = Skeleton([('rf.unite', [(- 1)]), ('rf.group_by', [1])])
        intermediates = [tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 13, 'timeout': False, '#sketches without SMT-based deduction': 52,
            '#partial programs with partial evaluation': 230, '#partial programs without partial evaluation': 3245,
            'Total time': 3.4, }
        replay_map = {
            'func_seq': ['rf.unite', 'rf.group_by', 'rf.summarise'], 'unite_new_col_name': ['group_id'],
            'unite_cols': [['band', 'non_spec']], 'group_by_group_cols': [['group_id']],
            'summarise_new_col': ['group_mean'], 'summarise_agg': ['mean'], 'summarise_col': ['reads'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_74(self):
        b_id = "b_74"
        inputs = [pd.DataFrame({
            'Which': {
                1: 'Control', 2: 'Control', 3: 'Treatment', 4: 'Treatment', }, 'Color': {
                1: 'Red', 2: 'Blue', 3: 'Red', 4: 'Blue', }, 'Response': {
                1: 2, 2: 3, 3: 1, 4: 4, }, 'Count': {
                1: 10, 2: 20, 3: 14, 4: 21, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH72', value_name='MORPH71', value_vars=['Response', 'Count'],
                          id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH83', cols=['MORPH72', 'Which'])
        morpheus = rf.spread(tbl_1, columns='MORPH83', values='MORPH71')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 26, '#partial programs without partial evaluation': 99,
            'Total time': 0.21, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH72'],
            'gather_value_name': ['MORPH71'], 'gather_id_vars': [['Which', 'Color']],
            'gather_value_vars': [['Response', 'Count']], 'gather_df': ['inputs[0]'], 'unite_new_col_name': ['MORPH83'],
            'unite_cols': [['MORPH72', 'Which']], 'spread_columns': ['MORPH83'], 'spread_values': ['MORPH71'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_75(self):
        b_id = "b_75"
        inputs = [pd.DataFrame({
            'id': {
                1: 1, 2: 2, }, 'time': {
                1: '2009-01-01', 2: '2009-01-02', }, 'Q1.1': {
                1: 0.48742885, 2: 0.73832471, }, 'Q1.2': {
                1: (- 0.01618826), 2: 0.94383621, }, 'Q2.1': {
                1: 1.52718069, 2: (- 0.40380049), }, 'Q2.2': {
                1: (- 0.29177677), 2: (- 1.19813815), }, })]
        tbl_7 = rf.gather(inputs[0], var_name='MORPH8042', value_name='MORPH8041', value_vars=None,
                          id_vars=['id', 'time'])
        tbl_3 = rf.separate(tbl_7, split_col='MORPH8042', into=['MORPH8197', 'MORPH8198'])
        tbl_1 = rf.spread(tbl_3, columns='MORPH8197', values='MORPH8041')
        morpheus = rf.select(tbl_1, columns_keep=None, columns_remove=['MORPH8198'])
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.separate', [1]), ('rf.spread', [2]), ('rf.select', [3])])
        intermediates = [tbl_7, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 6, 'timeout': False, '#sketches without SMT-based deduction': 51,
            '#partial programs with partial evaluation': 3124, '#partial programs without partial evaluation': 12864,
            'Total time': 39.31, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.separate', 'rf.spread', 'rf.select'], 'gather_var_name': ['MORPH8042'],
            'gather_value_name': ['MORPH8041'], 'gather_id_vars': [['id', 'time']],
            'gather_value_vars': [['Q1.1', 'Q1.2', 'Q2.1', 'Q2.2']], 'gather_df': ['inputs[0]'],
            'separate_split_col': ['MORPH8042'], 'separate_into': ['MORPH8197', 'MORPH8198'],
            'spread_columns': ['MORPH8197'], 'spread_values': ['MORPH8041'], 'select_keep_or_remove': [False],
            'select_columns_keep': [None], 'select_columns_remove': [['MORPH8198']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_76(self):
        b_id = "b_76"
        inputs = [pd.DataFrame({
            'x': {
                1: 'red', 2: 'red', 3: 'red', 4: 'red', 5: 'blue', 6: 'blue', 7: 'blue', 8: 'blue', }, 'y': {
                1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'a', 6: 'b', 7: 'c', 8: 'd', }, 'value.1': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, }, 'value.2': {
                1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18, 7: 19, 8: 20, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='MORPH72', value_name='MORPH71', value_vars=['value.1', 'value.2'],
                          id_vars=None)
        tbl_1 = rf.unite(tbl_3, new_col_name='MORPH79', cols=['MORPH72', 'y'])
        morpheus = rf.spread(tbl_1, columns='MORPH79', values='MORPH71')
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.unite', [1]), ('rf.spread', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 2, 'timeout': False, '#sketches without SMT-based deduction': 8,
            '#partial programs with partial evaluation': 20, '#partial programs without partial evaluation': 90,
            'Total time': 0.17, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.unite', 'rf.spread'], 'gather_var_name': ['MORPH72'],
            'gather_value_name': ['MORPH71'], 'gather_id_vars': [['x', 'y']],
            'gather_value_vars': [['value.1', 'value.2']], 'gather_df': ['inputs[0]'],
            'unite_new_col_name': ['MORPH79'], 'unite_cols': [['MORPH72', 'y']], 'spread_columns': ['MORPH79'],
            'spread_values': ['MORPH71'], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_77(self):
        b_id = "b_77"
        inputs = [pd.DataFrame({
            'obs': {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, }, 'year': {
                1: 2015, 2: 2015, 3: 2015, 4: 2014, 5: 2014, 6: 2014, 7: 2015, }, 'type': {
                1: 'A', 2: 'A', 3: 'B', 4: 'A', 5: 'B', 6: 'A', 7: 'A', }, })]
        tbl_7 = rf.filter_(inputs[0], filter_expr='year > 2014.0')
        tbl_3 = rf.group_by(tbl_7, group_cols=['type'])
        tbl_1 = rf.summarise(tbl_3, summaries={
            'freq': (None, 'count'), })
        morpheus = rf.inner_join(tbl_1, inputs[0])
        output = morpheus
        skeleton = Skeleton([('rf.filter', [(- 1)]), ('rf.group_by', [1]), ('rf.inner_join', [2, (- 1)])])
        intermediates = [tbl_7, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 68, 'timeout': False, '#sketches without SMT-based deduction': 168,
            '#partial programs with partial evaluation': 5428, '#partial programs without partial evaluation': 10788,
            'Total time': 24.79, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.group_by', 'rf.summarise', 'rf.inner_join'], 'filter_mode': ['relop'],
            'filter_column_eq': [], 'filter_column_relop': ['year'], 'filter_eq_op': ['!='], 'filter_relop': ['>'],
            'filter_value_eq': [], 'filter_value_relop': [2014.0], 'group_by_group_cols': [['type']],
            'summarise_new_col': ['freq'], 'summarise_agg': ['count'], 'summarise_col': [], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_78(self):
        b_id = "b_78"
        inputs = [pd.DataFrame({
            'x': {
                1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, }, 'x2': {
                1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 2, 8: 2, }, 'y': {
                1: 1.41, 2: 1.39, 3: 1.9, 4: 2.1, 5: 0.9, 6: 1.1, 7: 1.9, 8: 2.1, }, })]
        tbl_15 = rf.filter_(inputs[0], filter_expr='y < 1.9')
        tbl_7 = rf.group_by(tbl_15, group_cols=['x'])
        tbl_3 = rf.summarise(tbl_7, summaries={
            'a': ('y', 'mean'), })
        tbl_1 = rf.inner_join(tbl_3, inputs[0])
        morpheus = rf.mutate(tbl_1, new_col_name='z', operation='div', col_args=['y', 'a'])
        output = morpheus
        skeleton = Skeleton(
            [('rf.filter', [(- 1)]), ('rf.group_by', [1]), ('rf.inner_join', [2, (- 1)]), ('rf.mutate', [3])])
        intermediates = [tbl_15, tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 125, 'timeout': False, '#sketches without SMT-based deduction': 411,
            '#partial programs with partial evaluation': 31076, '#partial programs without partial evaluation': 106067,
            'Total time': 275.15, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.group_by', 'rf.summarise', 'rf.inner_join', 'rf.mutate'],
            'filter_mode': ['relop'], 'filter_column_eq': [], 'filter_column_relop': ['y'], 'filter_eq_op': ['!='],
            'filter_relop': ['<'], 'filter_value_eq': [], 'filter_value_relop': [1.9], 'group_by_group_cols': [['x']],
            'summarise_new_col': ['a'], 'summarise_agg': ['mean'], 'summarise_col': ['y'], 'mutate_new_col_name': ['z'],
            'mutate_operation': ['div'], 'mutate_col_args_normalize': [None], 'mutate_col_args_div': [['y', 'a']], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_79(self):
        b_id = "b_79"
        inputs = [pd.DataFrame({
            'id': {
                1: 1, 2: 2, 3: 3, 4: 4, }, 'dept': {
                1: 'CS', 2: 'EE', 3: 'Civil', 4: 'Physics', }, 'employee': {
                1: 'Yossi ', 2: 'Pitt ', 3: 'Deepak', 4: 'Golan', }, 'salary': {
                1: 21000, 2: 23400, 3: 26800, 4: 91000, }, })]
        tbl_1 = rf.filter_(inputs[0], filter_expr='salary > 23400.0')
        tbl_2 = rf.group_by(tbl_1, group_cols=['id'])
        tbl_3 = rf.summarise(tbl_2, summaries={
            'mean': ('salary', 'mean'), })
        morpheus = rf.select(tbl_3, columns_keep=['mean'], columns_remove=None)
        output = morpheus
        skeleton = Skeleton([('rf.filter', [(- 1)]), ('rf.group_by', [1]), ('rf.select', [2])])
        intermediates = [tbl_1, tbl_3, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 3, 'timeout': False, '#sketches without SMT-based deduction': 10,
            '#partial programs with partial evaluation': 17, '#partial programs without partial evaluation': 97,
            'Total time': 0.17, }
        replay_map = {
            'func_seq': ['rf.filter', 'rf.group_by', 'rf.summarise', 'rf.select'], 'filter_mode': ['relop'],
            'filter_column_eq': [], 'filter_column_relop': ['salary'], 'filter_eq_op': ['!='], 'filter_relop': ['>'],
            'filter_value_eq': [], 'filter_value_relop': [23400.0], 'group_by_group_cols': [['id']],
            'summarise_new_col': ['mean'], 'summarise_agg': ['mean'], 'summarise_col': ['salary'],
            'select_keep_or_remove': [True], 'select_columns_keep': [['mean']], 'select_columns_remove': [None], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_80(self):
        b_id = "b_80"
        inputs = [pd.DataFrame({
            'order_id': {
                1: 'A', 2: 'B', 3: 'C', }, 'Prod1': {
                1: 1, 2: 0, 3: 1, }, 'prod2': {
                1: 0, 2: 0, 3: 1, }, 'Prod3': {
                1: 1, 2: 1, 3: 0, }, 'Prod4': {
                1: 1, 2: 1, 3: 1, }, 'Prod5': {
                1: 1, 2: 0, 3: 1, }, })]
        tbl_3 = rf.gather(inputs[0], var_name='var', value_name='MORPH1244', value_vars=None, id_vars=['order_id'])
        tbl_1 = rf.filter_(tbl_3, filter_expr='MORPH1244 > 0.0')
        morpheus = rf.select(tbl_1, columns_keep=['order_id', 'var'], columns_remove=None)
        output = morpheus
        skeleton = Skeleton([('rf.gather', [(- 1)]), ('rf.filter', [1]), ('rf.select', [2])])
        intermediates = [tbl_3, tbl_1, morpheus]
        stats = {
            '#sketches with SMT-based deduction': 5, 'timeout': False, '#sketches without SMT-based deduction': 17,
            '#partial programs with partial evaluation': 316, '#partial programs without partial evaluation': 2170,
            'Total time': 2.18, }
        replay_map = {
            'func_seq': ['rf.gather', 'rf.filter', 'rf.select'], 'gather_var_name': ['var'],
            'gather_value_name': ['MORPH1244'], 'gather_id_vars': [['order_id']],
            'gather_value_vars': [['Prod1', 'prod2', 'Prod3', 'Prod4', 'Prod5']], 'gather_df': ['inputs[0]'],
            'filter_mode': ['relop'], 'filter_column_eq': [], 'filter_column_relop': ['MORPH1244'],
            'filter_eq_op': ['!='], 'filter_relop': ['>'], 'filter_value_eq': [], 'filter_value_relop': [0.0],
            'select_keep_or_remove': [True], 'select_columns_keep': [['order_id', 'var']],
            'select_columns_remove': [None], }
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_81(self):
        b_id = "b_81"
        inputs = [pd.DataFrame({'ID': {0: 'bo', 1: 'bo', 2: 'bo', 3: 'co', 4: 'co', 5: 'co', 6: 'do', 7: 'do', 8: 'do',
                                       9: 'fo', 10: 'fo', 11: 'fo'},
                                'a': {0: 1, 1: 4, 2: 2, 3: 9, 4: 6, 5: 5, 6: 7, 7: 8, 8: 11, 9: 3, 10: 10, 11: 12},
                                'b': {0: -36, 1: -16, 2: -18, 3: 48, 4: 59, 5: 12, 6: -7, 7: 36, 8: -34, 9: 57, 10: -21,
                                      11: 17},
                                'd': {0: -11, 1: -10, 2: -9, 3: -8, 4: -7, 5: -6, 6: -5, 7: -4, 8: -3, 9: -2, 10: -1,
                                      11: 0},
                                'O': {0: 'Nn', 1: 'Nn', 2: 'Nn', 3: 'Nn', 4: 'Nn', 5: 'Nn', 6: 'Hy', 7: 'Hy', 8: 'Hy',
                                      9: 'Hy', 10: 'Hy', 11: 'Hy'}})]
        # output = [pd.DataFrame({'O': {0: 'Nn'}, 'value': {0: 1.388889}})]

        interm_1 = rf.filter_(inputs[0], filter_expr='O == "Nn"')
        interm_2 = rf.gather(interm_1, var_name='k', value_name='v', value_vars=['a', 'b', 'd'], id_vars=None)
        interm_3 = rf.group_by(interm_2, group_cols=['O'])
        interm_4 = rf.summarise(interm_3, summaries={'value': ('v', 'mean')})
        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.filter', [-1]), ('rf.gather', [1]), ('rf.group_by', [2])])
        replay_map = {'filter_eq_op': ['=='], 'filter_column_eq': ['O'], 'filter_value_eq': ['Nn'],
                      'filter_mode': ['equality-inequality'], 'gather_df': ['interm_1'], 'gather_var_name': ['k'],
                      'gather_value_name': ['v'], 'gather_value_vars': [['a', 'b', 'd']],
                      'gather_id_vars': [['ID', 'O']],
                      'group_by_group_cols': [['O']], 'summarise_new_col': ['value'], 'summarise_col': ['v'],
                      'summarise_agg': ['mean'], 'func_seq': ['rf.filter', 'rf.gather', 'rf.group_by', 'rf.summarise']}

        stats = {}
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_82(self):
        b_id = "b_82"
        inputs = [pd.DataFrame({'Color': {0: 'red', 1: 'red', 2: 'blue', 3: 'green', 4: 'blue', 5: 'blue', 6: 'red',
                                          7: 'red', 8: 'yellow', 9: 'green'},
                                'Type': {0: 'Outdoor', 1: 'Indoor', 2: 'Both', 3: 'Indoor', 4: 'Indoor', 5: 'Both',
                                         6: 'Indoor', 7: 'Outdoor', 8: 'Outdoor', 9: 'Indoor'},
                                'W1': {0: 2, 1: 5, 2: 6, 3: 8, 4: 11, 5: 12, 6: 14, 7: 14, 8: 12, 9: 2},
                                'W2': {0: 3, 1: 6, 2: 7, 3: 9, 4: 12, 5: 13, 6: 15, 7: 15, 8: 13, 9: 3},
                                'W3': {0: 4, 1: 7, 2: 8, 3: 10, 4: 13, 5: 14, 6: 16, 7: 16, 8: 14, 9: 4},
                                'W4': {0: 5, 1: 8, 2: 9, 3: 11, 4: 14, 5: 15, 6: 17, 7: 17, 8: 15, 9: 5}})]
        output = [pd.DataFrame({'Color': {0: 'blue', 1: 'blue', 2: 'blue', 3: 'blue', 4: 'green', 5: 'red', 6: 'red',
                                          7: 'red', 8: 'red', 9: 'yellow', 10: 'yellow', 11: 'yellow', 12: 'yellow'},
                                'Week': {0: 'W1', 1: 'W2', 2: 'W3', 3: 'W4', 4: 'W4', 5: 'W1', 6: 'W2', 7: 'W3',
                                         8: 'W4', 9: 'W1', 10: 'W2', 11: 'W3', 12: 'W4'},
                                'Count': {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 1, 10: 1, 11: 1,
                                          12: 1}})]

        interm_1 = rf.gather(inputs[0], var_name='Week', value_name='value', value_vars=['W1', 'W2', 'W3', 'W4'],
                             id_vars=None)
        interm_2 = rf.filter_(interm_1, filter_expr='value > 10')
        interm_3 = rf.group_by(interm_2, group_cols=['Color', 'Week'])
        interm_4 = rf.summarise(interm_3, summaries={'count': (None, 'count')})
        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.filter', [1]), ('rf.group_by', [2])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['Week'], 'gather_value_name': ['value'],
                      'gather_value_vars': [['W1', 'W2', 'W3', 'W4']], 'gather_id_vars': [['Color', 'Type']],
                      'filter_relop': ['>'],
                      'filter_column_relop': ['value'], 'filter_value_relop': [10], 'filter_mode': ['relop'],
                      'group_by_group_cols': [['Color', 'Week']], 'summarise_new_col': ['count'],
                      'summarise_col': [None], 'summarise_agg': ['count'],
                      'func_seq': ['rf.gather', 'rf.filter', 'rf.group_by', 'rf.summarise']}

        stats = {}
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_83(self):
        b_id = "b_83"
        inputs = [pd.DataFrame({'Timepoint': {0: 7, 1: 14, 2: 21, 3: 28}, 'Group1': {0: 60, 1: 66, 2: 88, 3: 90},
                                'Error1_Group1': {0: 4, 1: 6, 2: 8, 3: 2}, 'Group2': {0: 60, 1: 90, 2: 120, 3: 150},
                                'Error2_Group1': {0: 14, 1: 16, 2: 13, 3: 25}})]

        interm_1 = rf.gather(inputs[0], var_name='Key', value_name='Val', value_vars=['Error1_Group1', 'Error2_Group1'],
                             id_vars=None)
        interm_2 = rf.separate(interm_1, split_col='Key', into=["err", "mGroup"])
        interm_3 = rf.spread(interm_2, columns='err', values='Val')
        interm_4 = rf.select(interm_3, columns_keep=None, columns_remove=['Timepoint'])

        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.separate', [1]), ('rf.spread', [2]), ('rf.select', [3])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['Key'], 'gather_value_name': ['Val'],
                      'gather_value_vars': [['Error1_Group1', 'Error2_Group1']],
                      'gather_id_vars': [['Timepoint', 'Group1', 'Group2']],
                      'separate_split_col': ['Key'], 'separate_into': ['err', 'mGroup'], 'spread_columns': ['err'],
                      'spread_values': ['Val'], 'select_columns_keep': [None], 'select_columns_remove': [['Timepoint']],
                      'select_keep_or_remove': [False],
                      'func_seq': ['rf.gather', 'rf.separate', 'rf.spread', 'rf.select']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_84(self):
        b_id = "b_84"
        inputs = [pd.DataFrame({'GeneID': {0: 'A2M', 1: 'ABL1', 2: 'ACP1'}, 'D_1': {0: 8876.5, 1: 2120.8, 2: 1266.6},
                                'T_1': {0: 8857.9, 1: 1664.9, 2: 1347.1}, 'D_2': {0: 10246.8, 1: 2525.0, 2: 910.95},
                                'T_2': {0: 9453.9, 1: 1546.4, 2: 725.1}, 'D_3': {0: 6279.6, 1: 1993.0, 2: 1327.6},
                                'T_3': {0: 3846.5, 1: 1713.7, 2: 1589.5}, 'D_4': {0: 8735.3, 1: 1849.7, 2: 1175.0},
                                'T_4': {0: 6609.9, 1: 1761.9, 2: 1086.9}, 'D_5': {0: 7732.95, 1: 2297.7, 2: 1187.3},
                                'T_5': {0: 2452.4, 1: 2462.5, 2: 1065.15}, 'D_6': {0: 8705.2, 1: 2698.2, 2: 1080.0},
                                'T_6': {0: 6679.0, 1: 1975.8, 2: 1048.2}, 'D_7': {0: 7510.5, 1: 2480.3, 2: 1213.8},
                                'T_7': {0: 4318.3, 1: 1694.6, 2: 1337.9}, 'D_8': {0: 8957.7, 1: 2471.0, 2: 831.5},
                                'T_8': {0: 4092.4, 1: 1784.1, 2: 814.1}})]
        output = [pd.DataFrame(
            {'col1': {0: 'D', 1: 'T'}, 'A2M': {0: 67044.55, 1: 46310.3}, 'ABL1': {0: 18435.7, 1: 14603.9},
             'ACP1': {0: 8992.75, 1: 9013.95}})]
        interm_1 = rf.gather(inputs[0], var_name='Key', value_name='Val', value_vars=None, id_vars=['GeneID'])
        interm_2 = rf.separate(interm_1, split_col='Key', into=["col1", "B"])
        interm_3 = rf.group_by(interm_2, group_cols=['GeneID', 'col1'])
        interm_4 = rf.summarise(interm_3, summaries={'count': ('Val', 'sum')})
        interm_5 = rf.spread(interm_4, columns='GeneID', values='count')
        output = interm_5
        intermediates = [interm_1, interm_2, interm_3, interm_4, interm_5]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.separate', [1]), ('rf.group_by', [2]), ('rf.spread', [3])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['Key'], 'gather_value_name': ['Val'],
                      'gather_value_vars': [
                          ['D_1', 'T_1', 'D_2', 'T_2', 'D_3', 'T_3', 'D_4', 'T_4', 'D_5', 'T_5', 'D_6', 'T_6', 'D_7',
                           'T_7', 'D_8', 'T_8']], 'gather_id_vars': [['GeneID']], 'separate_split_col': ['Key'],
                      'separate_into': ['col1', 'B'], 'group_by_group_cols': [['GeneID', 'col1']],
                      'summarise_new_col': ['count'], 'summarise_col': ['Val'], 'summarise_agg': ['sum'],
                      'spread_columns': ['GeneID'], 'spread_values': ['count'],
                      'func_seq': ['rf.gather', 'rf.separate', 'rf.group_by', 'rf.summarise', 'rf.spread']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_85(self):
        b_id = "b_85"
        inputs = [pd.DataFrame({'name': {0: 'Eric', 1: 'Bob', 2: 'Mark', 3: 'Bob', 4: 'Bob', 5: 'Eric', 6: 'Mark',
                                         7: 'Eric', 8: 'Bob', 9: 'Eric', 10: 'Tom', 11: 'Kara', 12: 'Jim', 13: 'Mark',
                                         14: 'Jim', 15: 'Kara', 16: 'Kara', 17: 'Jim', 18: 'Tom', 19: 'Tom'},
                                'metric': {0: 'height', 1: 'height', 2: 'height', 3: 'weight', 4: 'weight', 5: 'weight',
                                           6: 'weight', 7: 'grade', 8: 'grade', 9: 'weight', 10: 'weight', 11: 'grade',
                                           12: 'grade', 13: 'grade', 14: 'height', 15: 'height', 16: 'weight',
                                           17: 'weight', 18: 'grade', 19: 'height'},
                                'values': {0: 6, 1: 5, 2: 4, 3: 120, 4: 118, 5: 100, 6: 180, 7: 2, 8: 2, 9: 10, 10: 80,
                                           11: 9, 12: 8, 13: 4, 14: 11, 15: 33, 16: 99, 17: 90, 18: 5, 19: 109}})]
        output = [pd.DataFrame(
            {'name': {0: 'Bob', 1: 'Eric', 2: 'Jim', 3: 'Kara', 4: 'Tom'}, 'grade': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
             'height': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}})]

        interm_1 = rf.group_by(inputs[0], group_cols=['name', 'metric'])
        interm_2 = rf.summarise(interm_1, summaries={'count': (None, 'count')})
        interm_3 = rf.filter_(interm_2, filter_expr='metric != "weight"')
        interm_4 = rf.filter_(interm_3, filter_expr='name != "Mark"')
        interm_5 = rf.spread(interm_4, columns='metric', values='count')
        output = interm_5
        intermediates = [interm_1, interm_2, interm_3, interm_4, interm_5]
        skeleton = Skeleton([('rf.group_by', [-1]), ('rf.filter', [1]), ('rf.filter', [2]),
                             ('rf.spread', [3])])
        replay_map = {'group_by_group_cols': [['name', 'metric']], 'summarise_new_col': ['count'],
                      'summarise_col': [None], 'summarise_agg': ['count'], 'filter_eq_op': ['!=', '!='],
                      'filter_column_eq': ['metric', 'name'], 'filter_value_eq': ['weight', 'Mark'],
                      'filter_mode': ['equality-inequality', 'equality-inequality'], 'spread_columns': ['metric'],
                      'spread_values': ['count'],
                      'func_seq': ['rf.group_by', 'rf.summarise', 'rf.filter_', 'rf.filter_', 'rf.spread']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_86(self):
        b_id = "b_86"
        inputs = [pd.DataFrame(
            {'ID': {0: 'ID1', 1: 'ID2', 2: 'ID3', 3: 'ID4'}, 'Name': {0: 'A1', 1: 'B1', 2: 'C1', 3: 'D1'},
             'A': {0: 1, 1: 2, 2: 3, 3: 4}, 'B': {0: 5, 1: 6, 2: 7, 3: 8}, 'C': {0: 9, 1: 10, 2: 11, 3: 12}})]
        output = [pd.DataFrame({'ID': {0: 'ID1', 1: 'ID2', 2: 'ID3', 3: 'ID4'}, 'flag': {0: 14, 1: 16, 2: 18, 3: 20}})]
        interm_1 = rf.select(inputs[0], columns_keep=['ID', 'B', "C"], columns_remove=None)
        interm_2 = rf.gather(interm_1, var_name='var', value_name='v', value_vars=None, id_vars=['ID'])
        interm_3 = rf.group_by(interm_2, group_cols=['ID'])
        interm_4 = rf.summarise(interm_3, summaries={'flag': ('v', 'sum')})
        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.select', [-1]), ('rf.gather', [1]), ('rf.group_by', [2])])
        replay_map = {'select_columns_keep': [['ID', 'B', 'C']], 'select_columns_remove': [None],
                      'select_keep_or_remove': [True], 'gather_df': ['interm_1'], 'gather_var_name': ['var'],
                      'gather_value_name': ['v'], 'gather_value_vars': [['B', 'C']], 'gather_id_vars': [['ID']],
                      'group_by_group_cols': [['ID']], 'summarise_new_col': ['flag'], 'summarise_col': ['v'],
                      'summarise_agg': ['sum'], 'func_seq': ['rf.select', 'rf.gather', 'rf.group_by', 'rf.summarise']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_87(self):
        b_id = "b_87"
        inputs = [pd.DataFrame({'expr': {0: 'base_1d4', 1: 'base_1d4', 2: 'base_1d5', 3: 'base_1d5', 4: 'dplyr_1d4',
                                         5: 'dplyr_1d4', 6: 'dplyr_1d5', 7: 'dplyr_1d5'},
                                'time': {0: 20, 1: 40, 2: 40, 3: 80, 4: 2, 5: 4, 6: 8, 7: 12},
                                'ext': {0: 100, 1: 200, 2: 300, 3: 400, 4: 500, 5: 600, 6: 700, 7: 800},
                                'misc': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}})]
        output = [pd.DataFrame(
            {'col2': {0: '1d4', 1: '1d5'}, 'base': {0: 30, 1: 60}, 'dplyr': {0: 3, 1: 10}, 'ratio': {0: 10, 1: 6}})]
        interm_1 = rf.separate(inputs[0], split_col='expr', into=["col1", "col2"])
        interm_2 = rf.select(interm_1, columns_keep=['col1', 'col2', 'time'], columns_remove=None)
        interm_3 = rf.group_by(interm_2, group_cols=['col1', 'col2'])
        interm_4 = rf.summarise(interm_3, summaries={'avg': ('time', 'mean')})
        interm_5 = rf.spread(interm_4, columns='col1', values='avg')
        interm_6 = rf.mutate(interm_5, new_col_name='ratio', operation='div', col_args=['base', 'dplyr'])
        output = interm_6
        intermediates = [interm_1, interm_2, interm_3, interm_4, interm_5, interm_6]
        skeleton = Skeleton(
            [('rf.separate', [-1]), ('rf.select', [1]), ('rf.group_by', [2]), ('rf.spread', [3]), ('rf.mutate', [4])])
        replay_map = {'separate_split_col': ['expr'], 'separate_into': ['col1', 'col2'],
                      'select_columns_keep': [['col1', 'col2', 'time']], 'select_columns_remove': [None],
                      'select_keep_or_remove': [True], 'group_by_group_cols': [['col1', 'col2']],
                      'summarise_new_col': ['avg'], 'summarise_col': ['time'], 'summarise_agg': ['mean'],
                      'spread_columns': ['col1'], 'spread_values': ['avg'], 'mutate_new_col_name': ['ratio'],
                      'mutate_operation': ['div'], 'mutate_col_args_div': [['base', 'dplyr']],
                      'mutate_col_args_normalize': [None],
                      'func_seq': ['rf.separate', 'rf.select', 'rf.group_by', 'rf.summarise', 'rf.spread', 'rf.mutate']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_88(self):
        b_id = "b_88"
        inputs = [pd.DataFrame({'Id': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12},
                                'Group': {0: 'A', 1: 'A', 2: 'A', 3: 'A', 4: 'B', 5: 'B', 6: 'B', 7: 'B', 8: 'C',
                                          9: 'C', 10: 'C', 11: 'C'},
                                'Var1': {0: 'good', 1: 'good', 2: 'bad', 3: 'bad', 4: 'good', 5: 'bad', 6: 'good',
                                         7: 'good', 8: 'bad', 9: 'good', 10: 'bad', 11: 'bad'},
                                'Var2': {0: 10, 1: 2, 2: 3, 3: 2, 4: 10, 5: 9, 6: 2, 7: 8, 8: 7, 9: 5, 10: 7, 11: 9}})]
        output = [pd.DataFrame({'Group': {0: 'B'}, 'bad': {0: 9}, 'good': {0: 20}})]
        interm_1 = rf.filter_(inputs[0], filter_expr='Group == "B"')
        interm_2 = rf.group_by(interm_1, group_cols=['Group', 'Var1'])
        interm_3 = rf.summarise(interm_2, summaries={'s': ('Var2', 'sum')})
        interm_4 = rf.spread(interm_3, columns='Var1', values='s')
        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.filter', [-1]), ('rf.group_by', [1]), ('rf.spread', [2])])
        replay_map = {'filter_eq_op': ['=='], 'filter_column_eq': ['Group'], 'filter_value_eq': ['B'],
                      'filter_mode': ['equality-inequality'], 'group_by_group_cols': [['Group', 'Var1']],
                      'summarise_new_col': ['s'], 'summarise_col': ['Var2'], 'summarise_agg': ['sum'],
                      'spread_columns': ['Var1'], 'spread_values': ['s'],
                      'func_seq': ['rf.filter_', 'rf.group_by', 'rf.summarise', 'rf.spread']}

        stats = {}
        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_89(self):
        b_id = "b_89"
        inputs = [pd.DataFrame({'rowname': {0: 1, 1: 2, 2: 3}, 'CA': {0: 'A002', 1: 'A002', 2: 'A002'},
                                'DATE_1': {0: '07-27-13', 1: '07-28-13', 2: '07-29-13'},
                                'TIME_1': {0: '00:10:00', 1: '08:20:00', 2: '16:20:00'},
                                'ENTRIES_1': {0: 4209603, 1: 4210490, 2: 4211586},
                                'DATE_2': {0: '07-27-13', 1: '07-28-13', 2: '07-30-13'},
                                'TIME_2': {0: '08:00:00', 1: '16:40:00', 2: '00:00:00'},
                                'ENTRIES_2': {0: 4209663, 1: 4210775, 2: 4212845},
                                'DATE_3': {0: '07-27-17', 1: '07-28-17', 2: '07-30-17'},
                                'TIME_3': {0: '18:00:00', 1: '06:00:00', 2: '10:00:00'},
                                'ENTRIES_3': {0: 1209663, 1: 1210775, 2: 1212845}})]
        output = [pd.DataFrame({'rowname': {0: 2, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3},
                                'CA': {0: 'A002', 1: 'A002', 2: 'A002', 3: 'A002', 4: 'A002', 5: 'A002'},
                                'col2': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3},
                                'DATE': {0: '07-28-13', 1: '07-28-13', 2: '07-28-17', 3: '07-29-13', 4: '07-30-13',
                                         5: '07-30-17'},
                                'ENTRIES': {0: 4210490, 1: 4210775, 2: 1210775, 3: 4211586, 4: 4212845, 5: 1212845},
                                'TIME': {0: '08:20:00', 1: '16:40:00', 2: '06:00:00', 3: '16:20:00', 4: '00:00:00',
                                         5: '10:00:00'}})]
        interm_1 = rf.gather(inputs[0], var_name='k', value_name='v', value_vars=None, id_vars=['rowname', 'CA'])
        interm_2 = rf.separate(interm_1, split_col='k', into=["col1", "col2"])
        interm_3 = rf.filter_(interm_2, filter_expr='rowname != 1')
        interm_4 = rf.spread(interm_3, columns='col1', values='v')
        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.separate', [1]), ('rf.filter', [2]), ('rf.spread', [3])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['k'], 'gather_value_name': ['v'],
                      'gather_value_vars': [
                          ['DATE_1', 'TIME_1', 'ENTRIES_1', 'DATE_2', 'TIME_2', 'ENTRIES_2', 'DATE_3', 'TIME_3',
                           'ENTRIES_3']],
                      'gather_id_vars': [['rowname', 'CA']], 'separate_split_col': ['k'],
                      'separate_into': ['col1', 'col2'], 'filter_eq_op': ['!='], 'filter_column_eq': ['rowname'],
                      'filter_value_eq': [1], 'filter_mode': ['equality-inequality'], 'spread_columns': ['col1'],
                      'spread_values': ['v'],
                      'func_seq': ['rf.gather', 'rf.separate', 'rf.filter', 'rf.spread']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_90(self):
        b_id = "b_90"
        inputs = [pd.DataFrame(
            {'name1': {0: 'a', 1: 'b', 2: 'c'}, 'con1_1': {0: 23, 1: 25, 2: 28}, 'con1_2': {0: 33, 1: 34, 2: 29},
             'con1_3': {0: 22, 1: 22, 2: 22}, 'con2_1': {0: 23, 1: 22, 2: 30}, 'con2_2': {0: 40, 1: 50, 2: 60},
             'con2_3': {0: 40, 1: 40, 2: 40}})]
        output = [pd.DataFrame({'name1': {0: 'a', 1: 'b', 2: 'c'}, 'con1': {0: 26.0, 1: 27.0, 2: 26.3333333333333},
                                'con2': {0: 34.3333333333333, 1: 37.3333333333333, 2: 43.3333333333333}})]

        interm_1 = rf.gather(inputs[0], var_name='key', value_name='value', value_vars=None, id_vars=['name1'])
        interm_2 = rf.separate(interm_1, split_col='key', into=["col1", "col2"])
        interm_3 = rf.group_by(interm_2, group_cols=['name1', 'col1'])
        interm_4 = rf.summarise(interm_3, summaries={'s': ('value', 'mean')})
        interm_5 = rf.spread(interm_4, columns='col1', values='s')
        output = interm_5
        intermediates = [interm_1, interm_2, interm_3, interm_4, interm_5]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.separate', [1]), ('rf.group_by', [2]), ('rf.spread', [3])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['key'], 'gather_value_name': ['value'],
                      'gather_value_vars': [['con1_1', 'con1_2', 'con1_3', 'con2_1', 'con2_2', 'con2_3']],
                      'gather_id_vars': [['name1']], 'separate_split_col': ['key'],
                      'separate_into': ['col1', 'col2'], 'group_by_group_cols': [['name1', 'col1']],
                      'summarise_new_col': ['s'], 'summarise_col': ['value'], 'summarise_agg': ['mean'],
                      'spread_columns': ['col1'], 'spread_values': ['s'],
                      'func_seq': ['rf.gather', 'rf.separate', 'rf.group_by', 'rf.summarise', 'rf.spread']}

        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_91(self):
        b_id = "b_91"
        inputs = [pd.DataFrame({'Geotype': {0: 'A', 1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'B'},
                                'Strategy': {0: 'Demand', 1: 'Strategy_1', 2: 'Strategy_2', 3: 'Demand',
                                             4: 'Strategy_1', 5: 'Strategy_2'},
                                'Year.1': {0: 1, 1: 2, 2: 3, 3: 8, 4: 9, 5: 10},
                                'Year.2': {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10},
                                'Year.3': {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10}})]

        interm_1 = rf.filter_(inputs[0], filter_expr='Strategy != "Demand"')
        interm_2 = rf.gather(interm_1, var_name="var", value_name="val", value_vars=["Year.1", "Year.2", "Year.3"],
                             id_vars=["Geotype", "Strategy"])
        interm_3 = rf.group_by(interm_2, group_cols=['Geotype', "var"])
        interm_4 = rf.summarise(interm_3, summaries={'val': ('val', 'sum')})
        interm_5 = rf.spread(interm_4, columns="var", values='val')

        output = interm_5
        intermediates = [interm_1, interm_2, interm_3, interm_4, interm_5]
        skeleton = Skeleton(
            [('rf.filter', [-1]), ('rf.gather', [1]), ('rf.group_by', [2]), ('rf.spread', [3])])
        replay_map = {'filter_eq_op': ['!='], 'filter_column_eq': ['Strategy'], 'filter_value_eq': ['Demand'],
                      'filter_mode': ['equality-inequality'], 'gather_df': ['interm_1'], 'gather_var_name': ['var'],
                      'gather_value_name': ['val'], 'gather_value_vars': [['Year.1', 'Year.2', 'Year.3']],
                      'gather_id_vars': [['Geotype', 'Strategy']], 'group_by_group_cols': [['Geotype', 'var']],
                      'summarise_new_col': ['val'], 'summarise_col': ['val'], 'summarise_agg': ['sum'],
                      'spread_columns': ['var'], 'spread_values': ['val'],
                      'func_seq': ['rf.filter', 'rf.gather', 'rf.group_by', 'rf.summarise', 'rf.spread']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_92(self):
        b_id = "b_92"
        inputs = [pd.DataFrame({'Geotype': {0: 'A', 1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'B', 6: 'C', 7: 'C', 8: 'C'},
                                'Strategy': {0: 'Demand', 1: 'Strategy_1', 2: 'Strategy_2', 3: 'Demand',
                                             4: 'Strategy_1', 5: 'Strategy_2', 6: 'Demand', 7: 'Strategy_1',
                                             8: 'Strategy_2'},
                                'Year.1': {0: 1, 1: 2, 2: 3, 3: 8, 4: 9, 5: 10, 6: 8, 7: 9, 8: 10},
                                'Year.2': {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 8, 7: 9, 8: 10}})]
        output = [pd.DataFrame(
            {'Geotype': {0: 'A', 1: 'B', 2: 'C'}, 'Year.1': {0: 5, 1: 19, 2: 19}, 'Year.2': {0: 13, 1: 19, 2: 19}})]
        interm_1 = rf.filter_(inputs[0], filter_expr='Strategy != "Demand"')
        interm_2 = rf.gather(interm_1, var_name="var", value_name="val", value_vars=["Year.1", "Year.2"],
                             id_vars=["Geotype", "Strategy"])
        interm_3 = rf.group_by(interm_2, group_cols=['Geotype', "var"])
        interm_4 = rf.summarise(interm_3, summaries={'val': ('val', 'sum')})
        interm_5 = rf.spread(interm_4, columns="var", values='val')

        output = interm_5
        intermediates = [interm_1, interm_2, interm_3, interm_4, interm_5]
        skeleton = Skeleton(
            [('rf.filter', [-1]), ('rf.gather', [1]), ('rf.group_by', [2]), ('rf.spread', [3])])
        replay_map = {'filter_eq_op': ['!='], 'filter_column_eq': ['Strategy'], 'filter_value_eq': ['Demand'],
                      'filter_mode': ['equality-inequality'], 'gather_df': ['interm_1'], 'gather_var_name': ['var'],
                      'gather_value_name': ['val'], 'gather_value_vars': [['Year.1', 'Year.2']],
                      'gather_id_vars': [['Geotype', 'Strategy']], 'group_by_group_cols': [['Geotype', 'var']],
                      'summarise_new_col': ['val'], 'summarise_col': ['val'], 'summarise_agg': ['sum'],
                      'spread_columns': ['var'], 'spread_values': ['val'],
                      'func_seq': ['rf.filter', 'rf.gather', 'rf.group_by', 'rf.summarise', 'rf.spread']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_93(self):
        b_id = "b_93"
        inputs = [pd.DataFrame({'sample_ID': {0: 382870, 1: 382872, 2: 487405, 3: 487405, 4: 382899, 5: 382900,
                                              6: 382901, 7: 382902, 8: 382903, 9: 382904, 10: 382905, 11: 382906,
                                              12: 382907, 13: 382908},
                                'site': {0: 'site_3', 1: 'site_2', 2: 'site_3', 3: 'site_3', 4: 'site_1', 5: 'site_2',
                                         6: 'site_3', 7: 'site_2', 8: 'site_1', 9: 'site_2', 10: 'site_3', 11: 'site_3',
                                         12: 'site_1', 13: 'site_1'},
                                'species': {0: 'Species_C', 1: 'Species_B', 2: 'Species_A', 3: 'Species_A',
                                            4: 'Species_A', 5: 'Species_A', 6: 'Species_A', 7: 'Species_A',
                                            8: 'Species_B', 9: 'Species_C', 10: 'Species_A', 11: 'Species_B',
                                            12: 'Species_A', 13: 'Species_C'},
                                'TOT': {0: 5, 1: 1, 2: 4, 3: 1, 4: 1, 5: 1, 6: 1, 7: 5, 8: 1, 9: 9, 10: 13, 11: 1,
                                        12: 1, 13: 1},
                                'inf_status': {0: 'negative', 1: 'negative', 2: 'positive', 3: 'positive',
                                               4: 'positive', 5: 'positive', 6: 'positive', 7: 'negative',
                                               8: 'negative', 9: 'negative', 10: 'negative', 11: 'negative',
                                               12: 'negative', 13: 'negative'}})]
        output = [pd.DataFrame(
            {'site': {0: 'site_1', 1: 'site_2', 2: 'site_3'}, 'Species_A_negative': {0: 1, 1: 5, 2: 13},
             'Species_A_positive': {0: 1, 1: 1, 2: 6}, 'Species_B_negative': {0: 1, 1: 1, 2: 1},
             'Species_C_negative': {0: 1, 1: 9, 2: 5}})]
        interm_1 = rf.unite(inputs[0], new_col_name='join', cols=['species', 'inf_status'])
        interm_2 = rf.filter_(interm_1, filter_expr='sample_ID != 487405')
        interm_3 = rf.select(interm_2, columns_keep=None, columns_remove=['sample_ID'])
        interm_4 = rf.spread(interm_3, columns='join', values='TOT')
        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.unite', [-1]), ('rf.filter', [1]), ('rf.select', [2]), ('rf.spread', [3])])
        replay_map = {'unite_new_col_name': ['join'], 'unite_cols': [['species', 'inf_status']], 'filter_eq_op': ['!='],
                      'filter_column_eq': ['sample_ID'], 'filter_value_eq': [487405],
                      'filter_mode': ['equality-inequality'], 'select_columns_keep': [None],
                      'select_columns_remove': [['sample_ID']], 'select_keep_or_remove': [False],
                      'spread_columns': ['join'], 'spread_values': ['TOT'],
                      'func_seq': ['rf.unite', 'rf.filter', 'rf.select', 'rf.spread']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_94(self):
        b_id = "b_94"
        inputs = [pd.DataFrame({'ID': {0: 'C', 1: 'D', 2: 'E', 3: 'F', 4: 'G', 5: 'H', 6: 'I', 7: 'J'},
                                'c_Al': {0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0},
                                'c_D': {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                'c_Hy': {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1},
                                'occ': {0: 2581, 1: 1917, 2: 2708, 3: 2751, 4: 1522, 5: 657, 6: 469, 7: 2629}})]
        output = [pd.DataFrame(
            {'X0': {0: 1965.8333329999998, 1: 2402.0, 2: 1643.333333}, 'X1': {0: 1719.5, 1: 1605.6, 2: 2060.8}})]

        interm_1 = rf.gather(inputs[0], var_name='k', value_name='v', value_vars=['c_Al', 'c_D', 'c_Hy'], id_vars=None)
        interm_2 = rf.group_by(interm_1, group_cols=['k', 'v'])
        interm_3 = rf.summarise(interm_2, summaries={'m': ('occ', 'mean')})
        interm_4 = rf.spread(interm_3, columns='v', values='m')

        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.group_by', [1]), ('rf.spread', [2])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['k'], 'gather_value_name': ['v'],
                      'gather_value_vars': [['c_Al', 'c_D', 'c_Hy']], 'gather_id_vars': [['ID', 'occ']],
                      'group_by_group_cols': [['k', 'v']], 'summarise_new_col': ['m'], 'summarise_col': ['occ'],
                      'spread_columns': ['v'], 'spread_values': ['m'],
                      'summarise_agg': ['mean'], 'func_seq': ['rf.gather', 'rf.group_by', 'rf.summarise', 'rf.spread']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_95(self):
        b_id = "b_95"
        inputs = [pd.DataFrame({'ID': {0: 'A', 1: 'B', 2: 'C'}, 'name': {0: 'Tom', 1: 'Jim', 2: 'Kate'},
                                'MGW.one': {0: 10.0, 1: -13.29, 2: -6.95}, 'MGW.two': {0: 19, 1: 13, 2: 10},
                                'HEL.one': {0: 12, 1: 12, 2: 15}, 'HEL.two': {0: 13.0, 1: -0.12, 2: 4.0}})]
        output = [pd.DataFrame(
            {'ID': {0: 'A', 1: 'B', 2: 'C'}, 'HEL': {0: 12.5, 1: 5.94, 2: 9.5}, 'MGW': {0: 14.5, 1: -0.145, 2: 1.525}})]
        interm_1 = rf.select(inputs[0], columns_keep=None, columns_remove=['name'])
        interm_2 = rf.gather(interm_1, var_name='k', value_name='v',
                             value_vars=['MGW.one', 'MGW.two', 'HEL.one', 'HEL.two'], id_vars=['ID'])
        interm_3 = rf.separate(interm_2, split_col='k', into=["col1", "col2"])
        interm_4 = rf.group_by(interm_3, group_cols=['ID', 'col1'])
        interm_5 = rf.summarise(interm_4, summaries={'v': ('v', 'mean')})
        interm_6 = rf.spread(interm_5, columns='col1', values='v')
        output = interm_6
        intermediates = [interm_1, interm_2, interm_3, interm_4, interm_5, interm_6]
        skeleton = Skeleton(
            [('rf.select', [-1]), ('rf.gather', [1]), ('rf.separate', [2]), ('rf.group_by', [3]), ('rf.spread', [4])])
        replay_map = {'select_columns_keep': [None], 'select_columns_remove': [['name']],
                      'select_keep_or_remove': [False], 'gather_df': ['interm_1'], 'gather_var_name': ['k'],
                      'gather_value_name': ['v'], 'gather_value_vars': [['MGW.one', 'MGW.two', 'HEL.one', 'HEL.two']],
                      'gather_id_vars': [['ID']], 'separate_split_col': ['k'], 'separate_into': ['col1', 'col2'],
                      'group_by_group_cols': [['ID', 'col1']], 'summarise_new_col': ['v'], 'summarise_col': ['v'],
                      'summarise_agg': ['mean'], 'spread_columns': ['col1'], 'spread_values': ['v'],
                      'func_seq': ['rf.select', 'rf.gather', 'rf.separate', 'rf.group_by', 'rf.summarise', 'rf.spread']}

        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_96(self):
        b_id = "b_96"
        inputs = [pd.DataFrame(
            {'Subject': {0: 'A-pre', 1: 'A-post', 2: 'B-pre', 3: 'B-post'}, 'Var1': {0: 1, 1: 2, 2: 3, 3: 4},
             'Var2': {0: 8, 1: 9, 2: 10, 3: 11}, 'Var3': {0: 20, 1: 21, 2: 27, 3: 26}})]
        output = [pd.DataFrame(
            {'Col1': {0: 'A', 1: 'B'}, 'Var1_pre': {0: 1, 1: 3}, 'Var2_post': {0: 9, 1: 11}, 'Var2_pre': {0: 8, 1: 10},
             'Var3_post': {0: 21, 1: 26}, 'Var3_pre': {0: 20, 1: 27}})]

        interm_1 = rf.separate(inputs[0], split_col='Subject', into=["Col1", "Col2"])
        interm_2 = rf.gather(interm_1, var_name='k', value_name='v', value_vars=['Var1', 'Var2', 'Var3'],
                             id_vars=['Col1', 'Col2'])
        interm_3 = rf.unite(interm_2, new_col_name='key', cols=['k', 'Col2'])
        interm_4 = rf.spread(interm_3, columns='key', values='v')
        interm_5 = rf.select(interm_4, columns_keep=None, columns_remove=['Var1_post'])
        output = interm_5
        output = interm_5
        intermediates = [interm_1, interm_2, interm_3, interm_4, interm_5]
        skeleton = Skeleton(
            [('rf.separate', [-1]), ('rf.gather', [1]), ('rf.unite', [2]), ('rf.spread', [3]), ('rf.select', [4])])
        replay_map = {'separate_split_col': ['Subject'], 'separate_into': ['Col1', 'Col2'], 'gather_df': ['interm_1'],
                      'gather_var_name': ['k'], 'gather_value_name': ['v'],
                      'gather_value_vars': [['Var1', 'Var2', 'Var3']], 'gather_id_vars': [['Col1', 'Col2']],
                      'unite_new_col_name': ['key'], 'unite_cols': [['k', 'Col2']], 'spread_columns': ['key'],
                      'spread_values': ['v'], 'select_columns_keep': [None], 'select_columns_remove': [['Var1_post']],
                      'select_keep_or_remove': [False],
                      'func_seq': ['rf.separate', 'rf.gather', 'rf.unite', 'rf.spread', 'rf.select']}

        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_97(self):
        b_id = "b_97"
        inputs = [pd.DataFrame({'Title': {0: 'Carrie', 1: 'Fried-Green-Tomatoes', 2: 'Amadeus', 3: 'Amityville-Horror',
                                          4: 'Dracula', 5: 'Speed', 6: 'Zootopia', 7: 'BreakingBad'},
                                'Rating': {0: 4, 1: 2, 2: 5, 3: 1, 4: 2, 5: 4, 6: 5, 7: 5},
                                'Action': {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1},
                                'Sci.Fi': {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0},
                                'Animation': {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0},
                                'Crime': {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1}})]
        output = [pd.DataFrame({'genre': {0: 'Action', 1: 'Animation', 2: 'Crime', 3: 'Sci.Fi'},
                                'average': {0: 4.6666667, 1: 4.0, 2: 3.75, 3: 3.0}})]

        interm_1 = rf.gather(inputs[0], var_name='genre', value_name='value',
                             value_vars=['Action', 'Sci.Fi', 'Animation', 'Crime'], id_vars=['Title', 'Rating'])
        interm_2 = rf.filter_(interm_1, filter_expr='value == 1')
        interm_3 = rf.group_by(interm_2, group_cols=['genre'])
        interm_4 = rf.summarise(interm_3, summaries={'Rating': ('Rating', 'mean')})
        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.filter', [1]), ('rf.group_by', [2])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['genre'], 'gather_value_name': ['value'],
                      'gather_value_vars': [['Action', 'Sci.Fi', 'Animation', 'Crime']],
                      'gather_id_vars': [['Title', 'Rating']], 'filter_eq_op': ['=='], 'filter_column_eq': ['value'],
                      'filter_value_eq': [1], 'filter_mode': ['equality-inequality'],
                      'group_by_group_cols': [['genre']], 'summarise_new_col': ['Rating'], 'summarise_col': ['Rating'],
                      'summarise_agg': ['mean'], 'func_seq': ['rf.gather', 'rf.filter', 'rf.group_by', 'rf.summarise']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_98(self):

        b_id = "b_98"

        inputs = [pd.DataFrame({'Title': {0: 'Carrie', 1: 'Fried-Green-Tomatoes', 2: 'Amadeus', 3: 'Amityville-Horror',
                                          4: 'Dracula', 5: 'Speed', },
                                'Rating': {0: 4, 1: 2, 2: 5, 3: 1, 4: 2, 5: 4},
                                'Action': {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1},
                                'Sci.Fi': {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0},
                                'Animation': {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0}})]
        interm_1 = rf.gather(inputs[0], var_name='genre', value_name='value',
                             value_vars=['Action', 'Sci.Fi', 'Animation'], id_vars=['Title', 'Rating'])
        interm_2 = rf.filter_(interm_1, filter_expr='value == 1')
        interm_3 = rf.group_by(interm_2, group_cols=['genre'])
        interm_4 = rf.summarise(interm_3, summaries={'Rating': ('Rating', 'mean')})

        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.filter', [1]), ('rf.group_by', [2])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['genre'], 'gather_value_name': ['value'],
                      'gather_value_vars': [['Action', 'Sci.Fi', 'Animation']], 'gather_id_vars': [['Title', 'Rating']],
                      'filter_eq_op': ['=='], 'filter_column_eq': ['value'], 'filter_value_eq': [1],
                      'filter_mode': ['equality-inequality'], 'group_by_group_cols': [['genre']],
                      'summarise_new_col': ['Rating'], 'summarise_col': ['Rating'], 'summarise_agg': ['mean'],
                      'func_seq': ['rf.gather', 'rf.filter', 'rf.group_by', 'rf.summarise']}
        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_99(self):
        b_id = "b_99"
        inputs = [pd.DataFrame(
            {'id': {0: 1, 1: 2}, 'time': {0: '2009-01-01', 1: '2009-01-02'}, 'Q1.1': {0: 0.48742885, 1: 0.73832471},
             'Q1.2': {0: -0.01618826, 1: 0.94383621}, 'Q2.1': {0: 1.52718069, 1: -0.40380049},
             'Q2.2': {0: -0.29177677, 1: -1.19813815}, 'Q3.1': {0: 0.48742885, 1: 0.73832471},
             'Q3.2': {0: -0.01618826, 1: 0.94383621}, 'Q4.1': {0: 1.52718069, 1: -0.40380049},
             'Q4.2': {0: -0.29177677, 1: -1.19813815}})]
        output = [pd.DataFrame({'id': {0: 1, 1: 1, 2: 2, 3: 2},
                                'time': {0: '2009-01-01', 1: '2009-01-01', 2: '2009-01-02', 3: '2009-01-02'},
                                'Q1': {0: 0.48742885, 1: -0.01618826, 2: 0.73832471, 3: 0.94383621},
                                'Q2': {0: 1.52718069, 1: -0.29177677, 2: -0.40380049, 3: -1.19813815},
                                'Q3': {0: 0.48742885, 1: -0.01618826, 2: 0.73832471, 3: 0.94383621},
                                'Q4': {0: 1.52718069, 1: -0.29177677, 2: -0.40380049, 3: -1.19813815}})]
        interm_1 = rf.gather(inputs[0], var_name='key', value_name='val',
                             value_vars=['Q1.1', 'Q1.2', 'Q2.1', 'Q2.2', 'Q3.1', 'Q3.2', 'Q4.1', 'Q4.2'],
                             id_vars=['id', 'time'])
        interm_2 = rf.separate(interm_1, split_col='key', into=["col1", "col2"])
        interm_3 = rf.spread(interm_2, columns='col1', values='val')
        interm_4 = rf.select(interm_3, columns_keep=None, columns_remove=['col2'])

        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.separate', [1]), ('rf.spread', [2]), ('rf.select', [3])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['key'], 'gather_value_name': ['val'],
                      'gather_value_vars': [['Q1.1', 'Q1.2', 'Q2.1', 'Q2.2', 'Q3.1', 'Q3.2', 'Q4.1', 'Q4.2']],
                      'gather_id_vars': [['id', 'time']], 'separate_split_col': ['key'],
                      'separate_into': ['col1', 'col2'], 'spread_columns': ['col1'], 'spread_values': ['val'],
                      'select_columns_keep': [None], 'select_columns_remove': [['col2']],
                      'select_keep_or_remove': [False],
                      'func_seq': ['rf.gather', 'rf.separate', 'rf.spread', 'rf.select']}

        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)

    def test_100(self):
        b_id = "b_100"
        inputs = [pd.DataFrame(
            {'id': {0: 1, 1: 2}, 'time': {0: '2009-01-01', 1: '2009-01-02'}, 'Q1.1': {0: 0.48742885, 1: 0.73832471},
             'Q1.2': {0: -0.01618826, 1: 0.94383621}, 'Q2.1': {0: 1.52718069, 1: -0.40380049},
             'Q2.2': {0: -0.29177677, 1: -1.19813815}, 'Q3.1': {0: 0.48742885, 1: 0.73832471},
             'Q3.2': {0: -0.01618826, 1: 0.94383621}})]
        output = [pd.DataFrame({'id': {0: 1, 1: 1, 2: 2, 3: 2},
                                'time': {0: '2009-01-01', 1: '2009-01-01', 2: '2009-01-02', 3: '2009-01-02'},
                                'Q1': {0: 0.48742885, 1: -0.01618826, 2: 0.73832471, 3: 0.94383621},
                                'Q2': {0: 1.52718069, 1: -0.29177677, 2: -0.40380049, 3: -1.19813815},
                                'Q3': {0: 0.48742885, 1: -0.01618826, 2: 0.73832471, 3: 0.94383621}})]
        interm_1 = rf.gather(inputs[0], var_name='key', value_name='val',
                             value_vars=['Q1.1', 'Q1.2', 'Q2.1', 'Q2.2', 'Q3.1', 'Q3.2'], id_vars=['id', 'time'])
        interm_2 = rf.separate(interm_1, split_col='key', into=["col1", "col2"])
        interm_3 = rf.spread(interm_2, columns='col1', values='val')
        interm_4 = rf.select(interm_3, columns_keep=None, columns_remove=['col2'])
        output = interm_4
        intermediates = [interm_1, interm_2, interm_3, interm_4]
        skeleton = Skeleton([('rf.gather', [-1]), ('rf.separate', [1]), ('rf.spread', [2]), ('rf.select', [3])])
        replay_map = {'gather_df': ['inputs[0]'], 'gather_var_name': ['key'], 'gather_value_name': ['val'],
                      'gather_value_vars': [['Q1.1', 'Q1.2', 'Q2.1', 'Q2.2', 'Q3.1', 'Q3.2']],
                      'gather_id_vars': [['id', 'time']], 'separate_split_col': ['key'],
                      'separate_into': ['col1', 'col2'], 'spread_columns': ['col1'], 'spread_values': ['val'],
                      'select_columns_keep': [None], 'select_columns_remove': [['col2']],
                      'select_keep_or_remove': [False],
                      'func_seq': ['rf.gather', 'rf.separate', 'rf.spread', 'rf.select']}

        stats = {}

        return MorpheusBenchmark(b_id, inputs, intermediates, output, "", skeleton, stats, replay_map)
