"""
The definition of the domain. Establishes the interface with the main synthesis engine.
"""
from typing import List, Tuple, Dict, Any, Iterator, Iterable, Optional, Set

import attr
import autopep8
import pandas as pd

from gauss.domains.rlang.checker import check_result
from gauss.domains.rlang.datagen import datagen_dict
from gauss.domains.rlang.generators import generator_dict
from gauss.domains.rlang.graphs import DataFrameGraph, ELabel, NLabel
from gauss.domains.rlang.strategies import RandomizedGraphStrategy, DfsGraphStrategy, IntelligentEnumerationStrategy
from gauss.graphs import Graph, Node, equality_transitive_closure
from gauss.synthesis.domains import SynthesisDomain, EnumerationItem
from gauss.synthesis.witness_set import WitnessEntry


@attr.s
class RLangEnumerationItem(EnumerationItem):
    call_str: str = attr.ib()
    placeholders: Dict[str, str] = attr.ib(factory=dict)


@attr.s
class RLangSynthesisDomain(SynthesisDomain):
    _nlabel_map: Dict[int, str] = attr.ib(init=False, default={n.value: n.name for n in NLabel})
    _elabel_map: Dict[int, str] = attr.ib(init=False, default={e.value: e.name for e in ELabel})
    _valid_closure_combinations: Set[int] = attr.ib(init=False,
                                                    default={e.value for e in ELabel} - {ELabel.COLUMN, ELabel.ROW})

    def get_available_components(self) -> List[str]:
        """
        Returns:
            List[str]: The names of components available in this domain as a list of strings.
        """
        return list(generator_dict.keys())

    def generate_witness_entry(self, component_name: str, seed: int) -> WitnessEntry:
        """
        Generates an entry for the witness set for the given component (referenced by its name)
        Args:
            component_name (str): The name of the component to generate the entry for.
            seed (int): A seed that can be used to direct the generation (if a random process is involved).

        Returns:
            WitnessEntry: The generated entry.
        """

        inputs, output, program, graph = _generate_example(component_name)
        return WitnessEntry(inputs=inputs, output=output, program=program, graph=graph)

    def enumerate(self, component_name: str, inputs: List[Any], g_inputs: List[Graph],
                  replay: Dict[str, Iterable[Any]] = None,
                  **kwargs) -> Iterator[RLangEnumerationItem]:
        """
        Enumerate all possible programs (with output and graph abstraction) for the given component and
        inputs (along with the graph abstractions of the inputs).

        Args:
            component_name:
            inputs:
            g_inputs:
            replay: Replay map to use for the generators, if any.
            **kwargs:

        Returns:

        """
        gen = generator_dict[component_name]
        strategy = DfsGraphStrategy()

        if 'constants' in kwargs:
            constants = kwargs['constants'] or []
        else:
            constants = []

        if 'strengthening_constraint' in kwargs:
            #  We are in synthesis mode and have been supplied the strengthening constraint.
            #  We can use this to greatly prune down the argument space.
            strategy = IntelligentEnumerationStrategy(strengthening_constraint=kwargs['strengthening_constraint'])

        for output, call_str, graph, o_graph in gen.with_env(ignore_exceptions=False,
                                                             strategy=strategy,
                                                             replay=replay).generate(*inputs, *g_inputs):
            yield RLangEnumerationItem(
                output=output,
                graph=graph,
                o_graph=o_graph,
                call_str=call_str
            )

    def prepare_solution(self,
                         enumeration_items: List[RLangEnumerationItem],
                         arguments: List[List[int]],
                         int_to_names: Dict[int, str],
                         int_to_obj: Dict[int, Any]):
        program_lines = []

        for depth, (item, arg_ints) in enumerate(zip(enumeration_items, arguments), 1):
            call_str = item.call_str

            #  Depth is 1-indexed here.
            arg_strs = [int_to_names[i] if i < 0 else f"v{i}" for i in arg_ints]
            call_str = call_str.format(**{f"inp{idx}": arg_str for idx, arg_str in enumerate(arg_strs, 1)})
            if depth == len(enumeration_items):
                if int_to_names.get(depth, None) is None:
                    program_lines.append(call_str)
                else:
                    program_lines.append(f"{int_to_names[depth]} = {call_str}")

            else:
                program_lines.append(f"v{depth} = {call_str}")

        return autopep8.fix_code("\n".join(program_lines), options={'aggressive': 10})

    def check_equivalent(self, target: Any, actual: Any, **kwargs):
        """
        Check if the actual output is equivalent to the target output.
        Args:
            target:
            actual:

        Returns:

        """

        return self.check_partial(target, actual) and self.check_partial(actual, target)

    def check_partial(self, target: Any, actual: Any, **kwargs):
        """
        Check if the actual output *subsumes* the target output.
        Args:
            target:
            actual:

        Returns:

        """

        return check_result(target, actual, ignore_row_ordering=True)

    def get_equality_edge_label(self) -> Optional[int]:
        return ELabel.EQUAL

    def perform_transitive_closure(self, graph: Graph, join_nodes: Optional[Set[Node]] = None) -> None:
        """
        The closure is performed only w.r.t the equality edge in this domain.
        Args:
            graph: The graph to perform the transitive closure on.
            join_nodes: The set of nodes to restrict joins to, if any.

        Returns:

        """

        equality_transitive_closure(graph, ELabel.EQUAL,
                                    join_nodes=join_nodes,
                                    valid_combinations=self._valid_closure_combinations)

    def get_node_label_map(self) -> Dict[int, str]:
        return self._nlabel_map

    def get_edge_label_map(self) -> Dict[int, str]:
        return self._elabel_map


def _generate_example(component_name: str) -> Tuple[List[pd.DataFrame], pd.DataFrame, str, Graph]:
    while True:
        try:
            inputs, replay_map = datagen_dict[component_name]()
            #  Abstractions for the individual inputs.
            g_inputs = [DataFrameGraph(i) for i in inputs]
            strategy = RandomizedGraphStrategy()
            gen = generator_dict[component_name]
            output, program, graph, output_graph = gen.with_env(strategy=strategy, replay=replay_map).call(*inputs,
                                                                                                           *g_inputs,
                                                                                                           datagen=True)
            if 0 in output.shape:
                raise AssertionError("Got empty dataframe")

            #  Populate the placeholders
            program = program.format(**{f"inp{i}": f"inp{i}" for i in range(1, len(inputs) + 1)})
            return inputs, output, program, graph

        except Exception as e:
            pass
