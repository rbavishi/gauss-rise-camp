"""
The definition of the domain. Establishes the interface with the main synthesis engine.
"""
import collections
from typing import List, Tuple, Dict, Any, Iterator, Iterable, Optional, Set, AbstractSet

import attr
import autopep8
import pandas as pd

from gauss.domains.pandas_lite.checker import check_result
from gauss.domains.pandas_lite.datagen import datagen_dict
from gauss.domains.pandas_lite.generators import generator_dict
from gauss.domains.pandas_lite.graphs import ELabel, NLabel, DataFrameGraph, GraphPandas
from gauss.domains.pandas_lite.strategies import RandomizedGraphStrategy, DfsGraphStrategy, \
    IntelligentEnumerationStrategy
from gauss.graphs import Graph, equality_transitive_closure, Node, Edge
from gauss.synthesis.domains import SynthesisDomain, SynthesisUI, EnumerationItem, Solution
from gauss.synthesis.witness_set import WitnessEntry


@attr.s
class PandasLiteEnumerationItem(EnumerationItem):
    call_str: str = attr.ib()
    placeholders: Dict[str, str] = attr.ib(factory=dict)


@attr.s
class PandasLiteSolution(Solution):
    explanations: Dict[Tuple[int, int], Tuple[List[Tuple[int, int, int]], Optional[str]]] = attr.ib()


@attr.s(cmp=False, repr=False)
class PandasLiteSynthesisDomain(SynthesisDomain):
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

        inputs, output, program, graph = _generate_example(component_name, seed=seed)
        return WitnessEntry(inputs=inputs, output=output, program=program, graph=graph)

    def enumerate(self, component_name: str, inputs: List[Any], g_inputs: List[Graph],
                  replay: Dict[str, Iterable[Any]] = None,
                  **kwargs) -> Iterator[EnumerationItem]:
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

        for output, call_str, graph, o_graph in gen.with_env(ignore_exceptions=True,
                                                             strategy=strategy,
                                                             replay=replay).generate(*inputs, *g_inputs,
                                                                                     constants=constants):
            yield PandasLiteEnumerationItem(
                output=output,
                graph=graph,
                o_graph=o_graph,
                call_str=call_str,
                placeholders=strategy.placeholders.copy()
            )

    def prepare_solution(self,
                         inputs: List[Any],
                         output: Any,
                         graph: Graph,
                         graph_inputs: List[DataFrameGraph],
                         graph_output: DataFrameGraph,
                         enumeration_items: List[PandasLiteEnumerationItem],
                         arguments: List[List[int]],
                         int_to_names: Dict[int, str],
                         int_to_obj: Dict[int, Any]) -> PandasLiteSolution:
        program_lines = []
        placeholders: Dict[str, str] = {}
        call_strs: List[str] = []
        prog_length = len(enumeration_items)

        assert all(isinstance(i, pd.DataFrame) for i in int_to_obj.values())
        used_constants = set()
        for i in int_to_obj.values():
            used_constants.update(i.values.flatten().astype(str))
            used_constants.update(str(c) for c in i.columns)

        placeholder_mappings: Dict[str, str] = {}
        for item in enumeration_items:
            call_strs.append(item.call_str)
            placeholders.update(item.placeholders)

            for k, v in item.placeholders.items():
                if k in item.call_str:
                    idx = 0
                    while True:
                        new_name = v if idx == 0 else f"{v}{idx}"
                        if new_name not in used_constants:
                            placeholder_mappings[k] = new_name
                            used_constants.add(new_name)
                            break

                        idx += 1

        for idx, call_str in enumerate(call_strs):
            changed = True
            while changed:
                changed = False
                old_call_str = call_str
                for k, v in placeholder_mappings.items():
                    call_str = call_str.replace(k, v)

                changed = call_str != old_call_str

            call_strs[idx] = call_str

        for depth, (call_str, arg_ints) in enumerate(zip(call_strs, arguments), 1):
            #  Depth is 1-indexed here.
            arg_strs = [int_to_names[i] if i < 0 else f"v{i}" for i in arg_ints]
            call_str = call_str.format(**{f"inp{idx}": arg_str for idx, arg_str in enumerate(arg_strs, 1)})
            if depth == prog_length:
                if int_to_names.get(depth, None) is None:
                    program_lines.append(call_str)
                else:
                    program_lines.append(f"{int_to_names[depth]} = {call_str}")

            else:
                program_lines.append(f"v{depth} = {call_str}")

        #  Recompute the output if necessary
        if len(placeholder_mappings) > 0:
            int_to_obj = int_to_obj.copy()
            for depth in range(1, prog_length + 1):
                call_str = call_strs[depth - 1]
                arg_ints = arguments[depth - 1]
                arg_dict = globals()
                arg_dict.update({f"inp{idx}": int_to_obj[i] for idx, i in enumerate(arg_ints, 1)})
                call_str = call_str.format(**{f"inp{idx}": f"inp{idx}" for idx in range(1, len(arg_ints) + 1)})
                int_to_obj[depth] = eval(call_str, arg_dict)

        code, final_output = autopep8.fix_code("\n".join(program_lines),
                                               options={'aggressive': 10}), int_to_obj[prog_length]

        node_to_code_mapping = {}
        for idx, i_g in enumerate(graph_inputs):
            for c, node in enumerate(i_g.columns):
                node_to_code_mapping[node] = (idx, -1, c)

            for r, row in enumerate(i_g.values):
                for c, node in enumerate(row):
                    node_to_code_mapping[node] = (idx, r, c)

        inp_nodes = set(node_to_code_mapping.keys())

        explanations: Dict[Tuple[int, int], Tuple[List[Tuple[int, int, int]], Optional[str]]] = {}
        for c, node in enumerate(graph_output.columns):
            expr_str = _get_explanation_expr_str(graph, node, self._nlabel_map, self._elabel_map)
            involved_nodes = _get_involved_nodes(graph, node, self._nlabel_map, self._elabel_map)
            involved_nodes = [node_to_code_mapping[k] for k in involved_nodes & inp_nodes]
            explanations[-1, c] = (involved_nodes, expr_str)

        for r, row in enumerate(graph_output.values):
            for c, node in enumerate(row):
                expr_str = _get_explanation_expr_str(graph, node, self._nlabel_map, self._elabel_map)
                involved_nodes = _get_involved_nodes(graph, node, self._nlabel_map, self._elabel_map)
                involved_nodes = [node_to_code_mapping[k] for k in involved_nodes & inp_nodes]
                explanations[r, c] = (involved_nodes, expr_str)

        return PandasLiteSolution(
            code=code,
            output=final_output,
            explanations=explanations,
        )

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

    def rank_sequences(self, sequences: List[List[str]]) -> List[List[str]]:
        return sorted(sequences,
                      key=lambda x: -{
                          ('pd.drop_columns',): 3,
                          ('pd.groupby_agg',): 2,
                          ('pd.groupby_transform',): 1,
                          ('pd.melt', 'pd.groupby_transform'): 1,
                          ('pd.melt', 'pd.groupby_agg'): 2,
                          ('pd.mutate', 'pd.drop_columns'): 2,
                      }.get(tuple(x), 0))


@attr.s(cmp=False, repr=False)
class PandasLiteSynthesisUI(SynthesisUI):
    AGGREGATIONS = {
        "SUM": "Compute the sum of selected values.",
        "MEAN": "Compute the mean/average of selected values.",
        "MEDIAN": "Compute the median of selected values.",
        "PROD": "Compute the product of selected values.",
        "MIN": "Compute the minimum of selected values.",
        "MAX": "Compute the maximum of selected values.",
        "COUNT": "Count the number of selected values.",
        "COUNT_NON_NULL": "Count the number of non-null/non-NaN selected values.",
        "NUNIQUE": "Count the number of unique selected values.",
        "ANY": "True if at least one selected value is/evaluates to True. False otherwise.",
        "ALL": "True if all selected value are/evaluate to True. False otherwise.",
    }

    CUMULATIVE_TRANSFORMS = {
        "CUMSUM": "Compute the cumulative sum of the selected values.",
        "CUMPROD": "Compute the cumulative product of the selected values.",
        "CUMCOUNT": "Compute the cumulative count (non-null) of the selected values.",
        "CUMMIN": "Compute the cumulative min of the selected values.",
        "CUMMAX": "Compute the cumulative max of the selected values.",
    }

    FILTERING = {
        "CONTAINED_IN": "Keep rows where the column value is contained in the specified set of values.",
        "NOT_CONTAINED_IN": "Remove rows where the column value is contained in the specified set of values.",
        "LESS_THAN": "Keep rows where the column value is less than the specified value.",
        "GREATER_THAN": "Keep rows where the column value is greater than the specified value.",
        "EQUAL_TO": "Keep rows where the column value is equal to the specified value.",
        "NOT_EQUAL_TO": "Remove rows where the column value is equal to the specified value.",
        "CUSTOM": "Keep rows according to a custom expression. Use x as variable holding column value.",
    }

    FILTERING_DIALOGS = {
        "CONTAINED_IN": "Enter set of values. (Eg. {1, \\'a\\', 3})",
        "NOT_CONTAINED_IN": "Enter set of values. (Eg. {1, \\'a\\', 3})",
        "LESS_THAN": "Enter value (int/float).",
        "GREATER_THAN": "Enter value (int/float).",
        "EQUAL_TO": "Enter value. (Eg. \\'abcd\\')",
        "NOT_EQUAL_TO": "Enter value. (Eg. \\'abcd\\')",
        "CUSTOM": "Enter expression. Use `x` (with backticks) as variable holding value."
                  "\\n(Eg. `x` == \\'a\\' OR x == 10)."
    }

    BINARY_OPS = {
        "ADD": "Add the two selected values.",
        "SUB": "Subtract the second selected value from the first selected value.",
        "MUL": "Multiply the two values.",
        "DIV": "Divide the first selected value by the second selected value.",
    }

    STRING_OPS = {
        "STR_JOIN(_)": "Concatenate the selected values with underscore as a separator.",
        "STR_SPLIT": "Split the selected value with separators as any group of non-alpha-numeric characters."
    }

    def get_available_operations(self) -> List[Dict[str, Any]]:
        aggregations = [
            {'name': k, 'description': v} for k, v in self.AGGREGATIONS.items()
        ]

        cumulative_transformations = [
            {'name': k, 'description': v} for k, v in self.CUMULATIVE_TRANSFORMS.items()
        ]

        filtering = [
            {'name': k, 'description': v, 'dialog': self.FILTERING_DIALOGS.get(k, "")}
            for k, v in self.FILTERING.items()
        ]

        binary_ops = [
            {'name': k, 'description': v, 'arity': 2}
            for k, v in self.BINARY_OPS.items()
        ]

        string_ops = [
            {'name': "STR_JOIN(_)", "description": self.STRING_OPS["STR_JOIN(_)"]},
            {'name': "STR_SPLIT", "description": self.STRING_OPS["STR_SPLIT"]},
        ]

        return [
            {
                'name': 'Aggregations',
                'children': [
                    *aggregations
                ],
            },
            {
                'name': 'Transformations',
                'children': [
                    *cumulative_transformations
                ],
            },
            {
                'name': 'Binary Operations',
                'arity': 2,
                'children': [
                    *binary_ops
                ],
            },
            {
                'name': 'String Operations',
                'children': [
                    *string_ops
                ],
            },
            {
                'name': 'Filter',
                'enabledFor': 'SINGLE_COLUMN',
                'children': [
                    *filtering
                ]
            },
            {
                'name': 'Mark as Deleted',
                'enabledFor': 'ONLY_INPUTS',
                'builtin': 'DELETE',
                'description': 'Mark selected cells to be deleted while constructing the output.'
            }
        ]

    def perform_operation(self, operation: str, inputs: List[Tuple[Dict, Any]],
                          obj_store: Dict[str, Any],
                          trace_store: Dict[str, Dict],
                          kwargs: Dict[str, Any]):

        v_traces, values = list(zip(*inputs))

        if operation in self.AGGREGATIONS:
            series = pd.Series(values)
            if operation == "COUNT":
                result = len(series)

            elif operation == "COUNT_NON_NULL":
                result = series.count()

            else:
                result = getattr(series, operation.lower())()

            return {
                'traces': [[{
                    "from": v_traces,
                    "labels": [operation.upper()],
                    "value": result,
                }]]
            }

        elif operation in self.CUMULATIVE_TRANSFORMS:
            if operation == 'CUMCOUNT':
                result = list(pd.Series(values).expanding().count())

            else:
                result = list(getattr(pd.Series(values), operation.lower())())

            result_traces = []
            for idx, element in enumerate(result):
                result_traces.append([{
                    "from": v_traces[:idx + 1] + v_traces[:idx + 1],
                    "labels": [operation.upper()] * (idx + 1) + [operation[3:].upper()] * (idx + 1),
                    "value": element
                }])

            return {
                'traces': result_traces
            }

        elif operation in self.BINARY_OPS:
            result = getattr(pd.Series([values[0]]), operation.lower())(pd.Series([values[1]]))[0]
            label = {
                "ADD": "SUM",
                "MUL": "PROD",
                "SUB": "SUB",
                "DIV": "DIV"
            }[operation]

            return {
                'traces': [[{
                    "from": v_traces,
                    "labels": [label],
                    "value": result,
                }]]
            }

        elif operation in self.STRING_OPS:
            if operation.startswith("STR_JOIN"):
                result = "_".join(map(str, values))

                return {
                    'traces': [[{
                        "from": v_traces,
                        "labels": ["STR_JOIN"],
                        "value": result,
                    }]]
                }

            else:
                results = pd.Series(values).astype(str).str.split(r'[^A-Za-z0-9]+')

                result_traces = []
                for result, v_trace in zip(results, v_traces):
                    for idx, element in enumerate(result):
                        result_traces.append([{
                            "from": [v_trace],
                            "labels": ["STR_SPLIT"],
                            "value": element
                        }])

                return {
                    'traces': result_traces
                }

        elif operation in self.FILTERING:
            assert len(v_traces) == len(values) == 1
            grid_id = v_traces[0]['from'][0].split(':')[-1]
            src_df = obj_store[grid_id]
            col = values[0]

            result_traces = []

            if "CONTAINED" in operation:
                #  TODO : Security Check
                try:
                    set_val = eval(kwargs['user_input'])
                except Exception as e:
                    raise AssertionError("Could not interpret set of values.")

                if not isinstance(set_val, AbstractSet):
                    raise AssertionError("Input value must be a set.")

                if operation == 'CONTAINED_IN':
                    retained_indices = [idx for idx, elem in enumerate(list(src_df[col].isin(set_val)))
                                        if elem is True]
                else:
                    retained_indices = [idx for idx, elem in enumerate(list(~src_df[col].isin(set_val)))
                                        if elem is True]

                col_id = list(src_df.columns).index(col)
                for row_id in retained_indices:
                    result_traces.append([])
                    for col_idx in range(src_df.shape[1]):
                        result_traces[-1].append({
                            "from": [trace_store[f"{row_id}:{col_idx}:{grid_id}"],
                                     trace_store[f"{row_id}:{col_id}:{grid_id}"]],
                            "labels": ["EQUAL", operation],
                            "value": src_df.iloc[row_id, col_idx],
                        })

                return {
                    'traces': result_traces,
                    'constants': [set_val]
                }

            else:

                if operation == 'LESS_THAN':
                    expr = f"`{col}` < {kwargs['user_input']}"
                elif operation == "GREATER_THAN":
                    expr = f"`{col}` > {kwargs['user_input']}"
                elif operation == "EQUAL_TO":
                    expr = f"`{col}` == {kwargs['user_input']}"
                elif operation == "NOT_EQUAL_TO":
                    expr = f"`{col}` != {kwargs['user_input']}"
                elif operation == "CUSTOM":
                    expr = kwargs['user_input'].replace('`x`', f"`{col}`")
                else:
                    raise AssertionError(f"Unrecognized operation {operation}.")

                try:
                    retained_indices = list(src_df.reset_index(drop=True).query(expr).index)
                except Exception as e:
                    raise AssertionError("Query invalid.")

                for row_id in retained_indices:
                    result_traces.append([])
                    for col_idx in range(src_df.shape[1]):
                        result_traces[-1].append({
                            "from": [trace_store[f"{row_id}:{col_idx}:{grid_id}"]],
                            "labels": ["FILTER_EXPR"],
                            "value": src_df.iloc[row_id, col_idx],
                        })

                return {
                    'traces': result_traces,
                    'constants': [expr]
                }

        raise AssertionError(f"Unrecognized operation {operation}.")

    def process_ui_interaction(self,
                               inputs: Dict[str, Any],
                               interactions: List[Dict]) -> Tuple[Any, Graph, Dict[str, Graph]]:

        value_interactions = [interaction for interaction in interactions if interaction['to'] != ""]
        output_cells = [[int(r), int(c)] for r, c, _ in (interaction['to'].split(':')
                                                         for interaction in value_interactions)]
        if len(output_cells) > 0:
            row_nums, col_nums = list(zip(*output_cells))
        else:
            row_nums = []
            col_nums = []

        min_r, max_r = min([i for i in row_nums if i >= 0] + [0]), max([i for i in row_nums if i >= 0] + [0])
        min_c, max_c = min(col_nums, default=0), max(col_nums, default=0)

        num_rows = max_r - min_r + 1
        num_cols = max_c - min_c + 1

        output = pd.DataFrame([[f"_CELL_{r}_{c}" for c in range(num_cols)] for r in range(num_rows)],
                              columns=[f"_COL_{c}" for c in range(num_cols)])

        columns = list(output.columns)

        for interaction in value_interactions:
            value = interaction['value']
            r, c, _ = interaction['to'].split(':')
            r = int(r)
            c = int(c)
            if r == -1:
                columns[c - min_c] = value
            else:
                output.iloc[r - min_r, c - min_c] = value

        output.columns = columns

        graph = GraphPandas()
        g_inputs: Dict[str, DataFrameGraph] = {key: DataFrameGraph(inp) for key, inp in inputs.items()}
        g_output = DataFrameGraph(output)

        for g_inp in g_inputs.values():
            graph.merge(g_inp)

        graph.merge(g_output)

        for interaction in interactions:
            if interaction["labels"] == ["DELETE"]:
                node_to = g_output.deletion_node
                for r_from, c_from, inp_id in (i.split(':') for i in interaction['from']):
                    node_from = g_inputs[inp_id].get_node_xy(int(r_from), int(c_from))
                    graph.add_edge(Edge(node_from, node_to, ELabel.DELETE))

                continue

            value = interaction['value']
            r_to, c_to, _ = interaction['to'].split(':')
            r_to = int(r_to)
            c_to = int(c_to)

            if r_to >= 0:
                r_to -= min_r
            c_to -= min_c

            if r_to == -1:
                actual_value = output.columns[c_to]
            else:
                actual_value = output.iloc[r_to, c_to]

            if _not_equal(actual_value, value):
                continue

            node_to = g_output.get_node_xy(r_to, c_to)
            node_from = _generate_computation_node(interaction, graph, g_inputs)
            graph.add_edge(Edge(node_from, node_to, ELabel.EQUAL))

        return output, graph, g_inputs


def _generate_computation_node(interaction: Dict, graph: GraphPandas, g_inputs: Dict[str, DataFrameGraph]) -> Node:
    if interaction['labels'] == ["EQUAL"]:
        r_from, c_from, inp_id = interaction['from'][0].split(':')
        return g_inputs[inp_id].get_node_xy(int(r_from), int(c_from))

    from_nodes = [_generate_computation_node(t, graph, g_inputs) for t in interaction['from']]
    interm_node = graph.create_intermediate_node(interaction['value'])
    labels = interaction['labels']
    if len(labels) == 1 and len(from_nodes) > 1:
        labels = [labels[0]] * len(from_nodes)

    for label, node in zip(labels, from_nodes):
        graph.add_edge(Edge(node, interm_node, getattr(ELabel, label)))

    return interm_node


def _generate_example(component_name: str, seed: int) -> Tuple[List[pd.DataFrame], pd.DataFrame, str, Graph]:
    while True:
        try:
            inputs, constants, replay_map = datagen_dict[component_name](seed=seed)
            #  Abstractions for the individual inputs.
            g_inputs = [DataFrameGraph(i) for i in inputs]
            strategy = RandomizedGraphStrategy()
            gen = generator_dict[component_name]
            output, program, graph, output_graph = gen.with_env(strategy=strategy, replay=replay_map).call(*inputs,
                                                                                                           *g_inputs,
                                                                                                           constants,
                                                                                                           datagen=True)
            if 0 in output.shape:
                raise AssertionError("Got empty dataframe")

            #  Populate the placeholders
            program = program.format(**{f"inp{i}": f"inp{i}" for i in range(1, len(inputs) + 1)})
            return inputs, output, program, graph

        except Exception as e:
            # import logging
            # logging.exception(e)
            pass


def _not_equal(v1, v2):
    if pd.isnull(v1) and pd.isnull(v2):
        return False

    if v1 is v2:
        return False

    if v1 == v2:
        return False

    return True


def _get_explanation_expr_str(graph: Graph, node: Node,
                              node_label_dict: Dict[int, str], edge_label_dict: Dict[int, str]) -> Optional[str]:
    args = collections.defaultdict(list)
    for edge in graph.iter_edges(dst=node):
        label = edge_label_dict[edge.label]
        if label.startswith("CUM") or label == "COLUMN" or label == "ROW":
            continue

        if node_label_dict[edge.src.label] == "INTERM":
            args[label].append(
                         _get_explanation_expr_str(graph, edge.src, node_label_dict, edge_label_dict))
        else:
            args[label].append(str(edge.src.value))

    if len(args) == 0:
        return None

    if "EQUAL" in args:
        return args["EQUAL"][0]

    key = next(iter(args.keys()))
    arg_str = ", ".join(args[key])
    return f"({key.upper()}({arg_str}))"


def _get_involved_nodes(graph: Graph, node: Node,
                        node_label_dict: Dict[int, str], edge_label_dict: Dict[int, str]) -> Set[Node]:
    result = set()
    for edge in graph.iter_edges(dst=node):
        label = edge_label_dict[edge.label]
        if label.startswith("CUM") or label == "COLUMN" or label == "ROW":
            continue

        result.add(edge.src)
        result.update(_get_involved_nodes(graph, edge.src, node_label_dict, edge_label_dict))

    return result
