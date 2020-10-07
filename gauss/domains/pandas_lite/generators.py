import collections
import itertools

import pandas as pd
import numpy as np
from atlas import generator, Generator
from typing import List, Dict, Any, Optional, Collection

from atlas.exceptions import ExceptionAsContinue

from gauss.domains.pandas_lite.graphs import DataFrameGraph, GraphPandas, ELabel, MY_NAN
from gauss.domains.pandas_lite.strategies import DfsGraphStrategy, SubsetNode, SelectNode, OrderedSubsetNode, \
    SelectConst, \
    FreshColumn
from gauss.graphs import Graph, Edge, TaggedEdge, Node

agg_key_dict = {
    "AGG_nunique": ('pd.Series.nunique', pd.Series.nunique),
}

candidates_groupby_agg_op = ["size", "count",
                             "mean", "sum", "prod",
                             "all", "any", "max", "min", "median",
                             "AGG_nunique"]

candidates_groupby_transform_op = ["cumsum", "cumprod", "cummin", "cummax", "cumcount"]

op_to_edge_label_map = {
    "size": ELabel.COUNT,
    "count": ELabel.COUNT_NON_NULL,
    "AGG_nunique": ELabel.NUNIQUE,
    **{op: getattr(ELabel, op.upper()) for op in ["sum", "mean", "prod", "all", "any", "max", "min", "median",
                                                  "cumsum", "cumprod", "cummin", "cummax", "cumcount"]}
}


def _find_fillna_replacement_node(row: int, col: int,
                                  g_df: DataFrameGraph, g_res: DataFrameGraph,
                                  method: str, axis: str) -> Optional[Node]:
    if axis == "index":
        #  Look for the replacement vertically
        if method == "backfill":
            for r in range(row + 1, g_df.shape[0]):
                if not pd.isnull(g_df.iloc[r, col].value):
                    return g_df.iloc[r, col]

        else:
            for r in range(row - 1, -1, -1):
                if not pd.isnull(g_df.iloc[r, col].value):
                    return g_df.iloc[r, col]

    else:
        #  Look for the replacement horizontally
        if method == "backfill":
            for c in range(col + 1, g_df.shape[1]):
                if not pd.isnull(g_df.iloc[row, c].value):
                    return g_df.iloc[row, c]

        else:
            for c in range(col - 1, -1, -1):
                if not pd.isnull(g_df.iloc[row, c].value):
                    return g_df.iloc[row, c]

    return None


@generator(name='pd.groupby_agg', group='pandas', strategy=DfsGraphStrategy())
def gen_groupby_agg(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    cands_by_cols = g_df.columns
    by_cols = list(SubsetNode(cands_by_cols, uid="groupby_agg_by_cols"))

    op: str = SelectConst(candidates_groupby_agg_op, uid="groupby_agg_op")

    result_groupby = df.groupby(by=by_cols, axis=0, as_index=False)

    if op.startswith("AGG_"):
        result = result_groupby.agg(agg_key_dict[op][1])
        call_str = f"{{inp1}}.groupby(by={by_cols!r}, \naxis=0, as_index=False).aggregate({agg_key_dict[op][0]})"
    else:
        result = getattr(result_groupby, op)()
        call_str = f"{{inp1}}.groupby(by={by_cols!r}, \naxis=0, as_index=False).{op}()"

    if op == "size":
        new_col = FreshColumn(prefix=f"group_sizes", uid="groupby_agg_size_new_col")
        result = result.reset_index(name=new_col)
        call_str = f"{call_str}.reset_index(name={new_col!r})"

    else:
        new_col = None

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between group columns and corresponding columns in output.
    for c in by_cols:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))

    #  - EQUAL edges between the cells of the group columns and the corresponding cells in the output.
    #  - Aggregation edges between cells of aggregated columns (or group columns in case of 'size')
    #    and the resulting cells in the output.
    agg_cols = [c for c in result.columns if c not in by_cols]
    if op == 'size':
        df_agg_cols_list = [by_cols]
        res_agg_col_list = [new_col]

    else:
        df_agg_cols_list = [[c] for c in agg_cols]
        res_agg_col_list = agg_cols
        #  Equality edge between agg columns if agg is not size
        for c in agg_cols:
            added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))

    for idx, group in enumerate(result.loc[:, by_cols].values):
        group = group[0] if len(group) == 1 else tuple(group)
        df_indices = result_groupby.groups[group]  # Get the indices in df that correspond to this group
        for df_agg_cols, res_agg_col in zip(df_agg_cols_list, res_agg_col_list):
            out_node = g_res.loc[result.index[idx], res_agg_col]
            interm_node = graph.create_intermediate_node(out_node.value)
            added_edges.append(Edge(interm_node, out_node, ELabel.EQUAL))

            #  Aggregation edges
            for col in df_agg_cols:
                for val_node in g_df.loc[df_indices, col]:
                    added_edges.append(Edge(val_node, interm_node, op_to_edge_label_map[op]))

        #  Equality edges for by_col nodes
        for col in by_cols:
            group_node = g_res.loc[result.index[idx], col]
            for df_group_node in g_df.loc[df_indices, col]:
                added_edges.append(Edge(df_group_node, group_node, ELabel.EQUAL))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  - DELETE edges between the non-group cols and non-aggregated columns and their cells
    #    to the deletion node of the output.
    #  - DELETE edge between the column of the aggregated column (in case of sum/mean) and the deletion node
    #    of the output.
    non_by_cols = [c for c in df.columns if c not in by_cols and c not in agg_cols]
    for col in non_by_cols:
        added_edges.append(Edge(col_map_df[col], g_res.deletion_node, ELabel.DELETE))
        for cell in g_df.loc[:, col]:
            added_edges.append(Edge(cell, g_res.deletion_node, ELabel.DELETE))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #
    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_by_cols:
        if c_node.value in by_cols:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@groupby_agg_by_cols"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@groupby_agg_by_cols"))

    graph.add_tags([f"SELECTED@{cand}@groupby_agg_op"
                    if cand == op else
                    f"NOT_SELECTED@{cand}@groupby_agg_op"
                    for cand in candidates_groupby_agg_op])
    graph.add_tagged_edges(tagged_edges)

    return result, call_str, graph, g_res


@generator(name='pd.groupby_transform', group='pandas', strategy=DfsGraphStrategy())
def gen_groupby_transform(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    cands_by_cols = g_df.columns
    by_cols = list(SubsetNode(cands_by_cols, uid="groupby_transform_by_cols"))

    op: str = SelectConst(candidates_groupby_transform_op, uid="groupby_transform_op")

    #  For now, one column at a time.
    #  TODO : Think about how to neatly extend to multiple columns
    cands_op_col = [c for c in g_df.columns if c.value not in by_cols]
    op_col = SelectNode(cands_op_col, uid="groupby_transform_op_col")
    new_col = FreshColumn(prefix=f"{op.lower()}_{op_col}", uid="groupby_transform_new_col")

    result_groupby = df.groupby(by=by_cols, axis=0, as_index=False)[op_col]
    if op == 'cumcount':
        #  If I die of suspicious circumstances, this API should be the prime suspect.
        result = df.assign(**{new_col: result_groupby.cumcount()})
        call_str = f"{{inp1}}.assign(**{{{{{new_col!r}: {{inp1}}.groupby(by={by_cols!r}, \naxis=0, as_index=False)[{op_col!r}]" \
                   f".{op}()}}}})"
    else:
        result = df.assign(**{new_col: result_groupby.transform(op)})
        call_str = f"{{inp1}}.assign(**{{{{{new_col!r}: {{inp1}}.groupby(by={by_cols!r}, \naxis=0, as_index=False)[{op_col!r}]" \
                   f".transform({op!r})}}}})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between all columns and cells in df to output as everything is preserved.
    for c in col_map_df:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        for c1, c2 in zip(g_df.loc[:, c], g_res.loc[:, c]):
            added_edges.append(Edge(c1, c2, ELabel.EQUAL))

    if op.startswith("cum"):
        for idx, group in enumerate(result.loc[:, by_cols].values):
            group = group[0] if len(group) == 1 else tuple(group)
            df_indices = result_groupby.groups[group]  # Get the indices in df that correspond to this group
            cum_stack = []
            for v1, v2 in zip(g_df.loc[df_indices, op_col], g_res.loc[df_indices, new_col]):
                cum_stack.append(v1)
                #  Intermediate nodes for each cumulate
                interm_node = graph.create_intermediate_node(v2.value)

                for v in cum_stack:
                    added_edges.append(Edge(v, interm_node, op_to_edge_label_map[op]))
                    #  Add the primitive version as well as a fallback for the user.
                    added_edges.append(Edge(v, interm_node, op_to_edge_label_map[op[3:]]))

                added_edges.append(Edge(interm_node, v2, ELabel.EQUAL))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #
    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_by_cols:
        if c_node.value in by_cols:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@groupby_transform_by_cols"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@groupby_transform_by_cols"))

    for c_node in cands_op_col:
        if c_node.value == op_col:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@groupby_transform_op_col"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@groupby_transform_op_col"))

    graph.add_tags([f"SELECTED@{cand}@groupby_transform_op"
                    if cand == op else
                    f"NOT_SELECTED@{cand}@groupby_transform_op"
                    for cand in candidates_groupby_transform_op])

    graph.add_tagged_edges(tagged_edges)

    return result, call_str, graph, g_res


@generator(name='pd.pivot_table', group='pandas', strategy=DfsGraphStrategy())
def gen_pivot_table(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    cands_columns = g_df.columns
    columns = SelectNode(cands_columns, uid="pivot_columns")

    cands_values = [c for c in g_df.columns if c.value != columns]
    values = SelectNode(cands_values, uid="pivot_values")

    cands_index = [c for c in g_df.columns if c.value != columns and c.value != values]
    index = list(SubsetNode(cands_index, uid="pivot_index", allow_empty=False))

    call_str = f"{{inp1}}.pivot_table(columns={columns!r}, \nvalues={values!r}, \nindex={index!r}," \
               f"\naggfunc='first').reset_index()"
    result = df.pivot_table(columns=columns, values=values, index=index, aggfunc='first').reset_index()

    result.columns.names = [None] * result.columns.nlevels

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edge between the nodes in the `columns` column and the column node of the result.
    for cell in g_df.loc[:, columns]:
        added_edges.append(Edge(cell, col_map_res[cell.value], ELabel.EQUAL))

    #  - EQUAL edge between the `index` column and the corresponding column node of the result.
    #  - EQUAL edge between the cells of `index` column and the corresponding cells in the result.
    for c in index:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        value_map = {n.value: n for n in g_res.loc[:, c]}
        for cell in g_df.loc[:, c]:
            added_edges.append(Edge(cell, value_map[cell.value], ELabel.EQUAL))

    #  - EQUAL edge between the cells of `values` column and the corresponding cells in the result.
    for idx_vals, col_val, df_val_node in zip(df[index].values if len(index) > 0 is not None else df.index,
                                              df.loc[:, columns], g_df.loc[:, values]):
        if len(index) == 0:
            filtered = [g_res.loc[idx_vals, col_val]]
        else:
            idx_mask = True
            for idx_val, idx in zip(idx_vals, index):
                idx_mask = idx_mask & (result[idx] == idx_val)

            filtered = list(g_res.loc[idx_mask][col_val])

        assert len(filtered) == 1
        res_val_node = filtered[0]
        added_edges.append(Edge(df_val_node, res_val_node, ELabel.EQUAL))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #
    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_columns:
        if c_node.value == columns:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@pivot_columns"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@pivot_columns"))

    for c_node in cands_values:
        if c_node.value == values:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@pivot_values"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@pivot_values"))

    for c_node in cands_index:
        if c_node.value in index:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@pivot_index"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@pivot_index"))

    graph.add_tagged_edges(tagged_edges)

    return result, call_str, graph, g_res


@generator(name='pd.filtering_expr', group='pandas', strategy=DfsGraphStrategy())
def gen_filtering_expr(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    #  We will use expressions like `a` > 100 to filter rows.
    #  For now we don't care about synthesizing these and rely on the user to provide them.
    #  TODO : May want to use something more robust than strings so users don't have to worry about formatting
    #  TODO : validity constraints of Pandas.
    available_constants = [c for c in constants if isinstance(c, str)]
    expression = SelectConst(available_constants, uid="filtering_expr_expression")

    result = df.query(expression)
    filtered_indices = list(result.index)
    removed_indices = list(set(df.index) - set(filtered_indices))
    call_str = f"{{inp1}}.query({expression!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between the corresponding columns as all columns are preserved.
    #  - EQUAL edges between the cells that are preserved.
    for c in col_map_df:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        #  Only consider the filtered rows
        for index, v1, v2 in zip(filtered_indices, g_df.loc[filtered_indices, c], g_res.loc[:, c]):
            interm_node = graph.create_intermediate_node(value=v2.value)
            added_edges.append(Edge(v1, interm_node, ELabel.FILTER_EXPR))
            added_edges.append(Edge(interm_node, v2, ELabel.EQUAL))

        #  Mark the rest as deleted
        for v in g_df.loc[removed_indices, c]:
            added_edges.append(Edge(v, g_res.deletion_node, ELabel.DELETE))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    #  No arguments as such
    return result, call_str, graph, g_res


@generator(name='pd.filtering_contains', group='pandas', strategy=DfsGraphStrategy())
def gen_filtering_contains(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    available_constants = [c for c in constants if isinstance(c, Collection)]
    collection = SelectConst(available_constants, uid="filtering_contains_collection")

    cands_filter_col = g_df.columns
    filter_col = SelectNode(cands_filter_col, uid="filtering_contains_filter_col")

    result = df[df[filter_col].isin(collection)]
    filtered_indices = list(result.index)
    removed_indices = list(set(df.index) - set(filtered_indices))
    call_str = f"{{inp1}}[{{inp1}}[{filter_col!r}] \\\n.isin({list(collection)!r})]"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between the corresponding columns as all columns are preserved.
    #  - EQUAL edges between the cells that are preserved.
    for c in col_map_df:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        #  Only consider the filtered rows
        for index, v1, v2 in zip(filtered_indices, g_df.loc[filtered_indices, c], g_res.loc[:, c]):
            interm_node = graph.create_intermediate_node(value=v2.value)
            added_edges.append(Edge(g_df.loc[index, filter_col], interm_node, ELabel.CONTAINED_IN))
            added_edges.append(Edge(v1, interm_node, ELabel.EQUAL))
            added_edges.append(Edge(interm_node, v2, ELabel.EQUAL))

        #  Mark the rest as deleted
        for v in g_df.loc[removed_indices, c]:
            added_edges.append(Edge(v, g_res.deletion_node, ELabel.DELETE))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #
    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_filter_col:
        if c_node.value == filter_col:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@filtering_contains_filter_col"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@filtering_contains_filter_col"))

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name='pd.melt', group='pandas', strategy=DfsGraphStrategy())
def gen_melt(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    cands_id_vars = g_df.columns
    id_vars = list(SubsetNode(cands_id_vars, uid="melt_id_vars", allow_empty=True))

    cands_value_vars = [c for c in g_df.columns if c.value not in id_vars]
    value_vars = list(SubsetNode(cands_value_vars, uid="melt_value_vars"))

    var_name = FreshColumn(prefix='Var', uid="melt_var_name")
    value_name = FreshColumn(prefix='Value', uid="melt_value_name")

    result = df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    call_str = f"{{inp1}}.melt(id_vars={id_vars!r}, \nvalue_vars={value_vars!r}, " \
               f"\nvar_name={var_name!r}, \nvalue_name={value_name!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between id_var columns and corresponding columns in output.
    for c1, c2 in ((col_map_df[c], col_map_res[c]) for c in id_vars or []):
        added_edges.append(Edge(c1, c2, ELabel.EQUAL))

    #  - EQUAL edges between value_var columns and corresponding cells in output.
    for var_node in g_res.loc[:, var_name]:
        added_edges.append(Edge(col_map_df[var_node.value], var_node, ELabel.EQUAL))

    #  - EQUAL edges between cells of id_var and value_var columns to the corresponding cells in output.
    for col in id_vars or []:
        df_nodes_concat = list(g_df.loc[:, col]) * len(value_vars)
        for n1, n2 in zip(df_nodes_concat, g_res.loc[:, col]):
            added_edges.append(Edge(n1, n2, ELabel.EQUAL))

    value_nodes_concat = sum((list(g_df.loc[:, c]) for c in value_vars), [])
    for n1, n2 in zip(value_nodes_concat, g_res.loc[:, value_name]):
        added_edges.append(Edge(n1, n2, ELabel.EQUAL))

    #  - DELETE edge between the columns not in id_vars and value_vars to the output deletion node.
    #  - DELETE edge between the cells of columns not in id_vars and value_vars to the output deletion node.
    unused_cols = [c for c in df.columns if c not in id_vars and c not in value_vars]
    for col in unused_cols:
        added_edges.append(Edge(col_map_df[col], g_res.deletion_node, ELabel.DELETE))
        for val_node in g_df.loc[:, col]:
            added_edges.append(Edge(val_node, g_res.deletion_node, ELabel.DELETE))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #
    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_id_vars:
        if id_vars is not None and c_node.value in id_vars:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@melt_id_vars"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@melt_id_vars"))

    for c_node in cands_value_vars:
        if c_node.value in value_vars:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@melt_value_vars"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@melt_value_vars"))

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name="pd.drop_columns", group="pandas", strategy=DfsGraphStrategy())
def gen_drop_columns(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    cands_cols = list(g_df.columns)
    drop_cols = list(SubsetNode(cands_cols, uid="drop_columns_cols"))

    result = df.drop(columns=drop_cols)
    call_str = f"{{inp1}}.drop(columns={drop_cols!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between the corresponding preserved columns.
    #  - EQUAL edges between the cells that are preserved.
    for c in col_map_res:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        for v1, v2 in zip(g_df.loc[:, c], g_res.loc[:, c]):
            added_edges.append(Edge(v1, v2, ELabel.EQUAL))

    #  - DELETE edges for the deleted columns and their cells.
    for c in drop_cols:
        added_edges.append(Edge(col_map_df[c], g_res.deletion_node, ELabel.DELETE))
        for v in g_df.loc[:, c]:
            added_edges.append(Edge(v, g_res.deletion_node, ELabel.DELETE))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #
    tagged_edges: List[TaggedEdge] = []

    for c_node in cands_cols:
        if c_node.value in drop_cols:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@drop_columns_cols"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@drop_columns_cols"))

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name="pd.fillna", group="pandas", strategy=DfsGraphStrategy())
def gen_fillna(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    mode = SelectConst(["value", "method"], uid="fillna_mode")
    if mode == 'value':
        cands_value = [c for c in constants if np.isscalar(c)]
        value = SelectConst(cands_value, uid="fillna_value")
        axis = None
        method = None

    else:
        value = None
        method = SelectConst(["backfill", "pad"], uid="fillna_method")
        axis = SelectConst(["index", "columns"], uid="fillna_axis")

    result = df.fillna(value=value, axis=axis, method=method)
    call_str = f"{{inp1}}.fillna(value={value!r}, \naxis={axis!r}, \nmethod={method!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between all columns as they are all preserved.
    for c in col_map_df:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))

    #  - EQUAL edges between non-null values.
    #  - REPLACE/REPLACE_CONST edges for null values.
    for row in range(0, df.shape[0]):
        for col in range(0, df.shape[1]):
            inp_node = g_df.iloc[row, col]
            out_node = g_res.iloc[row, col]
            if not pd.isnull(df.iloc[row, col]):
                added_edges.append(Edge(inp_node, out_node, ELabel.EQUAL))
            else:
                if mode == "value":
                    #  Simple replacement by a constant.
                    interm_node = graph.create_intermediate_node(value=out_node.value)
                    added_edges.append(Edge(inp_node, interm_node, ELabel.REPLACE_CONST))
                    added_edges.append(Edge(interm_node, out_node, ELabel.EQUAL))

                else:
                    #  Need to locate the node used to fill this null value.
                    r_node = _find_fillna_replacement_node(row, col, g_df, g_res, method, axis)
                    if r_node is None:
                        assert out_node.value is MY_NAN
                        added_edges.append(Edge(inp_node, out_node, ELabel.EQUAL))
                    else:
                        interm_node = graph.create_intermediate_node(value=out_node.value)
                        added_edges.append(Edge(inp_node, interm_node, ELabel.REPLACED))
                        added_edges.append(Edge(r_node, interm_node, ELabel.REPLACEMENT))
                        added_edges.append(Edge(interm_node, out_node, ELabel.EQUAL))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    if mode == "method":
        if axis == "index":
            graph.add_tags(["SELECTED@index@fillna_axis", "NOT_SELECTED@columns@fillna_axis"])
        else:
            graph.add_tags(["NOT_SELECTED@index@fillna_axis", "SELECTED@columns@fillna_axis"])

        if method == "backfill":
            graph.add_tags(["SELECTED@backfill@fillna_method", "NOT_SELECTED@pad@fillna_method"])
        else:
            graph.add_tags(["NOT_SELECTED@backfill@fillna_method", "SELECTED@pad@fillna_method"])

    return result, call_str, graph, g_res


@generator(name="pd.dropna", group="pandas", strategy=DfsGraphStrategy())
def gen_dropna(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    #  For now, we only have the ability to drop rows
    cands_inspect_cols = g_df.columns
    inspect_cols = list(SubsetNode(cands_inspect_cols, uid="dropna_inspect_cols"))

    how = SelectConst(["any", "all"], uid="dropna_how")

    result = df.dropna(subset=inspect_cols, how=how, axis="index")
    filtered_indices = list(result.index)
    removed_indices = list(set(df.index) - set(filtered_indices))
    call_str = f"{{inp1}}.dropna(axis='index', \nhow={how!r}, \nsubset={inspect_cols!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between the corresponding columns as all columns are preserved.
    #  - EQUAL edges between the cells that are preserved.
    for c in col_map_df:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        #  Only consider the filtered rows
        for index, v1, v2 in zip(filtered_indices, g_df.loc[filtered_indices, c], g_res.loc[:, c]):
            added_edges.append(Edge(v1, v2, ELabel.EQUAL))

        #  Mark the rest as deleted
        for v in g_df.loc[removed_indices, c]:
            added_edges.append(Edge(v, g_res.deletion_node, ELabel.DELETE))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_inspect_cols:
        if c_node.value in inspect_cols:
            tagged_edges.append(TaggedEdge(c_node, c_node, "SELECTED@dropna_inspect_cols"))
        else:
            tagged_edges.append(TaggedEdge(c_node, c_node, "NOT_SELECTED@dropna_inspect_cols"))

    if how == "any":
        graph.add_tags(["SELECTED@any@dropna_how", "NOT_SELECTED@all@dropna_how"])
    else:
        graph.add_tags(["NOT_SELECTED@any@dropna_how", "SELECTED@all@dropna_how"])

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name="pd.merge", group="pandas", strategy=DfsGraphStrategy())
def gen_merge(df1: pd.DataFrame, df2: pd.DataFrame, g_df1: DataFrameGraph, g_df2: DataFrameGraph,
              constants: List[Any], datagen: bool = False):
    #  For now, simply an inner_join.
    result = df1.merge(df2)
    call_str = f"{{inp1}}.merge({{inp2}}, how='inner')"

    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ExceptionAsContinue

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df1, g_df2, g_res])
    added_edges: List[Edge] = []

    col_map_df1 = {c.value: c for c in g_df1.columns}  # Map from df1's columns to their column nodes
    col_map_df2 = {c.value: c for c in g_df2.columns}  # Map from df2's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  All the columns will be preserved
    for c in col_map_df1:
        added_edges.append(Edge(col_map_df1[c], col_map_res[c], ELabel.EQUAL))
    for c in col_map_df2:
        added_edges.append(Edge(col_map_df2[c], col_map_res[c], ELabel.EQUAL))

    #  Get the merge cols
    merge_cols = list(set(col_map_df1.keys()) & set(col_map_df2.keys()))
    for c in merge_cols:
        added_edges.append(Edge(col_map_df1[c], col_map_df2[c], ELabel.EQUAL))
        added_edges.append(Edge(col_map_df2[c], col_map_df1[c], ELabel.EQUAL))

    #  Get the indices for each value tuple for df1 and df2 and result
    value_dict_df1 = collections.defaultdict(list)
    value_dict_df2 = collections.defaultdict(list)
    value_dict_res = collections.defaultdict(list)

    for idx, values in zip(df1.index, df1.loc[:, merge_cols].values):
        values = tuple(values)
        value_dict_df1[values].append(idx)

    for idx, values in zip(df2.index, df2.loc[:, merge_cols].values):
        values = tuple(values)
        value_dict_df2[values].append(idx)

    for idx, values in zip(result.index, result.loc[:, merge_cols].values):
        values = tuple(values)
        value_dict_res[values].append(idx)

    #  Start adding the equality edges
    deleted_df1 = set(df1.index)
    deleted_df2 = set(df2.index)

    for value, res_indices in value_dict_res.items():
        df1_indices = value_dict_df1[value]
        df2_indices = value_dict_df2[value]
        deleted_df1.difference_update(df1_indices)
        deleted_df2.difference_update(df2_indices)
        for idx_res, (idx1, idx2) in zip(res_indices, itertools.product(df1_indices, df2_indices)):
            for c in df1.columns:
                added_edges.append(Edge(g_df1.loc[idx1, c], g_res.loc[idx_res, c], ELabel.EQUAL))
            for c in df2.columns:
                added_edges.append(Edge(g_df2.loc[idx2, c], g_res.loc[idx_res, c], ELabel.EQUAL))
            for c in merge_cols:
                added_edges.append(Edge(g_df1.loc[idx1, c], g_df2.loc[idx2, c], ELabel.EQUAL))
                added_edges.append(Edge(g_df2.loc[idx2, c], g_df1.loc[idx1, c], ELabel.EQUAL))

    for idx1 in deleted_df1:
        for c in df1.columns:
            added_edges.append(Edge(g_df1.loc[idx1, c], g_res.deletion_node, ELabel.DELETE))

    for idx2 in deleted_df2:
        for c in df2.columns:
            added_edges.append(Edge(g_df2.loc[idx2, c], g_res.deletion_node, ELabel.DELETE))

    #  - EQUAL edge between the input deletion nodes and the output deletion node.
    added_edges.append(Edge(g_df1.deletion_node, g_res.deletion_node, ELabel.EQUAL))
    added_edges.append(Edge(g_df2.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    return result, call_str, graph, g_res


@generator(name="pd.combine_first", group="pandas", strategy=DfsGraphStrategy())
def gen_combine_first(df1: pd.DataFrame, df2: pd.DataFrame, g_df1: DataFrameGraph, g_df2: DataFrameGraph,
                      constants: List[Any], datagen: bool = False):
    order = SelectConst(["1", "2"], uid="combine_first_order")
    if order == "1":
        result = df1.combine_first(df2)
        call_str = f"{{inp1}}.combine_first({{inp2}})"
    else:
        df1, df2, g_df1, g_df2 = df2, df1, g_df2, g_df1
        result = df1.combine_first(df2)
        call_str = f"{{inp2}}.combine_first({{inp1}})"

    df1_indices = set(df1.index)
    df2_indices = set(df2.index)

    df1_columns = set(df1.columns)
    df2_columns = set(df2.columns)

    common_indices = df1_indices & df2_indices
    common_columns = df1_columns & df2_columns

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df1, g_df2, g_res])
    added_edges: List[Edge] = []

    col_map_df1 = {c.value: c for c in g_df1.columns}  # Map from df1's columns to their column nodes
    col_map_df2 = {c.value: c for c in g_df2.columns}  # Map from df2's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    for c in col_map_res:
        if c in df1_columns:
            added_edges.append(Edge(col_map_df1[c], col_map_res[c], ELabel.EQUAL))
        if c in df2_columns:
            added_edges.append(Edge(col_map_df2[c], col_map_res[c], ELabel.EQUAL))

    for index in result.index:
        for column in result.columns:
            out_node = g_res.loc[index, column]
            if index in common_indices and column in common_columns:
                df1_node = g_df1.loc[index, column]
                df2_node = g_df2.loc[index, column]
                if (df1_node.value is not MY_NAN) or (df2_node.value is MY_NAN):
                    added_edges.append(Edge(df1_node, out_node, ELabel.EQUAL))
                else:
                    added_edges.append(Edge(df2_node, out_node, ELabel.EQUAL))

            elif index in df1_indices and column in df1_columns:
                df1_node = g_df1.loc[index, column]
                added_edges.append(Edge(df1_node, out_node, ELabel.EQUAL))

            elif index in df2_indices and column in df2_columns:
                df2_node = g_df2.loc[index, column]
                added_edges.append(Edge(df2_node, out_node, ELabel.EQUAL))

    #  - EQUAL edge between the input deletion nodes and the output deletion node.
    added_edges.append(Edge(g_df1.deletion_node, g_res.deletion_node, ELabel.EQUAL))
    added_edges.append(Edge(g_df2.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    return result, call_str, graph, g_res


@generator(name='pd.unite', group='pandas', strategy=DfsGraphStrategy())
def gen_unite(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    cands_cols = g_df.columns
    cols = list(OrderedSubsetNode(cands_cols, min_len=2, uid="unite_cols"))

    new_col = FreshColumn(prefix="strjoin_col", uid="unite_new_col")

    result = df.drop(columns=cols).assign(**{new_col: df[cols[0]].astype(str).str.cat(df[cols[1:]].astype(str),
                                                                                      sep='_')})
    call_str = f"{{inp1}}.drop(columns={cols!r}) \\\n.assign(**{{{{{new_col!r}: " \
               f"{{inp1}}[{cols[0]!r}].astype(str).str.cat(\n{{inp1}}[{cols[1:]!r}].astype(str), \nsep='_')\n}}}})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between columns that are unused and their cells to their counterparts.
    unused_cols = set(col_map_df)
    unused_cols.difference_update(cols)

    for c in unused_cols:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        for v1, v2 in zip(g_df.loc[:, c], g_res.loc[:, c]):
            added_edges.append(Edge(v1, v2, ELabel.EQUAL))

    for df_nodes, res_node in zip(g_df.loc[:, cols].values, g_res.loc[:, new_col]):
        interm_node = graph.create_intermediate_node(res_node.value)
        for n in df_nodes:
            added_edges.append(Edge(n, interm_node, ELabel.STR_JOIN))

        added_edges.append(Edge(interm_node, res_node, ELabel.EQUAL))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_cols:
        if c_node.value in cols:
            tagged_edges.append(TaggedEdge(c_node, c_node, "SELECTED@unite_cols"))
        else:
            tagged_edges.append(TaggedEdge(c_node, c_node, "NOT_SELECTED@unite_cols"))

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name='pd.separate', group='pandas', strategy=DfsGraphStrategy())
def gen_separate(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    cands_cols = g_df.columns
    col = SelectNode(cands_cols, min_len=2, uid="separate_col")

    new_columns_df = df[col].str.split(r'[^A-Za-z0-9]+', expand=True)

    new_cols = [FreshColumn(prefix=f"strsplit_col{i}", uid="separate_new_col")
                for i in range(new_columns_df.shape[1])]
    rename_dict = {idx: col for idx, col in enumerate(new_cols)}

    result = pd.concat([df.drop(columns=[col]), new_columns_df.rename(columns=rename_dict)], axis=1)
    call_str = f"pd.concat([\n{{inp1}}.drop(columns=[{col!r}]), \n" \
               f"{{inp1}}[{col!r}].str.split(r'[^A-Za-z0-9]+', expand=True) \\\n" \
               f".rename(columns={{{rename_dict!r}}})\n], axis=1)"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between columns that are unused and their cells to their counterparts.
    unused_cols = set(col_map_df)
    unused_cols.difference_update(col)

    deletion_node = g_res.deletion_node

    for c in col_map_df:
        if c != col:
            added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
            for v1, v2 in zip(g_df.loc[:, c], g_res.loc[:, c]):
                added_edges.append(Edge(v1, v2, ELabel.EQUAL))

    for df_node, res_nodes in zip(g_df.loc[:, col], g_res.loc[:, new_cols].values):
        for n in res_nodes:
            interm_node = graph.create_intermediate_node(n.value)
            added_edges.append(Edge(interm_node, n, ELabel.EQUAL))
            added_edges.append(Edge(df_node, interm_node, ELabel.STR_SPLIT))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_cols:
        if c_node.value == col:
            tagged_edges.append(TaggedEdge(c_node, c_node, "SELECTED@separate_col"))
        else:
            tagged_edges.append(TaggedEdge(c_node, c_node, "NOT_SELECTED@separate_col"))

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name='pd.mutate', group='pandas', strategy=DfsGraphStrategy())
def gen_mutate(df: pd.DataFrame, g_df: DataFrameGraph, constants: List[Any], datagen: bool = False):
    cands_cols = g_df.columns
    cols = list(OrderedSubsetNode(cands_cols, min_len=2, max_len=2, uid="mutate_cols"))
    op = SelectConst(["add", "sub", "mul", "div"], uid="mutate_op")

    new_col = FreshColumn(prefix=f"{op.lower()}_{cols[0].lower()}_{cols[1].lower()}", uid="mutate_new_col")

    result = df.assign(**{new_col: getattr(df[cols[0]], op)(df[cols[1]])})
    call_str = f"{{inp1}}.assign(**{{{{\n{new_col!r}: " \
               f"{{inp1}}[{cols[0]!r}].{op}({{inp1}}[{cols[1]!r}])\n}}}})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphPandas.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between the corresponding columns as all columns are preserved.
    #  - EQUAL edges between the cells that are preserved.
    for c in col_map_df:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        for v1, v2 in zip(g_df.loc[:, c], g_res.loc[:, c]):
            added_edges.append(Edge(v1, v2, ELabel.EQUAL))

    #  - Respective OP edges
    if op == "add":
        label = ELabel.SUM
    elif op == "mul":
        label = ELabel.PROD
    else:
        label = getattr(ELabel, op.upper())

    for cell_nodes, res_node in zip(g_df.loc[:, cols].values, g_res.loc[:, new_col]):
        interm_node = graph.create_intermediate_node(res_node.value)
        added_edges.append(Edge(interm_node, res_node, ELabel.EQUAL))
        for n in cell_nodes:
            added_edges.append(Edge(n, interm_node, label))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_cols:
        if c_node.value in cols:
            tagged_edges.append(TaggedEdge(c_node, c_node, "SELECTED@mutate_cols"))
        else:
            tagged_edges.append(TaggedEdge(c_node, c_node, "NOT_SELECTED@mutate_cols"))

    graph.add_tags([f"SELECTED@{cand}@mutate_op"
                    if cand == op else
                    f"NOT_SELECTED@{cand}@mutate_op"
                    for cand in ["add", "sub", "mul", "div"]])

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


generator_dict: Dict[str, Generator] = {
    'pd.groupby_agg': gen_groupby_agg,
    'pd.groupby_transform': gen_groupby_transform,
    'pd.pivot_table': gen_pivot_table,
    'pd.filtering_expr': gen_filtering_expr,
    'pd.filtering_contains': gen_filtering_contains,
    'pd.melt': gen_melt,
    'pd.drop_columns': gen_drop_columns,
    'pd.fillna': gen_fillna,
    'pd.dropna': gen_dropna,
    'pd.merge': gen_merge,
    'pd.combine_first': gen_combine_first,
    'pd.unite': gen_unite,
    'pd.separate': gen_separate,
    'pd.mutate': gen_mutate,
}
