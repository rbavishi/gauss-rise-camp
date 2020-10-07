import collections
import itertools
import re

import pandas as pd
from atlas import generator, Generator
from typing import Dict, List

from atlas.exceptions import ExceptionAsContinue

from gauss.domains.rlang.graphs import DataFrameGraph, GraphRLang, ELabel
from gauss.domains.rlang.interpreter import RInterpreter
from gauss.domains.rlang.strategies import DfsGraphStrategy, SubsetNode, FreshColumn, SelectConst, SelectNode, \
    OrderedSubsetNode
from gauss.graphs import Edge, TaggedEdge


@generator(name='rf.gather', group='rlang', strategy=DfsGraphStrategy())
def gen_gather(df: pd.DataFrame, g_df: DataFrameGraph, datagen: bool = False):
    """
    GATHER
    ------
    Example:
     gather(id_vars=['C1'], value_vars=['C2', 'C3'], var_name='var', value_name='value')

             df                            result
       C1 C2 C3                    C1  var  value
     0  a  b  e       -->       0   a   C2      b
     1  c  d  f                 1   c   C2      d
                                2   a   C3      e
                                3   c   C3      f

    ---------------
    Graph Abstraction:
    - EQUAL edges between id_var columns and corresponding columns in output.
    - EQUAL edges between value_var columns and corresponding cells in output.
    - EQUAL edges between cells of id_var and value_var columns to the corresponding cells in output.
    - EQUAL edge between the input deletion node and the output deletion node.
    - DELETE edge between the columns not in id_vars and value_vars to the output deletion node.
    - DELETE edge between the cells of columns not in id_vars and value_vars to the output deletion node.
    """

    cands_id_vars = g_df.columns
    id_vars = list(SubsetNode(cands_id_vars, uid="gather_id_vars", allow_empty=True))

    cands_value_vars = [c for c in g_df.columns if c.value not in id_vars]
    value_vars = list(SubsetNode(cands_value_vars, uid="gather_value_vars"))

    var_name = FreshColumn(uid="gather_var_name")
    value_name = FreshColumn(uid="gather_value_name")

    result = RInterpreter.gather(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    call_str = f"gather({{inp1}}, id_vars={id_vars!r}, value_vars={value_vars!r}, " \
               f"var_name={var_name!r}, value_name={value_name!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphRLang.assemble([g_df, g_res])
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

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  - DELETE edge between the columns not in id_vars and value_vars to the output deletion node.
    #  - DELETE edge between the cells of columns not in id_vars and value_vars to the output deletion node.
    unused_cols = [c for c in df.columns if c not in id_vars and c not in value_vars]
    for col in unused_cols:
        added_edges.append(Edge(col_map_df[col], g_res.deletion_node, ELabel.DELETE))
        for val_node in g_df.loc[:, col]:
            added_edges.append(Edge(val_node, g_res.deletion_node, ELabel.DELETE))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #
    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_id_vars:
        if id_vars is not None and c_node.value in id_vars:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@gather_id_vars"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@gather_id_vars"))

    for c_node in cands_value_vars:
        if c_node.value in value_vars:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@gather_value_vars"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@gather_value_vars"))

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name='rf.group_by', group='rlang', strategy=DfsGraphStrategy())
def gen_group_by_summarise(df: pd.DataFrame, g_df: DataFrameGraph, datagen: bool = False):
    """
    GROUP_BY (+ SUMMARISE)
    ------
    Example:
     group_by(group_cols=['C1']).summarise(summaries={"C3": ("C2", "mean")})
              df              result
        C1    C2            C1    C3
     0   A   100         0   A   150
     1   A   200   -->   1   B   300
     2   B   300

    ---------------
    Graph Abstraction:
    - EQUAL edges between group columns and corresponding columns in output.
    - EQUAL edges between the cells of the group columns and the corresponding cells in the output.
    - SUM/MEAN/COUNT edges between the cells of aggregated columns (or group columns in case of 'count') and the
      resulting cells in the output.
    - EQUAL edge between the input deletion node and the output deletion node.
    - DELETE edges between the non-group cols and non-aggregated columns and their cells
      to the deletion node of the output.
    - DELETE edge between the column of the aggregated column (in case of sum/mean) and the deletion node of the output.
    """

    cands_group_cols = g_df.columns
    group_cols = list(SubsetNode(cands_group_cols, uid="group_by_group_cols"))
    new_col = FreshColumn(uid="summarise_new_col")
    agg_cands = ["count", "mean", "sum"]
    agg = SelectConst(agg_cands, uid="summarise_agg")

    if agg != "count":
        cands_agg_col = [c for c in g_df.columns if c.value not in group_cols]
        agg_col = SelectNode(cands_agg_col, uid="summarise_col")
    else:
        cands_agg_col = None
        agg_col = None

    summaries = {new_col: (agg_col, agg)}
    result_groupby = RInterpreter.group_by(df, group_cols)
    result = RInterpreter.summarise(result_groupby, summaries)

    call_str = f"summarise(group_by({{inp1}}, group_cols={group_cols!r}), summaries={{{summaries!r}}})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphRLang.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between group columns and corresponding columns in output.
    for c in group_cols:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))

    #  - EQUAL edges between the cells of the group columns and the corresponding cells in the output.
    #  - SUM/MEAN/COUNT edges between the cells of aggregated columns (or group columns in case of 'count') and the
    #    resulting cells in the output.
    agg_cols = [agg_col] if agg != "count" else group_cols
    for idx, group in enumerate(result.loc[:, group_cols].values):
        group = group[0] if len(group) == 1 else tuple(group)
        df_indices = result_groupby.groups[group]  # Get the indices in df that correspond to this group
        out_node = g_res.loc[result.index[idx], new_col]
        interm_node = graph.create_intermediate_node(out_node.value)
        added_edges.append(Edge(interm_node, out_node, ELabel.EQUAL))

        #  Aggregation edges
        for col in agg_cols:
            for val_node in g_df.loc[df_indices, col]:
                added_edges.append(Edge(val_node, interm_node, getattr(ELabel, agg.upper())))

        #  Equality edges for group_col nodes
        for col in group_cols:
            group_node = g_res.loc[result.index[idx], col]
            for df_group_node in g_df.loc[df_indices, col]:
                added_edges.append(Edge(df_group_node, group_node, ELabel.EQUAL))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  - DELETE edges between the non-group cols and non-aggregated columns and their cells
    #    to the deletion node of the output.
    #  - DELETE edge between the column of the aggregated column (in case of sum/mean) and the deletion node
    #    of the output.
    non_group_cols = [c for c in df.columns if c not in group_cols]
    for col in non_group_cols:
        added_edges.append(Edge(col_map_df[col], g_res.deletion_node, ELabel.DELETE))
        if col != agg_col:
            for cell in g_df.loc[:, col]:
                added_edges.append(Edge(cell, g_res.deletion_node, ELabel.DELETE))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #
    tagged_edges: List[TaggedEdge] = []
    for c_node in cands_group_cols:
        if c_node.value in group_cols:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@group_by_group_cols"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@group_by_group_cols"))

    if cands_agg_col is not None:
        for c_node in cands_agg_col:
            if c_node.value == agg_col:
                tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@summarise_col"))
            else:
                tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@summarise_col"))

    graph.add_tags([f"SELECTED@{cand}@summarise_agg" if cand == agg else f"NOT_SELECTED@{cand}@summarise_agg"
                    for cand in agg_cands])
    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name='rf.unite', group='rlang', strategy=DfsGraphStrategy())
def gen_unite(df: pd.DataFrame, g_df: DataFrameGraph, datagen: bool = False):
    """
    UNITE
    ------
    Example:
     unite(cols=['C2', 'C3'], new_col_name='C4')
      ---------------
                df             result
        C1  C2  C3           C1    C4
     0   3   a   d  -->   0   3   a_d
     1   4   b   e        1   4   b_e
     2   5   c   f        2   5   c_f

    ---------------
    Graph Abstraction:
    - EQUAL edges between the columns, cells of the preserved columns and the corresponding cells in the output.
    - STR_JOIN edges between concerned cells.
    """

    cands_cols = g_df.columns
    cols = list(OrderedSubsetNode(cands_cols, min_len=2, uid="unite_cols"))

    new_col_name = FreshColumn(uid="unite_new_col_name")

    result = RInterpreter.unite(df, cols=cols, new_col_name=new_col_name)
    call_str = f"unite({{inp1}}, cols={cols!r}), new_col_name={new_col_name!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphRLang.assemble([g_df, g_res])
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

    for df_nodes, res_node in zip(g_df.loc[:, cols].values, g_res.loc[:, new_col_name]):
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


@generator(name='rf.separate', group='rlang', strategy=DfsGraphStrategy())
def gen_separate(df: pd.DataFrame, g_df: DataFrameGraph, datagen: bool = False):
    """
    SEPARATE
    ------
    Example:
     separate(split_col='C2', into=["C3", "C4"])
      ---------------
              df           result
        C1    C2       C1  C3  C4
     0   3   a_d    0   3   a   d
     1   4   b_e    1   4   b   e
     2   5   c_f    2   5   c   f

    ---------------
    Graph Abstraction:
    - EQUAL edges between the columns, cells of the preserved columns and the corresponding cells in the output.
    - STR_SPLIT edges between concerned cells.
    """
    cands_cols = g_df.columns
    split_col = SelectNode(cands_cols, min_len=2, uid="separate_split_col")
    max_into_len = max([len(re.compile("[^a-zA-Z0-9]+").split(str(x))) for x in df[split_col]])
    if max_into_len <= 1:
        raise ExceptionAsContinue

    into = [FreshColumn(uid="separate_into") for _ in range(max_into_len)]
    result = RInterpreter.separate(df, split_col=split_col, into=into)
    call_str = f"separate({{inp1}}, split_col={split_col!r}, into={into!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphRLang.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between columns that are unused and their cells to their counterparts.
    unused_cols = set(col_map_df)
    unused_cols.difference_update(split_col)

    for c in col_map_df:
        if c != split_col:
            added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
            for v1, v2 in zip(g_df.loc[:, c], g_res.loc[:, c]):
                added_edges.append(Edge(v1, v2, ELabel.EQUAL))

    for df_node, res_nodes in zip(g_df.loc[:, split_col], g_res.loc[:, into].values):
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
        if c_node.value == split_col:
            tagged_edges.append(TaggedEdge(c_node, c_node, "SELECTED@separate_split_col"))
        else:
            tagged_edges.append(TaggedEdge(c_node, c_node, "NOT_SELECTED@separate_split_col"))

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name='rf.spread', group='rlang', strategy=DfsGraphStrategy())
def gen_spread(df: pd.DataFrame, g_df: DataFrameGraph, datagen: bool = False):
    """
    SPREAD
    ------
    Example:
     spread(columns='var', values='value')
      ---------------
                    df                      result
        C1  var  value                C1   C2   C3
     0   c   C2      b            0    a    d    e
     1   a   C2      d     -->    1    c    b  NaN
     2   a   C3      e

    ---------------
    Graph Abstraction:
     - EQUAL edge between the nodes in the `columns` column and the column node of the result.
     - EQUAL edge between the `index` column and the corresponding column node of the result.
     - EQUAL edge between the cells of `values` column and the corresponding cells in the result.
     - EQUAL edge between the input deletion node and the output deletion node.
    """

    cands_columns = g_df.columns
    columns = SelectNode(cands_columns, uid="spread_columns")

    cands_values = [c for c in g_df.columns if c.value != columns]
    values = SelectNode(cands_values, uid="spread_values")

    index = [c for c in df.columns if c != columns and c != values]  # The columns that will remain as is.

    result = RInterpreter.spread(df, columns=columns, values=values)
    call_str = f"spread({{inp1}}, columns={columns!r}, values={values!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphRLang.assemble([g_df, g_res])
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
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@spread_columns"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@spread_columns"))

    for c_node in cands_values:
        if c_node.value == values:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@spread_values"))
        else:
            tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@spread_values"))

    graph.add_tagged_edges(tagged_edges)

    return result, call_str, graph, g_res


@generator(name="rf.select", group="rlang", strategy=DfsGraphStrategy())
def gen_select(df: pd.DataFrame, g_df: DataFrameGraph, datagen: bool = False):
    """
    SELECT
    ------
    Example:
     select(df, columns_keep=None, columns_remove=['C3'])
      ---------------
                df        result
        C1  C2  C3           C1  C2
     0   3   a   d  -->   0   3   a
     1   4   b   e        1   4   b
     2   5   c   f        2   5   c

    ---------------
    Graph Abstraction:
    - EQUAL edges between the columns, cells of the preserved columns and the corresponding cells in the output.
    - DELETE edges for the removed columns and their cells.
    """
    cands_cols = list(g_df.columns)
    choice = SelectConst([True, False], uid="select_keep_or_remove")
    if choice:
        columns_keep = list(SubsetNode(cands_cols, uid="select_columns_keep", allow_empty=False))
        columns_remove = None
    else:
        columns_keep = None
        columns_remove = list(SubsetNode(cands_cols, uid="select_columns_remove", allow_empty=False))

    result = RInterpreter.select(df, columns_keep=columns_keep, columns_remove=columns_remove)
    call_str = f"select({{inp1}}, columns_keep={columns_keep!r}, columns_remove={columns_remove!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphRLang.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    if choice:
        kept_cols = set(columns_keep)
        removed_cols = [c for c in df.columns if c not in kept_cols]

    else:
        removed_cols = set(columns_remove)
        kept_cols = [c for c in df.columns if c not in removed_cols]

    #  - EQUAL edges between the corresponding preserved columns.
    #  - EQUAL edges between the cells that are preserved.
    for c in kept_cols:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        for v1, v2 in zip(g_df.loc[:, c], g_res.loc[:, c]):
            added_edges.append(Edge(v1, v2, ELabel.EQUAL))

    #  - DELETE edges for the deleted columns and their cells.
    for c in removed_cols:
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

    if choice:
        for c_node in cands_cols:
            if c_node.value in columns_keep:
                tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@select_columns_keep"))
            else:
                tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@select_columns_keep"))

    else:
        for c_node in cands_cols:
            if c_node.value in columns_remove:
                tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="SELECTED@select_columns_remove"))
            else:
                tagged_edges.append(TaggedEdge(src=c_node, dst=c_node, tag="NOT_SELECTED@select_columns_remove"))

    for cand_choice in [True, False]:
        if cand_choice == choice:
            graph.add_tags([f"SELECTED@{cand_choice}@select_keep_or_remove"])
        else:
            graph.add_tags([f"NOT_SELECTED@{cand_choice}@select_keep_or_remove"])

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name='rf.mutate', group='rlang', strategy=DfsGraphStrategy())
def gen_mutate(df: pd.DataFrame, g_df: DataFrameGraph, datagen: bool = False):
    """
    MUTATE
    ------
    Example:
     mutate(df, new_col_name='C4', operation='normalize', col_args='C1')
      ---------------
                df                          result
        C1  C2  C3           C1  C2  C3         C4
     0   3   a   d  -->   0   3   a   d   0.250000
     1   4   b   e        1   4   b   e   0.333333
     2   5   c   f        2   5   c   f   0.416666

     mutate(df, new_col_name='C4', operation='div', col_args=['C1', 'C2'])
      ---------------
                 df                    result
        C1  C2   C3           C1  C2  C3   C4
     0   3   5    d  -->   0   3   a   d   0.6
     1   4   10   e        1   4   b   e   0.4
     2   5   20   f        2   5   c   f   0.25
    ---------------
    Graph Abstraction:
    - EQUAL edges between the all columns and cells in the input to the corresponding nodes of the output.
    - If normalize, SUM and DIV edges representing the computation. A unique sum intermediate node is created for each
      cell of the column being normalized.
    """

    cands_cols = g_df.columns
    operation = SelectConst(["normalize", "div"], uid="mutate_operation")

    new_col_name = FreshColumn(uid="mutate_new_col_name")
    if operation == "normalize":
        col_arg = SelectNode(cands_cols, uid="mutate_col_args_normalize")
        col_args = [col_arg]
        result = RInterpreter.mutate(df, new_col_name=new_col_name, operation=operation, col_args=col_arg)
        call_str = f"mutate({{inp1}}, new_col_name={new_col_name!r}, operation={operation!r}, col_args={col_arg!r})"

    else:
        col_arg1, col_arg2 = OrderedSubsetNode(cands_cols, uid="mutate_col_args_div",
                                               min_len=2, max_len=2)
        col_args = [col_arg1, col_arg2]
        result = RInterpreter.mutate(df, new_col_name=new_col_name, operation=operation, col_args=col_args)
        call_str = f"mutate({{inp1}}, new_col_name={new_col_name!r}, operation={operation!r}, col_args={col_args!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphRLang.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges between the corresponding columns as all columns are preserved.
    #  - EQUAL edges between the cells that are preserved.
    for c in col_map_df:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        for v1, v2 in zip(g_df.loc[:, c], g_res.loc[:, c]):
            added_edges.append(Edge(v1, v2, ELabel.EQUAL))

    if operation == 'normalize':
        summation = df[col_args[0]].sum()
        for cell_node, res_node in zip(g_df.loc[:, col_args[0]], g_res.loc[:, new_col_name]):
            #  Add the sum edges
            interm_node_sum = graph.create_intermediate_node(summation)
            for c in g_df.loc[:, col_args[0]]:
                added_edges.append(Edge(c, interm_node_sum, ELabel.SUM))

            interm_node_div = graph.create_intermediate_node(res_node.value)
            added_edges.append(Edge(interm_node_sum, interm_node_div, ELabel.DIV))
            added_edges.append(Edge(cell_node, interm_node_div, ELabel.DIV))
            added_edges.append(Edge(interm_node_div, res_node, ELabel.EQUAL))

    else:
        for cell_nodes, res_node in zip(g_df.loc[:, col_args].values, g_res.loc[:, new_col_name]):
            interm_node = graph.create_intermediate_node(res_node.value)
            added_edges.append(Edge(interm_node, res_node, ELabel.EQUAL))
            for n in cell_nodes:
                added_edges.append(Edge(n, interm_node, ELabel.DIV))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    tagged_edges: List[TaggedEdge] = []
    if operation == 'normalize':
        for c_node in cands_cols:
            if c_node.value in col_args:
                tagged_edges.append(TaggedEdge(c_node, c_node, "SELECTED@mutate_col_args_normalize"))
            else:
                tagged_edges.append(TaggedEdge(c_node, c_node, "NOT_SELECTED@mutate_col_args_normalize"))

    else:
        for c_node in cands_cols:
            if c_node.value in col_args:
                tagged_edges.append(TaggedEdge(c_node, c_node, "SELECTED@mutate_col_args_div"))
            else:
                tagged_edges.append(TaggedEdge(c_node, c_node, "NOT_SELECTED@mutate_col_args_div"))

    graph.add_tags([f"SELECTED@{cand}@mutate_operation"
                    if cand == operation else
                    f"NOT_SELECTED@{cand}@mutate_operation"
                    for cand in ["normalize", "div"]])

    graph.add_tagged_edges(tagged_edges)
    return result, call_str, graph, g_res


@generator(name='rf.filter', group='rlang', strategy=DfsGraphStrategy())
def gen_filter(df: pd.DataFrame, g_df: DataFrameGraph, datagen: bool = False):
    """
    FILTER
    ------
    Example:
     filter(df, 'C1 > 3')
      ---------------
                df               result
        C1  C2  C3           C1  C2  C3
     0   3   a   d  -->   0   4   b   e
     1   4   b   e        1   5   c   f
     2   5   c   f

    ---------------
    Graph Abstraction:
    - EQUAL edges between the all columns and cells in the input that are preserved to the
      corresponding nodes of the output.
    - Additional dependency edges from the cells of the column used in the filtering condition (C1 in the example).
    """

    cands_column = g_df.columns
    mode = SelectConst(["equality-inequality", "relop"], uid="filter_mode")

    if mode == "equality-inequality":
        column = SelectNode(cands_column, uid="filter_column_eq")
        all_values = set(df[column])
        value = SelectConst(list(all_values), uid="filter_value_eq")
        op = SelectConst(["==", "!="], uid="filter_eq_op")
        filter_expr = f"{column} {op} {value!r}"

    else:
        numeric_cols = set(df.select_dtypes('number').columns)
        column = SelectNode([c for c in cands_column if c.value in numeric_cols], uid="filter_column_relop")
        all_values = set(df[column])
        value = SelectConst(list(all_values), uid="filter_value_relop")
        op = SelectConst(["<", ">"], uid="filter_relop")
        filter_expr = f"{column} {op} {value!r}"

    result = RInterpreter.filter_(df, filter_expr, reset_index=False)

    filtered_indices = list(result.index)
    removed_indices = list(set(df.index) - set(filtered_indices))
    result = result.reset_index(drop=True)
    call_str = f"filter({{inp1}}, {filter_expr!r})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphRLang.assemble([g_df, g_res])
    added_edges: List[Edge] = []

    col_map_df = {c.value: c for c in g_df.columns}  # Map from df's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    if op == "==":
        dep_label = ELabel.DEPENDENT_EQ
    elif op == "!=":
        dep_label = ELabel.DEPENDENT_INEQ
    elif op == "<":
        dep_label = ELabel.DEPENDENT_LT
    else:
        dep_label = ELabel.DEPENDENT_GT

    deletion_node_out = g_res.deletion_node

    #  - EQUAL edges for all columns.
    for c in col_map_df:
        added_edges.append(Edge(col_map_df[c], col_map_res[c], ELabel.EQUAL))
        #  - EQUAL edges for kept rows
        for index, v1, v2 in zip(filtered_indices, g_df.loc[filtered_indices, c], g_res.loc[:, c]):
            interm_node = graph.create_intermediate_node(v2.value)
            added_edges.append(Edge(g_df.loc[index, column], interm_node, dep_label))
            added_edges.append(Edge(v1, interm_node, ELabel.EQUAL))
            added_edges.append(Edge(interm_node, v2, ELabel.EQUAL))

        #  - Mark the rest as deleted
        for v in g_df.loc[removed_indices, c]:
            added_edges.append(Edge(v, deletion_node_out, ELabel.DELETE))

    #  - EQUAL edge between the input deletion node and the output deletion node.
    added_edges.append(Edge(g_df.deletion_node, g_res.deletion_node, ELabel.EQUAL))

    #  Add all the edges to the graph in one go.
    graph.add_nodes_and_edges(edges=added_edges)

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Add information about arguments
    #  --------------------------------------------------------------------------------------------------------------  #

    tagged_edges: List[TaggedEdge] = []
    if mode == 'equality-inequality':
        for c_node in cands_column:
            if column == c_node.value:
                tagged_edges.append(TaggedEdge(c_node, c_node, "SELECTED@filter_column_eq"))
            else:
                tagged_edges.append(TaggedEdge(c_node, c_node, "NOT_SELECTED@filter_column_eq"))

        for cand_op in ["==", "!="]:
            if cand_op == op:
                graph.add_tags([f"SELECTED@{cand_op}@filter_eq_op"])
            else:
                graph.add_tags([f"NOT_SELECTED@{cand_op}@filter_eq_op"])

    else:
        for c_node in cands_column:
            if column == c_node.value:
                tagged_edges.append(TaggedEdge(c_node, c_node, "SELECTED@filter_column_relop"))
            else:
                tagged_edges.append(TaggedEdge(c_node, c_node, "NOT_SELECTED@filter_column_relop"))

        for cand_op in ["<", ">"]:
            if cand_op == op:
                graph.add_tags([f"SELECTED@{cand_op}@filter_relop"])
            else:
                graph.add_tags([f"NOT_SELECTED@{cand_op}@filter_relop"])

    graph.add_tagged_edges(tagged_edges)

    for cand_mode in ["equality-inequality", "relop"]:
        if cand_mode == mode:
            graph.add_tags([f"SELECTED@{cand_mode}@filter_mode"])
        else:
            graph.add_tags([f"NOT_SELECTED@{cand_mode}@filter_mode"])

    return result, call_str, graph, g_res


@generator(name='rf.inner_join', group='rlang', strategy=DfsGraphStrategy())
def gen_inner_join(df1: pd.DataFrame, df2: pd.DataFrame,
                   g_df1: DataFrameGraph, g_df2: DataFrameGraph, datagen: bool = False):
    """
    INNER_JOIN
    ------
    Example:
      inner_join(df1, df2)

        c1 c2 c3      c4 c2 c5          c1 c2 c3 c4 c5
      0  a  b  c    0  x  g  z        0  a  b  c  w  u
      1  d  g  c    1  w  b  u   ->   1  f  b  h  w  u
      2  f  b  h ,  2  y  g  j        2  d  g  c  x  z
                                      3  d  g  c  y  j

    ---------------
    Graph Abstraction:
    - EQUAL edges from all the columns in df1 and df2 to the corresponding column in the output.
    - EQUAL edges from all the rows in df1 and df2 to the corresponding row in the output, if included.
    - EQUAL edge between the input deletion nodes and the output deletion node.
    - DELETE edges from all the non-included cells in both df1 and df2 to the deletion node of the output.
    """

    result = RInterpreter.inner_join(df1, df2)
    call_str = f"inner_join({{inp1}}, {{inp2}})"

    #  --------------------------------------------------------------------------------------------------------------  #
    #  Graph Construction
    #  --------------------------------------------------------------------------------------------------------------  #

    g_res = DataFrameGraph(result)
    graph = GraphRLang.assemble([g_df1, g_df2, g_res])
    added_edges: List[Edge] = []

    col_map_df1 = {c.value: c for c in g_df1.columns}  # Map from df1's columns to their column nodes
    col_map_df2 = {c.value: c for c in g_df2.columns}  # Map from df2's columns to their column nodes
    col_map_res = {c.value: c for c in g_res.columns}  # Map from result's columns to their column nodes

    #  - EQUAL edges from all the columns in df1 and df2 to the corresponding column in the output.
    for c, node in itertools.chain(col_map_df1.items(), col_map_df2.items()):
        added_edges.append(Edge(node, col_map_res[c], ELabel.EQUAL))

    #  Get the merge cols
    merge_cols = list(set(col_map_df1.keys()) & set(col_map_df2.keys()))

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

    #  - EQUAL edges from all the rows in df1 and df2 to the corresponding row in the output, if included.
    #  - DELETE edges from all the non-included cells in both df1 and df2 to the deletion node of the output.
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

    #  No arguments for this component.
    return result, call_str, graph, g_res


generator_dict: Dict[str, Generator] = {
    'rf.gather': gen_gather,
    'rf.spread': gen_spread,
    'rf.unite': gen_unite,
    'rf.separate': gen_separate,
    'rf.group_by': gen_group_by_summarise,
    'rf.select': gen_select,
    "rf.mutate": gen_mutate,
    "rf.filter": gen_filter,
    "rf.inner_join": gen_inner_join,
}
