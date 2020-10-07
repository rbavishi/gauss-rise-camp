"""
Graph utilities specific to the rlang domain
"""

from enum import IntEnum

import attr
import pandas as pd
from typing import List, Any, Dict

from gauss.graphs import Graph, Node, Entity, Edge, SYMBOLIC_VALUE, DEFAULT_ENTITY


class NLabel(IntEnum):
    """
    Mappings from readable node labels to integers
    """
    COL = 0
    IDX = 1
    CELL = 2
    DELETED = 3
    INTERM = 4


class ELabel(IntEnum):
    """
    Mappings from readable edge labels to integers
    """
    EQUAL = 0

    #  Structural
    COLUMN = 1
    ROW = 2

    #  Computational
    STR_JOIN = 3
    STR_SPLIT = 4
    SUM = 5
    MEAN = 6
    COUNT = 7
    DIV = 8

    #  Relational Operators
    DEPENDENT_EQ = 9
    DEPENDENT_INEQ = 10
    DEPENDENT_LT = 11
    DEPENDENT_GT = 12

    #  Deletion
    DELETE = 13


@attr.s(cmp=False)
class GraphRLang(Graph):
    """ Convenience wrapper around the base graph for the RLang domain """

    @classmethod
    def assemble(cls, subgraphs: List[Graph]) -> 'GraphRLang':
        graph = cls()
        for s in subgraphs:
            graph.merge(s)

        return graph

    def create_intermediate_node(self, value):
        node = Node(label=NLabel.INTERM, entity=DEFAULT_ENTITY, value=value)
        self.add_node(node)
        return node


@attr.s(cmp=False)
class DataFrameGraph(Graph):
    """
    Creates the graph representation of a dataframe.
    Formally, the "phi"-abstraction of a dataframe.

    TODO : This is in the critical path, enumerating a program entails the creation of this graph
    TODO : for every output. So ideally we should have a c++ version of this as well. We defer this decision to
    TODO : see performance for the pandas domain and then we can see if this is really necessary.
    """
    df: pd.DataFrame = attr.ib()

    entity: Entity = attr.ib(init=False)
    columns: List[Node] = attr.ib(init=False)
    index: List[Node] = attr.ib(init=False)
    values: List[List[Node]] = attr.ib(init=False)
    deletion_node: Node = attr.ib(init=False)

    #  Mimic the iloc and loc accessors of Pandas dataframes for easy access to nodes/collections of nodes
    loc = attr.ib(init=False)
    iloc = attr.ib(init=False)

    #  Temporary caches
    _all_nodes: List[Node] = attr.ib(init=False, factory=list)
    _all_edges: List[Edge] = attr.ib(init=False, factory=list)

    def __attrs_post_init__(self):
        df = self.df
        self.entity = Entity(value=df)
        self.columns = [self._create_node(label=NLabel.COL, value=col) for col in df.columns]
        self.index = [self._create_node(label=NLabel.IDX, value=idx) for idx in df.index]
        self.values = [[self._create_node(label=NLabel.CELL, value=df.iloc[i, j]) for j in range(df.shape[1])]
                       for i in range(df.shape[0])]
        self.deletion_node = self._create_node(label=NLabel.DELETED, value=SYMBOLIC_VALUE)

        self.iloc = ILocIndexer(self)
        self.loc = LocIndexer(self)

        #  Structural Edges
        for i, col in enumerate(self.columns):
            for row in self.values:
                self._create_edge(col, row[i], ELabel.COLUMN)
                self._create_edge(row[i], col, ELabel.COLUMN)

        for i, idx in enumerate(self.index):
            for cell in self.values[i]:
                self._create_edge(idx, cell, ELabel.ROW)
                self._create_edge(cell, idx, ELabel.ROW)

        #  Bulk push
        self.add_nodes_and_edges(nodes=self._all_nodes, edges=self._all_edges)

    def _create_node(self, label: int, value: Any):
        node = Node(label=label, entity=self.entity, value=value)
        self._all_nodes.append(node)
        return node

    def _create_edge(self, src: Node, dst: Node, label: int):
        edge = Edge(src, dst, label)
        self._all_edges.append(edge)
        return edge


@attr.s
class ILocIndexer:
    df_graph = attr.ib()
    df = attr.ib(init=False)

    col_to_idx: Dict[Any, int] = attr.ib(init=False)
    index_to_idx: Dict[Any, int] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.df = self.df_graph.df
        self.col_to_idx = {c: idx for idx, c in enumerate(self.df.columns)}
        self.index_to_idx = {i: idx for idx, i in enumerate(self.df.index)}

    def __getitem__(self, item):
        df = self.df
        #  Get the result of executing this on the actual dataframe
        result = df.iloc[item]

        #  If the result is a dataframe, then there is a straightforward mapping
        if isinstance(result, pd.DataFrame):
            row_idxs = [self.index_to_idx[i] for i in result.index]
            col_idxs = [self.col_to_idx[i] for i in result.columns]

            return pd.DataFrame([[self.df_graph.values[i][j] for j in col_idxs] for i in row_idxs],
                                index=result.index, columns=result.columns)

        #  If Series, first figure out which axis we're slicing over, and return a series accordingly
        elif isinstance(result, pd.Series):
            if result.name in list(df.columns):
                row_idxs = [self.index_to_idx[i] for i in result.index]
                return pd.Series([self.df_graph.values[i][item[1]] for i in row_idxs], index=result.index)

            else:
                col_idxs = [self.col_to_idx[i] for i in result.index]
                return pd.Series([self.df_graph.values[item[0]][i] for i in col_idxs], index=result.index)

        #  Seems to be the singular value, fall back to regular indexing
        r_acc, c_acc = item
        return self.df_graph.values[r_acc][c_acc]


@attr.s
class LocIndexer:
    df_graph = attr.ib()
    df = attr.ib(init=False)

    col_to_idx: Dict[Any, int] = attr.ib(init=False)
    index_to_idx: Dict[Any, int] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.df = self.df_graph.df
        self.col_to_idx = {c: idx for idx, c in enumerate(self.df.columns)}
        self.index_to_idx = {i: idx for idx, i in enumerate(self.df.index)}

    def __getitem__(self, item):
        df = self.df
        #  Get the result of executing this on the actual dataframe
        result = df.loc[item]

        #  If the result is a dataframe, then there is a straightforward mapping
        if isinstance(result, pd.DataFrame):
            row_idxs = [self.index_to_idx[i] for i in result.index]
            col_idxs = [self.col_to_idx[i] for i in result.columns]

            return pd.DataFrame([[self.df_graph.values[i][j] for j in col_idxs] for i in row_idxs],
                                index=result.index, columns=result.columns)

        #  If Series, first figure out which axis we're slicing over, and return a series accordingly
        elif isinstance(result, pd.Series):
            if result.name in list(df.columns):
                row_idxs = [self.index_to_idx[i] for i in result.index]
                col = self.col_to_idx[item[1]]
                return pd.Series([self.df_graph.values[i][col] for i in row_idxs], index=result.index)

            else:
                col_idxs = [self.col_to_idx[i] for i in result.index]
                row = self.index_to_idx[item[0]]
                return pd.Series([self.df_graph.values[row][i] for i in col_idxs], index=result.index)

        #  Seems to be the singular value, fall back to regular indexing
        r_acc, c_acc = item
        r_acc = self.index_to_idx[r_acc]
        c_acc = self.col_to_idx[c_acc]
        return self.df_graph.values[r_acc][c_acc]
