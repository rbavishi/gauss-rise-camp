import collections
from typing import Optional, Set

from gauss.graphs import Graph, Edge, Node


def equality_transitive_closure(graph: Graph, equality_label: int,
                                join_nodes: Optional[Set[Node]] = None,
                                valid_combinations: Optional[Set[int]] = None):
    worklist = collections.deque(e for e in graph.iter_edges(label=equality_label))
    seen_edges = set(worklist)

    while len(worklist) > 0:
        edge_item = worklist.popleft()
        added = set()
        if join_nodes is None or edge_item.src in join_nodes:
            for e in graph.iter_edges(dst=edge_item.src):
                if valid_combinations is None or e.label in valid_combinations:
                    added.add(Edge(e.src, edge_item.dst, e.label))

        if join_nodes is None or edge_item.dst in join_nodes:
            for e in graph.iter_edges(src=edge_item.dst):
                if valid_combinations is None or e.label in valid_combinations:
                    added.add(Edge(edge_item.src, e.dst, e.label))

        added -= seen_edges
        if len(added) > 0:
            graph.add_nodes_and_edges(edges=added)
            for e in added:
                if e.label == equality_label:
                    worklist.append(e)

                seen_edges.add(e)
