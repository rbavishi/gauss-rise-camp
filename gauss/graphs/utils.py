from typing import Tuple

from gauss.graphs import Graph, Entity, SYMBOLIC_VALUE, Node, Edge
from gauss.graphs.common.graphmapping import GraphMapping


def create_symbolic_copy(graph: Graph) -> Tuple[Graph, GraphMapping]:
    mapping = GraphMapping()
    for entity in graph.iter_entities():
        mapping.m_ent[entity] = Entity(value=SYMBOLIC_VALUE)

    for node in graph.iter_nodes():
        mapping.m_node[node] = Node(label=node.label, entity=mapping.m_ent[node.entity], value=SYMBOLIC_VALUE)

    new_graph = Graph.from_nodes_and_edges(nodes=set(mapping.m_node.values()),
                                           edges={Edge(src=mapping.m_node[e.src], dst=mapping.m_node[e.dst],
                                                       label=e.label) for e in graph.iter_edges()})

    return new_graph, mapping
