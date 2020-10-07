import collections
from typing import Any, Set, Optional, Iterable, Collection, Tuple, Dict, TypeVar, Iterator, List

import attr

from gauss.graphs import Entity, Node, Edge
from gauss.graphs.python.building_blocks import TaggedEdge

G = TypeVar("G", bound='Graph')


def _construct_collections_defaultdict_set():
    return collections.defaultdict(set)


@attr.s(cmp=False, repr=False)
class _NodeDatabase:
    _node_dict: Dict[Tuple[Optional[int], Optional[Entity]], Set[Node]] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._node_dict = collections.defaultdict(set)

    def iter_nodes(self, label: Optional[int] = None, entity: Optional[Entity] = None) -> Iterable[Node]:
        return self._node_dict[label, entity]

    def add_nodes(self, nodes: Collection[Node]):
        for node in nodes:
            self._node_dict[node.label, node.entity].add(node)
            self._node_dict[None, node.entity].add(node)
            self._node_dict[node.label, None].add(node)

    def copy(self):
        copied = _NodeDatabase()
        copied._node_dict = self._node_dict.copy()
        return copied


@attr.s(cmp=False, repr=False)
class _EdgeDatabase:
    _edge_dict: Dict[Tuple[Optional[Node], Optional[Node]],
                     Dict[Optional[int], Set[Edge]]] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._edge_dict = collections.defaultdict(_construct_collections_defaultdict_set)

    def iter_edges(self,
                   src: Optional[Node] = None,
                   dst: Optional[Node] = None,
                   label: Optional[int] = None) -> Iterable[Edge]:
        return self._edge_dict[src, dst][label]

    def add_edges(self, edges: Collection[Edge]):
        for edge in edges:
            src = edge.src
            dst = edge.dst
            label = edge.label

            d_src_dst = self._edge_dict[src, dst]
            d_src = self._edge_dict[src, None]
            d_dst = self._edge_dict[None, dst]

            d_src_dst[label].add(edge)
            d_src_dst[None].add(edge)

            d_src[label].add(edge)
            d_src[None].add(edge)

            d_dst[label].add(edge)
            d_dst[None].add(edge)

    def copy(self):
        copied = _EdgeDatabase()
        copied._edge_dict = collections.defaultdict(_construct_collections_defaultdict_set,
                                                    {k: v.copy() for k, v in self._edge_dict.items()})
        return copied


@attr.s(cmp=False, repr=False)
class Graph:
    """
    The basic graph.

    NOTE : No constructors exposed directly. All factory methods should be @classmethods and the name
    should reflect the information used to construct the graph. Also, these factory methods should avoid
    using the `add_nodes_and_edges` method as subclasses of Graph can choose to make graphs immutable by
    raising exceptions in that method.
    """

    _entities: Set[Entity] = attr.ib(init=False, factory=set)
    _nodes: Set[Node] = attr.ib(init=False, factory=set)
    _edges: Set[Edge] = attr.ib(init=False, factory=set)

    _tags: Set[str] = attr.ib(init=False, factory=set)
    _tagged_edges: Set[TaggedEdge] = attr.ib(init=False, factory=set)

    #  Data-structures for optimizing queries.
    _node_database: _NodeDatabase = attr.ib(init=False, factory=_NodeDatabase)
    _edge_database: _EdgeDatabase = attr.ib(init=False, factory=_EdgeDatabase)

    @classmethod
    def from_nodes_and_edges(cls, nodes: Collection[Node], edges: Collection[Edge]):
        """
        Args:
            nodes:
            edges:

        Returns:
            The created graph.

        """
        graph = cls()
        graph._nodes = set(nodes)
        graph._edges = set(edges)
        graph._entities = {n.entity for n in nodes}
        graph._node_database.add_nodes(nodes)
        graph._edge_database.add_edges(edges)
        return graph

    @classmethod
    def from_graph(cls, graph: 'Graph'):
        """
        Gives a chance to copy some of the internal data-structures directly to improve performance.
        Equivalent to a shallow copy.

        Args:
            graph:

        Returns:
            The created graph.

        """
        result = cls()
        result._nodes.update(graph._nodes)
        result._edges.update(graph._edges)
        result._entities.update(graph._entities)
        result._tags.update(graph._tags)
        result._tagged_edges.update(graph._tagged_edges)

        #  TODO : Copy the data-structures once the iter methods below are improved.
        result._node_database = graph._node_database.copy()
        result._edge_database = graph._edge_database.copy()

        return result

    def shallow_copy(self):
        return self.from_graph(self)

    def deepcopy(self) -> Tuple['Graph', Dict[Entity, Entity], Dict[Node, Node]]:
        ent_mapping: Dict[Entity, Entity] = {}
        node_mapping: Dict[Node, Node] = {}
        for entity in self._entities:
            ent_mapping[entity] = Entity(value=entity.value)

        for node in self._nodes:
            node_mapping[node] = Node(label=node.label, entity=ent_mapping[node.entity], value=node.value)

        new_graph = self.from_nodes_and_edges(nodes=set(node_mapping.values()),
                                              edges={Edge(src=node_mapping[e.src], dst=node_mapping[e.dst],
                                                          label=e.label) for e in self._edges})

        return new_graph, ent_mapping, node_mapping

    def add_nodes_and_edges(self, nodes: Optional[Collection[Node]] = None, edges: Optional[Collection[Edge]] = None):
        if nodes is not None:
            self._nodes.update(nodes)
            self._entities.update(n.entity for n in nodes)
            self._node_database.add_nodes(nodes)

        if edges is not None:
            self._edges.update(edges)
            self._edge_database.add_edges(edges)

    def add_node(self, node: Node) -> None:
        """
        Convenience method on top of add_nodes_and_edges.
        The bulk modification method `add_nodes_and_edges` should be the preferred way to minimize calls.
        Args:
            node: The node to add

        Returns:

        """

        self.add_nodes_and_edges(nodes=[node])

    def add_edge(self, edge: Edge) -> None:
        """
        Convenience method on top of add_nodes_and_edges.
        The bulk modification method `add_nodes_and_edges` should be the preferred way to minimize calls.
        Args:
            edge: The edge to add

        Returns:

        """

        self.add_nodes_and_edges(edges=[edge])

    def merge(self: G, other: 'Graph') -> G:
        """
        Convenience method on top of add_nodes_and_edges to absorb another graph into this one.
        Args:
            other: The graph whose nodes and edges need to be added to this graph.

        Returns:
            self: This graph itself
        """

        self.add_nodes_and_edges(nodes=other._nodes, edges=other._edges)
        self._tags.update(other._tags)
        self._tagged_edges.update(other._tagged_edges)
        return self

    def add_tags(self, tags: Iterable[str]):
        self._tags.update(tags)

    def add_tagged_edges(self, tagged_edges: Iterable[TaggedEdge]):
        self._tagged_edges.update(tagged_edges)

    def iter_tags(self) -> Iterator[str]:
        yield from self._tags

    def iter_tagged_edges(self, src: Optional[Node] = None, dst: Optional[Node] = None,
                          tag: Optional[str] = None) -> Iterator[TaggedEdge]:
        #  TODO : Improve performance
        edges = self._tagged_edges
        if src is not None:
            edges = (e for e in edges if e.src == src)

        if dst is not None:
            edges = (e for e in edges if e.dst == dst)

        if tag is not None:
            edges = (e for e in edges if e.tag == tag)

        return edges

    def induced_subgraph(self: G, keep_nodes: Collection[Node]) -> G:
        if not isinstance(keep_nodes, set):
            keep_nodes = set(keep_nodes)

        new_nodes = keep_nodes
        new_edges = {e for e in self._edges if e.src in keep_nodes and e.dst in keep_nodes}
        result: 'Graph' = self.from_nodes_and_edges(nodes=new_nodes, edges=new_edges)
        result._tags.update(self._tags)
        result._tagged_edges.update(e for e in self._tagged_edges if e.src in keep_nodes and e.dst in keep_nodes)
        return result

    def get_num_nodes(self) -> int:
        return len(self._nodes)

    def get_num_edges(self) -> int:
        return len(self._edges)

    def get_all_nodes(self) -> Collection[Node]:
        return self._nodes

    def get_all_edges(self) -> Collection[Edge]:
        return self._edges

    def iter_entities(self) -> Iterable[Entity]:
        return self._entities

    def iter_nodes(self, label: Optional[int] = None, entity: Optional[Entity] = None,
                   value: Any = None) -> Iterable[Node]:

        if label is None and entity is None and value is None:
            return self._nodes

        nodes = self._node_database.iter_nodes(label=label, entity=entity)

        if value is not None:
            nodes = (n for n in nodes if n.value == value)

        return nodes

    def iter_edges(self,
                   src: Optional[Node] = None,
                   dst: Optional[Node] = None,
                   label: Optional[int] = None) -> Iterable[Edge]:

        if src is None and dst is None and label is None:
            return self._edges

        #  TODO : Improve performance
        edges = self._edges
        if src is not None:
            edges = (e for e in edges if e.src is src)
        if dst is not None:
            edges = (e for e in edges if e.dst is dst)
        if label is not None:
            edges = (e for e in edges if e.label == label)

        return edges

    def has_edge(self, src: Node, dst: Node, label: int) -> bool:
        return (src, dst, label) in self._edges

    def has_tagged_edge(self, src: Node, dst: Node, tag: str) -> bool:
        return (src, dst, tag) in self._tagged_edges

    def is_subgraph(self, graph: 'Graph', *args, **kwargs):
        return is_subgraph(self, graph, *args, **kwargs)

    def get_subgraph_mappings(self, graph: 'Graph', *args, **kwargs):
        return get_subgraph_mappings(self, graph, *args, **kwargs)

    def get_greatest_common_universal_supergraph(self,
                                                 graphs: List['Graph'],
                                                 all_mappings: Optional[Dict['Graph', List['GraphMapping']]] = None):
        return get_greatest_common_universal_supergraph(self, graphs, all_mappings=all_mappings)


#  Hack to avoid cyclic imports
from gauss.graphs.python.subgraph import is_subgraph, get_subgraph_mappings
from gauss.graphs.python.universal_supergraph import get_greatest_common_universal_supergraph
