"""
Graph routines/classes essential for synthesis
"""
import collections
import itertools
from typing import Optional, Collection, List, Type, Dict, Iterator, Tuple, Union

import attr

from gauss.graphs import Node, Graph, SYMBOLIC_VALUE, Edge, Entity
from gauss.graphs.common.graphmapping import GraphMapping

PLACEHOLDER_LABEL = 0xdeadbeef


@attr.s(cmp=False, repr=False)
class PlaceholderNode(Node):
    """
    An empty wrapper around Node, useful for precomputation of query plans
    """
    label = attr.ib(default=PLACEHOLDER_LABEL)
    value = attr.ib(default=SYMBOLIC_VALUE)


@attr.s(cmp=False, repr=False)
class Transformation(Graph):
    """
    Graph representing the transformation being performed by a program. That is, its graph abstraction.
    Just like a regular graph, but most importantly allows hashing / equality checking on top of other utilities.
    """

    #  Useful for caching
    _hash_value: Optional[int] = attr.ib(init=False, default=None)

    #  Meta-data about the transformation
    _input_entities: List[Entity] = attr.ib(init=False, default=None)
    _output_entity: Entity = attr.ib(init=False, default=None)
    _entity_groups: Dict[Entity, int] = attr.ib(init=False, default=None)

    @classmethod
    def build_from_graph(cls: Type['Transformation'],
                         graph: Graph,
                         input_entities: List[Entity], output_entity: Entity):

        result = cls.from_graph(graph)
        result._input_entities = list(input_entities)
        result._output_entity = output_entity
        result._entity_groups = {ent: 2 for ent in result.iter_entities()}
        for i in result._input_entities:
            result._entity_groups[i] = 0

        result._entity_groups[result._output_entity] = 1
        return result

    def add_nodes_and_edges(self, nodes: Optional[Collection[Node]] = None, edges: Optional[Collection[Edge]] = None):
        raise AssertionError("Transformations are read-only. Create a fresh-copy instead.")

    def deepcopy(self) -> Tuple['Transformation', GraphMapping]:
        copy: Transformation
        copy, ent_mapping, node_mapping = super().deepcopy()
        copy._input_entities = [ent_mapping[i] for i in self._input_entities]
        copy._output_entity = ent_mapping[self._output_entity]
        copy._entity_groups = {ent_mapping[k]: v for k, v in self._entity_groups.items()}

        return copy, GraphMapping(m_ent=ent_mapping, m_node=node_mapping)

    def reset_hash(self):
        #  Take the outgoing and ingoing edge counts by edge type for each node and take the hash of the
        #  resulting list of tuples (sorted of course).
        edge_counts_out = collections.defaultdict(collections.Counter)
        edge_counts_in = collections.defaultdict(collections.Counter)

        for edge in self.iter_edges():
            edge_counts_out[edge.src][edge.label] += 1
            edge_counts_in[edge.dst][edge.label] += 1

        neighbourhood_key = frozenset(collections.Counter((node.label,
                                                           frozenset(edge_counts_out[node].items()),
                                                           frozenset(edge_counts_in[node].items()))
                                                          for node in self.iter_nodes()).items())

        key = (neighbourhood_key, self.get_num_nodes(), self.get_num_edges())

        self._hash_value = hash(key)

    def is_subgraph(self, other: 'Transformation', partial_mapping: Optional[GraphMapping] = None,
                    **kwargs) -> bool:
        assert isinstance(other, Transformation), \
            "The argument to Transformation.is_subgraph should be another Transformation"

        return super().is_subgraph(graph=other,
                                   partial_mapping=partial_mapping,
                                   entity_groups_query=self._entity_groups,
                                   entity_groups_graph=other._entity_groups, **kwargs)

    def get_subgraph_mappings(self,
                              other: 'Transformation',
                              partial_mapping: Optional[GraphMapping] = None,
                              **kwargs) -> Iterator[GraphMapping]:

        assert isinstance(other, Transformation), \
            "The argument to Transformation.get_subgraph_mappings should be another Transformation"

        return super().get_subgraph_mappings(graph=other,
                                             partial_mapping=partial_mapping,
                                             entity_groups_query=self._entity_groups,
                                             entity_groups_graph=other._entity_groups,
                                             **kwargs)

    def get_input_entities(self) -> Collection[Entity]:
        return self._input_entities

    def get_output_entity(self) -> Entity:
        return self._output_entity

    def get_input_nodes(self) -> Iterator[Node]:
        yield from itertools.chain(*(self.iter_nodes(entity=ent) for ent in self._input_entities))

    def get_output_nodes(self) -> Iterator[Node]:
        yield from self.iter_nodes(entity=self._output_entity)

    def __hash__(self):
        if self._hash_value is None:
            self.reset_hash()

        return self._hash_value

    def __eq__(self, other):
        if not isinstance(other, Transformation):
            return False

        if hash(self) != hash(other):
            return False

        #  The following should be unnecessary as the number of nodes and edges is included in the hash,
        #  but still avoiding the possibility of a hash collision.
        if self.get_num_nodes() != other.get_num_nodes() or self.get_num_edges() != other.get_num_edges():
            return False

        return super().is_subgraph(graph=other,
                                   entity_groups_query=self._entity_groups,
                                   entity_groups_graph=other._entity_groups)

    def __setstate__(self, state):
        """
        Upon restoration from pickling, set the cached hash value to None.
        """

        self.__dict__.update(state)
        self._hash_value = None

    def __str__(self):
        return self.to_str(domain=None)

    def to_str(self, domain: Optional['SynthesisDomain'] = None):
        if domain is not None:
            nlabel_map: Dict[int, Union[str, int]] = domain.get_node_label_map()
            elabel_map: Dict[int, Union[str, int]] = domain.get_edge_label_map()

        else:
            nlabel_map: Dict[int, Union[str, int]] = {}
            elabel_map: Dict[int, Union[str, int]] = {}

        _idx_map_ent: Dict[Entity, int] = {ent: idx for idx, ent in enumerate(self.iter_entities())}
        _idx_map_node: Dict[Node, int] = {}
        _label_map: Dict[Node, str] = {}
        _role_map: Dict[Node, str] = {}

        for idx, n in enumerate(self.iter_nodes()):
            _idx_map_node[n] = idx
            _label_map[n] = str(nlabel_map.get(n.label, n.label))
            if n.entity in self._input_entities:
                role = "INPUT,"
            elif n.entity is self._output_entity:
                role = "OUTPUT,"
            else:
                role = ""

            _role_map[n] = role

        s = [f"Num Nodes : {self.get_num_nodes()}, Num Edges : {self.get_num_edges()}"]
        for n in self.iter_nodes():
            value = "" if n.value is SYMBOLIC_VALUE else f", {n.value}"
            n_str = f"{_role_map[n]}{_label_map[n]}, {_idx_map_ent[n.entity]}:{_idx_map_node[n]}{value}"
            s.append(n_str)
        s.append("----------")

        for e in self.iter_edges():
            s.append(f"{_role_map[e.src]}{_label_map[e.src]}, {_idx_map_ent[e.src.entity]}:{_idx_map_node[e.src]} "
                     f"-> "
                     f"{_role_map[e.dst]}{_label_map[e.dst]}, {_idx_map_ent[e.dst.entity]}:{_idx_map_node[e.dst]} "
                     f"({elabel_map.get(e.label, e.label)})")

        s.append("----------")
        return "\n".join(s)
