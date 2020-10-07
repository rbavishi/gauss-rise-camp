import collections
import itertools

from typing import Dict, List, Set, Tuple, FrozenSet, Any

import attr

from gauss.graphs import Entity, SYMBOLIC_VALUE, Node, Edge, Graph
from gauss.graphs.common.graphmapping import GraphMapping
from gauss.graphs.utils import create_symbolic_copy
from gauss.synthesis.graphs import PlaceholderNode, Transformation
from gauss.synthesis.problem import SynthesisProblem
from gauss.synthesis.skeleton import Skeleton

Path = Tuple[Set[Node], List[Edge]]


@attr.s(cmp=False)
class ArgumentNumberMapping:
    mapping: Dict[int, int] = attr.ib()
    _hash_value: int = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):
        self._hash_value = hash(frozenset(self.mapping.items()))

    def __getitem__(self, item):
        return self.mapping[item]

    def __hash__(self):
        return self._hash_value

    def __eq__(self, other):
        if not isinstance(other, ArgumentNumberMapping):
            return False

        return self.mapping == other.mapping

    def apply_list(self, items: List[Any]):
        """
        Permute the list using the mapping.
        Args:
            items: The list to be permuted.

        Returns:

        """
        assert len(items) == len(self.mapping), "List of items has to the same as the size of the mapping"
        return [items[self.mapping[i]] for i in range(0, len(items))]

    def apply_skeleton(self, skeleton: Skeleton):
        ext_inp_dict = {(-k - 1): (-v - 1) for k, v in self.mapping.items()}
        new_skeleton = Skeleton([(func, [ext_inp_dict.get(i, i) for i in arg_ints]) for func, arg_ints in skeleton])
        return new_skeleton


@attr.s(cmp=False, repr=False)
class Query:
    transformation: Transformation = attr.ib()
    subgraph: Transformation = attr.ib()
    mapping: GraphMapping = attr.ib()
    arg_number_mapping: ArgumentNumberMapping = attr.ib()


def extract_paths(graph: Graph, input_entities: List[Entity], output_entity: Entity):
    path_dict: Dict[Entity, List[Path]] = {ent: [] for ent in input_entities}
    for node in itertools.chain(*(graph.iter_nodes(entity=ent) for ent in input_entities)):
        #  Find all the paths from node to an output node, without any other input or output nodes in between.
        #  An entry is the set of visited nodes, the current node to explore, and the current set of edges.
        entry: Tuple[Set[Node], Node, List[Edge]] = ({node}, node, [])
        worklist = collections.deque([entry])
        paths: List[Path] = []
        while len(worklist) > 0:
            visited, cur_node, edges = worklist.popleft()
            for edge in graph.iter_edges(src=cur_node):
                dst = edge.dst
                if dst in visited or dst.entity in input_entities:
                    continue

                if dst.entity is output_entity:
                    paths.append((visited | {dst}, edges + [edge]))
                else:
                    worklist.append((visited | {dst}, dst, edges + [edge]))

        path_dict[node.entity].extend(paths)

    return path_dict


def extract_queries(problem: SynthesisProblem) -> Dict[Transformation, List[Query]]:
    #  Stage 1 : Get all paths from one of the input nodes to an output node without any input/output node in between.
    graph = problem.graph
    inputs = problem.inputs
    output = problem.output

    entities = list(graph.iter_entities())
    input_entities = [next(ent for ent in entities if ent.value is inp) for inp in inputs]
    output_entity = next(ent for ent in entities if ent.value is output)
    arg_numbering = {ent: idx for idx, ent in enumerate(input_entities)}

    placeholder_dict = {}
    #  A placeholder node can represent any node belonging to an entity.
    #  This helps coalesce equivalent query plans.
    for ent in itertools.chain(input_entities, [output_entity]):
        placeholder_dict[ent] = PlaceholderNode(entity=ent)

    path_dict: Dict[Entity, List[Path]] = extract_paths(graph, input_entities, output_entity)

    canonical_transformations: Dict[Transformation, Transformation] = {}
    #  Only keep one transformation for a set of nodes as satisfying that query means satisfying all the others.
    seen: Set[Tuple[Transformation, FrozenSet[Node]]] = set()

    queries: Dict[Transformation, List[Query]] = collections.defaultdict(list)
    set_input_entities = set(input_entities)

    #  Find queries by taking exactly one path, and placeholder nodes for the
    #  input entities not present in the path.
    for path_ent, paths in path_dict.items():
        remaining_entities = [ent for ent in set_input_entities if ent is not path_ent]
        for path in paths:
            path_nodes, path_edges = path
            nodes = list(path_nodes) + [placeholder_dict[ent] for ent in remaining_entities]
            edges = path_edges

            #  Get the corresponding subgraph.
            subgraph = Graph.from_nodes_and_edges(nodes=set(nodes), edges=set(edges))
            subgraph_transformation = Transformation.build_from_graph(subgraph,
                                                                      input_entities=input_entities,
                                                                      output_entity=output_entity)

            #  Compute the symbolic counter-part to group subgraphs together by the underlying transformation.
            symbolic_copy, mapping = create_symbolic_copy(subgraph)
            mapped_input_entities = [mapping.m_ent[i] for i in input_entities]
            mapped_output_entity = mapping.m_ent[output_entity]
            transformation = Transformation.build_from_graph(symbolic_copy,
                                                             input_entities=mapped_input_entities,
                                                             output_entity=mapped_output_entity)

            #  Check if the transformation was seen before
            if transformation not in canonical_transformations:
                canonical_transformations[transformation] = transformation
                seen.add((transformation, frozenset(n for n in subgraph.iter_nodes()
                                                    if n.entity in set_input_entities)))

                mapping = mapping.reverse()
                arg_number_mapping = ArgumentNumberMapping({idx: arg_numbering[mapping.m_ent[ent]]
                                                            for idx, ent in enumerate(mapped_input_entities)})

                # We need a mapping from transformation to the subgraph.
                queries[transformation].append(Query(transformation=transformation,
                                                     subgraph=subgraph_transformation,
                                                     mapping=mapping,
                                                     arg_number_mapping=arg_number_mapping))

            else:
                canonical = canonical_transformations[transformation]
                key = (transformation, frozenset(n for n in subgraph.iter_nodes() if n.entity in set_input_entities))
                #  Check if the transformation was seen before with the same input nodes. If yes, continue. This is
                #  because if a graph satisfies the already seen transformation for these input nodes, it will satisfy
                #  this one as well. So no point in checking it.
                if key in seen:
                    continue

                seen.add(key)
                # We need a mapping from the canonical transformation to the subgraph.
                mapping = next(canonical.get_subgraph_mappings(transformation)).apply_mapping(mapping.reverse())
                arg_number_mapping = ArgumentNumberMapping({idx: arg_numbering[mapping.m_ent[ent]]
                                                            for idx, ent in enumerate(canonical.get_input_entities())})

                queries[canonical].append(Query(transformation=canonical,
                                                subgraph=subgraph_transformation,
                                                mapping=mapping,
                                                arg_number_mapping=arg_number_mapping))

    return queries
