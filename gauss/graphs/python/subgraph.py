"""
Contains the core implementation of the subgraph isomorphism operation for graphs
"""
import collections
from typing import Optional, Iterator, Dict, Set, Iterable, Tuple, List, Union

import attr

from gauss.graphs.python.building_blocks import Entity, Node, SYMBOLIC_VALUE
from gauss.graphs.python.graph import Graph
from gauss.graphs.common.graphmapping import GraphMapping, GraphMultiMapping


@attr.s
class SearchState:
    """
    Helper class to store all the necessary information to guide the search.
    """
    worklist: List[Node] = attr.ib()
    current_mapping: GraphMapping = attr.ib()
    candidate_mappings: GraphMultiMapping = attr.ib()
    return_depth: int = attr.ib(init=False, default=-2)
    success_record: List[bool] = attr.ib(init=False, default=None)
    _already_mapped_nodes: Set[Node] = attr.ib(init=False, factory=set)
    _already_mapped_entities: Set[Entity] = attr.ib(init=False, factory=set)
    _assigned_at_depth: Dict[Union[Node, Entity], int] = attr.ib(init=False, factory=dict)
    _num_assignments: int = attr.ib(init=False, default=0)

    def __attrs_post_init__(self):
        self.success_record = [False] * len(self.worklist)
        for nk, nv in self.current_mapping.m_node.items():
            self._already_mapped_nodes.add(nv)
            self._already_mapped_entities.add(nv.entity)
            self._assigned_at_depth[nv] = -1
            self._assigned_at_depth[nv.entity] = -1

    def get_candidates(self, node: Node) -> Iterable[Node]:
        return (n for n in self.candidate_mappings.m_node[node] if n not in self._already_mapped_nodes)

    def get_original_candidates(self, node: Node) -> Iterable[Node]:
        return self.candidate_mappings.m_node[node]

    def entity_already_mapped(self, entity: Entity) -> bool:
        return entity in self._already_mapped_entities

    def node_already_mapped(self, node: Node) -> bool:
        return node in self._already_mapped_nodes

    def get_node_assignment_depth(self, node: Node) -> int:
        return self._assigned_at_depth[node]

    def get_entity_assignment_depth(self, entity: Entity) -> int:
        return self._assigned_at_depth[entity]

    def perform_assignment(self, depth: int, query_node: Node, graph_node: Node, entity_assigned: bool) -> None:
        self._num_assignments += 1
        self.current_mapping.m_node[query_node] = graph_node
        self._assigned_at_depth[graph_node] = depth
        self._already_mapped_nodes.add(graph_node)
        if entity_assigned:
            self.current_mapping.m_ent[query_node.entity] = graph_node.entity
            self._assigned_at_depth[graph_node.entity] = depth
            self._already_mapped_entities.add(graph_node.entity)

    def undo_assignment(self, depth: int, query_node: Node, graph_node: Node, entity_assigned: bool) -> None:
        self.current_mapping.m_node.pop(query_node)
        self._assigned_at_depth.pop(graph_node)
        self._already_mapped_nodes.discard(graph_node)
        if entity_assigned:
            self.current_mapping.m_ent.pop(query_node.entity)
            self._assigned_at_depth.pop(graph_node.entity)
            self._already_mapped_entities.discard(graph_node.entity)


def is_subgraph(query: Graph, graph: Graph, partial_mapping: Optional[GraphMapping] = None,
                entity_groups_query: Optional[Dict[Entity, int]] = None,
                entity_groups_graph: Optional[Dict[Entity, int]] = None) -> bool:
    try:
        _ = next(get_subgraph_mappings(query, graph, partial_mapping=partial_mapping,
                                       entity_groups_query=entity_groups_query,
                                       entity_groups_graph=entity_groups_graph))
        return True

    except StopIteration:
        return False


def get_subgraph_mappings(query: Graph, graph: Graph, partial_mapping: Optional[GraphMapping] = None,
                          entity_groups_query: Optional[Dict[Entity, int]] = None,
                          entity_groups_graph: Optional[Dict[Entity, int]] = None,
                          _worklist_order: List[Node] = None) -> Iterator[GraphMapping]:
    """
    Returns an iterator for all the possible subgraph isomorphism mappings between the nodes of `query` and the
    nodes of `graph`. A partial mapping may be provided binding certain nodes of `query` to certain nodes of `graph`.
    Args:
        query:
        graph:
        partial_mapping:
        entity_groups_query: Entity group info for `query`. Only entities belonging to the same group can be matched.
        entity_groups_graph: Entity group info for `graph`. Only entities belonging to the same group can be matched.
        _worklist_order: Only for debugging purposes

    Returns:

    """
    candidate_mappings = _get_candidate_mappings(query, graph, partial_mapping=partial_mapping,
                                                 entity_groups_query=entity_groups_query,
                                                 entity_groups_graph=entity_groups_graph)
    if candidate_mappings is None:
        return

    worklist, state = _init_state_and_worklist(query, graph, candidate_mappings, _worklist_order=_worklist_order)
    yield from _get_subgraph_mappings_recursive(worklist, query, graph, state=state, _depth=0)


def _init_state_and_worklist(query: Graph, graph: Graph,
                             candidate_mappings: GraphMultiMapping,
                             _worklist_order: List[Node]) -> Tuple[List[Node], SearchState]:
    """
    Helper function to initialize the search state and the worklist. The worklist starts of with an order that
    tries to maximize the information gathered from the initial assignments.

    Args:
        query:
        graph:
        candidate_mappings:
        _worklist_order: For debugging purposes

    Returns:

    """
    current_mapping = GraphMapping()
    worklist = []
    for k, v in candidate_mappings.m_node.items():
        if len(v) == 1:
            n_v = next(iter(v))
            current_mapping.m_node[k] = n_v
            current_mapping.m_ent[k.entity] = n_v.entity
        else:
            worklist.append(k)

    #  Set the initial order of nodes in the worklist based on the in/out degrees of the nodes
    #  Assigning nodes with high degrees first enables quick pruning of the space for the other nodes.
    #  NOTE : Should be cached ideally, but keeping it simple here.
    degree_counts = collections.Counter()
    for e in query.iter_edges():
        degree_counts[e.src] += 1
        degree_counts[e.dst] += 1

    if _worklist_order is None:
        worklist = sorted(worklist, key=lambda x: -degree_counts[x])
    else:
        worklist = sorted(worklist, key=lambda x: _worklist_order.index(x))

    state = SearchState(worklist, current_mapping, candidate_mappings)

    return worklist, state


def _get_subgraph_mappings_recursive(worklist: List[Node],
                                     query: Graph,
                                     graph: Graph,
                                     state: SearchState,
                                     _depth: int = 0) -> Iterator[GraphMapping]:
    """
    The recursive driver of the subgraph isomorphism finder.

    Args:
        worklist:
        query:
        graph:
        state:
        _depth: The current recursive depth, starts of with zero.

    Returns:

    """
    if _depth == len(worklist):
        #  Return a copy to safeguard from in-place editing
        yield state.current_mapping.copy()
        for i in range(len(worklist)):
            state.success_record[i] = True

        return

    current_mapping = state.current_mapping
    cur_node: Node = worklist[_depth]
    mapped_entity: Optional[Entity] = current_mapping.m_ent.get(cur_node.entity, None)
    entity_assigned_here: bool = mapped_entity is None
    failure_depth: int = -1
    state.success_record[_depth] = False

    for graph_node in state.get_candidates(cur_node):
        ok = True

        #  Check consistency with the current entity mapping
        if (not entity_assigned_here) and mapped_entity is not graph_node.entity:
            #  The decision point where the entity was actually assigned is a candidate for conflict analysis
            failure_depth = max(failure_depth, state.get_entity_assignment_depth(mapped_entity))
            continue

        elif entity_assigned_here and state.entity_already_mapped(graph_node.entity):
            #  The decision point where the entity was actually assigned is a candidate for conflict analysis
            failure_depth = max(failure_depth, state.get_entity_assignment_depth(graph_node.entity))
            continue

        #  Check consistency of the edge profile
        #  In principle, we could do something similar to unit propagation, which would update the mappings
        #  for all the other nodes, but that entails creation of temporary objects to a large extent,
        #  so we stick with on-demand checks. However, this would be desirable in C++. Resources on BCP in
        #  modern SAT solvers should be useful.
        for edge in query.iter_edges(src=cur_node):
            if edge.dst in current_mapping.m_node:
                #  Check if the edge is present in graph as well
                dst_mapped = current_mapping.m_node[edge.dst]
                if not graph.has_edge(src=graph_node, dst=dst_mapped, label=edge.label):
                    #  The decision point where the node was assigned to is a candidate for conflict analysis
                    failure_depth = max(failure_depth, state.get_node_assignment_depth(dst_mapped))
                    ok = False
                    break

            else:
                #  Check if the edge is present for one of the candidates of dst
                if all(not graph.has_edge(src=graph_node, dst=cand, label=edge.label)
                       for cand in state.get_candidates(edge.dst)):
                    #  Hard to say which decision point would have done this, so nothing on that front
                    #  TODO : Think about this
                    #  Being conservative for now
                    failure_depth = max(failure_depth, _depth - 1)
                    ok = False
                    break

        #  Move on to the next candidate if the check failed.
        if not ok:
            continue

        #  Do a similar check for the edges with dst as node
        for edge in query.iter_edges(dst=cur_node):
            if edge.src in current_mapping.m_node:
                #  Check if the edge is present in graph as well
                src_mapped = current_mapping.m_node[edge.src]
                if not graph.has_edge(src=src_mapped, dst=graph_node, label=edge.label):
                    #  The decision point where the node was assigned to is a candidate for conflict analysis
                    failure_depth = max(failure_depth, state.get_node_assignment_depth(src_mapped))
                    ok = False
                    break

            else:
                #  Check if the edge is present for one of the candidates of src
                if all(not graph.has_edge(src=cand, dst=graph_node, label=edge.label)
                       for cand in state.get_candidates(edge.src)):
                    #  Hard to say which decision point would have done this, so nothing on that front
                    #  TODO : Think about this
                    #  Being conservative for now
                    failure_depth = max(failure_depth, _depth - 1)
                    ok = False
                    break

        #  Move on to the next candidate if the check failed.
        if not ok:
            continue

        #  Update the mapping and move on to the next item on the worklist
        state.perform_assignment(_depth, cur_node, graph_node, entity_assigned=entity_assigned_here)
        yield from _get_subgraph_mappings_recursive(worklist, query, graph, state, _depth=_depth + 1)

        #  Rollback the assignment
        state.undo_assignment(_depth, cur_node, graph_node, entity_assigned=entity_assigned_here)

        if state.return_depth != -2:
            if _depth > state.return_depth:
                #  Pop the call stack further as the root cause of the conflict downstream is further up the call stack
                return

            else:
                #  We are at the right depth, reset.
                state.return_depth = -2

    if not state.success_record[_depth]:
        #  No combination of decisions from this point onwards yielded a solution.
        #  Perform conflict analysis to find the latest decision point which could influence the current point.
        #  Then pop the stack till that point. Also modify the worklist to push this decision point earlier so this
        #  conflict is solved first before making any decisions for other nodes.

        #  Was a viable candidate consumed at a previous decision point?
        for n in state.get_original_candidates(cur_node):
            if state.node_already_mapped(n):
                failure_depth = max(failure_depth, state.get_node_assignment_depth(n))

        state.return_depth = failure_depth
        if failure_depth == _depth - 1:
            state.return_depth = -2
        else:
            if failure_depth >= 0:
                #  Swap the worklist items
                worklist[failure_depth + 1], worklist[_depth] = worklist[_depth], worklist[failure_depth + 1]


def _get_candidate_mappings(query: Graph,
                            graph: Graph,
                            partial_mapping: Optional[GraphMapping] = None,
                            entity_groups_query: Optional[Dict[Entity, int]] = None,
                            entity_groups_graph: Optional[Dict[Entity, int]] = None) -> Optional[GraphMultiMapping]:
    """
    Given a `query` to check against a graph, this procedure returns the candidate mappings from the
    entities and nodes of `query` to the entities and nodes of `graph` respectively. This essentially
    establishes the search space for the isomorphisms. If there is no valid mapping, `None` is returned.

    Args:
        query: Query graph.
        graph: Graph to get the isomorphism mappings from query.
        partial_mapping: An existing mapping from entities and nodes of `query` to entities and nodes of `graph`.
        entity_groups_query: Entity group info for `query`. Only entities belonging to the same group can be matched.
        entity_groups_graph: Entity group info for `graph`. Only entities belonging to the same group can be matched.

    Returns:
        Optional[GraphMultiMapping]: The candidate mappings. `None` if no valid mapping exists.
    """

    candidates = GraphMultiMapping()
    candidates.m_ent.update({ent: None for ent in query.iter_entities()})
    candidates.m_node.update({node: None for node in query.iter_nodes()})

    if partial_mapping is not None:
        if not _init_candidate_mappings(candidates, partial_mapping):
            return None

    #  Stage 1 : Initial Unit Propagation
    #  Decide as much of the mapping as possible, starting with the partial mapping. If a node in `query` is forced
    #  to be assigned to a particular node in `graph`, called a `unit` node, use the edge-profile of that unit node
    #  to establish mappings of its neighbors. This may produce more `unit` nodes, for which we repeat the process.
    processed = set()
    if not _propagate_unit_nodes(candidates, query, graph, processed=processed):
        return None

    #  Stage 2 : Use neighbour profiles to find candidates for non-mapped nodes
    for n_query in query.iter_nodes():
        if candidates.m_node[n_query] is not None:
            continue

        #  Was not assigned yet. Get all the nodes matching the label, value and entity, if any.
        label = n_query.label
        value = None if n_query.value is SYMBOLIC_VALUE else n_query.value
        entities = candidates.m_ent.get(n_query.entity, None) or [None]
        cands = set()
        for entity in entities:
            cands.update(graph.iter_nodes(label=label, entity=entity, value=value))

        candidates.m_node[n_query] = cands

        #  Verify that the neighbour profiles of the candidates for n_query are consistent with the neighbour profile
        #  of n_query. A neighbour profile is simply a dictionary with counts for each edge type with the src/dst as
        #  n_query. The consistency criterion enforces that the number of edges of a certain type emanating from a
        #  candidate should be at least as large as the the number of edges of that type emanating from n_query.
        query_profile_src = collections.Counter(e.label for e in query.iter_edges(src=n_query))
        query_profile_dst = collections.Counter(e.label for e in query.iter_edges(dst=n_query))
        filtered_candidates = []
        for n_graph in candidates.m_node[n_query]:
            profile_src = collections.Counter(e.label for e in graph.iter_edges(src=n_graph))
            if any(profile_src[k] < v for k, v in query_profile_src.items()):
                continue

            profile_dst = collections.Counter(e.label for e in graph.iter_edges(dst=n_graph))
            if any(profile_dst[k] < v for k, v in query_profile_dst.items()):
                continue

            filtered_candidates.append(n_graph)

        if len(filtered_candidates) == 0:
            return None

        candidates.m_node[n_query].intersection_update(filtered_candidates)

    #  Stage 3 : Perform a final unit propagation.
    if not _propagate_unit_nodes(candidates, query, graph, processed=processed):
        return None

    #  Stage 4 : Final pruning using entity groups, if any.
    if entity_groups_query is not None:
        assert entity_groups_graph is not None, "Entity groups have to be supplied for both query and graph."
        candidates.m_node = {k: {n for n in v
                                 if entity_groups_query.get(k.entity, 0) == entity_groups_graph.get(n.entity, 0)}
                             for k, v in candidates.m_node.items()}
        if any(len(v) == 0 for v in candidates.m_node.values()):
            return None

    #  Stage 5 : Use Hopcroft-Karp maximum matching for bipartite-graphs to verify if a one-to-one mapping is possible
    #  TODO : Do if needed, doesn't affect correctness

    return candidates


def _propagate_unit_nodes(candidates: GraphMultiMapping,
                          query: Graph,
                          graph: Graph,
                          processed: Optional[Set[Node]],) -> bool:
    """
    The unit-propagation procedure. If a node is forced to be assigned to a single node, use the edge-profile of that
    node to establish mappings for its neighbours. This may result in more unit-nodes, for which we repeat the process.

    Args:
        candidates: The candidate mappings to use.
        query: The query graph
        graph: The graph the query is to be processed against.
        processed: The nodes which have already been processed and hence should be ignored.

    Returns:
        bool: `True` if successful, `False` if inconsistencies discovered.
    """

    if processed is None:
        processed = set()

    worklist = collections.deque(k for k, v in candidates.m_node.items()
                                 if v is not None and len(v) == 1 and k not in processed)
    while len(worklist) > 0:
        n_query = worklist.popleft()
        if n_query in processed:
            continue

        processed.add(n_query)
        n_graph = next(iter(candidates.m_node[n_query]))

        #  Use edge-profiles to narrow down possibilities for other nodes
        for edge in query.iter_edges(src=n_query):
            dst = edge.dst
            label = edge.label
            dst_candidates = {e.dst for e in graph.iter_edges(src=n_graph, label=label)
                              if e.dst.label == dst.label and (dst.value is SYMBOLIC_VALUE or dst.value == e.dst.value)}

            #  Compare with the existing set of mappings.
            if candidates.m_node[dst] is None:
                candidates.m_node[dst] = dst_candidates
            else:
                candidates.m_node[dst].intersection_update(dst_candidates)

            new_len = len(candidates.m_node[dst])
            if new_len == 0:
                return False
            elif new_len == 1:
                worklist.append(dst)

        for edge in query.iter_edges(dst=n_query):
            src = edge.src
            label = edge.label
            src_candidates = {e.src for e in graph.iter_edges(dst=n_graph, label=label)
                              if e.src.label == src.label and (src.value is SYMBOLIC_VALUE or src.value == e.src.value)}

            #  Compare with the existing set of mappings.
            if candidates.m_node[src] is None:
                candidates.m_node[src] = src_candidates
            else:
                candidates.m_node[src].intersection_update(src_candidates)

            new_len = len(candidates.m_node[src])
            if new_len == 0:
                return False
            elif new_len == 1:
                worklist.append(src)

    return True


def _init_candidate_mappings(candidates: GraphMultiMapping,
                             partial_mapping: GraphMapping) -> bool:
    """
    Initializes the candidate mappings from a given partial mapping, while checking for correctness
    Args:
        candidates: The candidate mappings to initialize
        partial_mapping: The partial mapping to use.

    Returns:
        bool: Whether successful or not.
    """

    m_ent: Dict[Entity, Entity] = {}
    m_node: Dict[Node, Node] = {}

    #  Add in the entity mappings from the partial mapping
    for ent_query, ent_graph in partial_mapping.m_ent.items():
        current = m_ent.get(ent_query, None)
        if current is not None and current is not ent_graph:
            return False

        m_ent[ent_query] = ent_graph

    #  Add in the node mappings from the partial mapping
    for n_query, n_graph in partial_mapping.m_node.items():
        current = m_node.get(n_query, None)
        if current is not None and current is not n_graph:
            return False

        #  Also check if entities match
        cur_entity = m_ent.get(n_query.entity, None)
        if cur_entity is not None and cur_entity is not n_graph.entity:
            return False

        m_node[n_query] = n_graph
        m_ent[n_query.entity] = n_graph.entity

    for k, v in m_ent.items():
        candidates.m_ent[k] = {v}
    for k, v in m_node.items():
        candidates.m_node[k] = {v}

    return True
