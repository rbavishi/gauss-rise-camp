import collections
from typing import List, Optional, Dict, Tuple, Set, Deque

from gauss.graphs.python.building_blocks import TaggedEdge, Node, Edge, Entity, SYMBOLIC_VALUE
from gauss.graphs.python.graph import Graph
from gauss.graphs.common.graphmapping import GraphMapping


def get_greatest_common_universal_supergraph(query: Graph,
                                             graphs: List[Graph],
                                             all_mappings: Optional[Dict[Graph, List[GraphMapping]]] = None,
                                             ) -> Tuple[Graph, GraphMapping]:
    """
    Returns the universal supergraph corresponding to greatest lower bound of all the maximal universal supergraphs of
    `query` w.r.t `graphs` in the partial order of universal supergraphs of `query` w.r.t `graphs`.

    Args:
        query: The query graph to find the universal subgraph for.
        graphs: The graphs w.r.t which the universal subgraph is to be computed.
        all_mappings: Mapping of graphs to subgraph isomorphism mappings of `query` for each graph in `graphs`.
            If None, they are computed by iterating over the result of `get_subgraph_mappings` till exhaustion.

    Returns:
        Tuple[Graph, GraphMapping]: The universal supergraph along with mappings to query.
            The mapping only contains nodes already present in `query`.
    """

    if all_mappings is None:
        all_mappings = {g: list(query.get_subgraph_mappings(g)) for g in graphs}
        #  Filter out the graphs in which query is not present at all
        graphs = [g for g in graphs if len(all_mappings[g]) != 0]
        all_mappings = {g: all_mappings[g] for g in graphs}
        assert len(all_mappings) > 0, "Did not find any graph which contains the query."

    #  We will use the first mapping for the first graph to incrementally construct the desired supergraph.
    #  The rationale is that since the universal supergraph needs to be consistent it all the mappings of
    #  all the graphs, we can use one mapping to grow the graph while using the others to check correctness of
    #  every incremental update.
    exemplar = all_mappings[graphs[0]][0].copy()

    #  Instead of tracking mappings w.r.t the query, track them w.r.t the exemplar mapping.
    work_mappings: Dict[Graph, List[GraphMapping]] = {g: [m.apply_mapping(exemplar, only_keys=True) for m in g_mappings]
                                                      for g, g_mappings in all_mappings.items()}

    known_nodes: Set[Node] = set(exemplar.m_node.values())
    orig_known_nodes: Set[Node] = known_nodes.copy()
    known_edges: Set[Edge] = {Edge(src=exemplar.m_node[e.src],
                                   dst=exemplar.m_node[e.dst],
                                   label=e.label) for e in query.iter_edges()}

    assert all(e.src in known_nodes and e.dst in known_nodes for e in known_edges)

    #  Maintain a worklist of edges with at least one end-point already known
    worklist: Deque[Edge] = collections.deque(e for e in graphs[0].iter_edges()
                                              if e not in known_edges and
                                              (e.src in known_nodes or e.dst in known_nodes))

    #  Also keep track of all the nodes mapped in every mapping
    already_mapped_dict: Dict[GraphMapping, Set[Node]] = {m: set(m.m_node.values())
                                                          for mappings in work_mappings.values()
                                                          for m in mappings}

    while len(worklist) > 0:
        edge = worklist.popleft()
        if edge in known_edges:
            continue

        if edge.src in known_nodes and edge.dst in known_nodes:
            #  Both end-points already present, simply check for the presence of this edge
            #  in all the graphs and w.r.t all the mappings for every graph.
            for graph, mappings in work_mappings.items():
                if any(not graph.has_edge(src=m.m_node[edge.src],
                                          dst=m.m_node[edge.dst],
                                          label=edge.label) for m in mappings):
                    break

            else:
                #  Did not break, so we can safely add this edge.
                known_edges.add(edge)

        elif edge.src in known_nodes:
            #  edge.dst is not known yet. Check if a counter-part of edge.dst exists for every mapping for every graph,
            #  such that there is an incoming edge of label edge.label with the counter-part of edge.src as the src.
            success = True
            counterparts_dict = {}
            for graph, mappings in work_mappings.items():

                for mapping in mappings:
                    already_mapped = already_mapped_dict[mapping]
                    #  Get the possible counter-parts.
                    candidates = _get_counterpart_candidates(graph, mapping, edge, already_mapped, known_src=True)
                    if len(candidates) == 0:
                        #  Can't extend this mapping, so this edge is not useful overall.
                        #  Exit out of all the loops.
                        success = False
                        break

                    else:
                        counterparts_dict[mapping] = candidates

                if not success:
                    break

            if not success:
                #  Can't do anything with this edge. Move on to the next item on the worklist.
                continue

            #  Can safely add this edge to the current supergraph. Adjust the meta-data being tracked.
            #  Specifically, extend all the mappings with the node corresponding to edge.dst
            work_mappings = _get_new_work_mappings(work_mappings,
                                                   counterparts_dict,
                                                   already_mapped_dict,
                                                   edge.dst)

            known_nodes.add(edge.dst)
            known_edges.add(edge)
            worklist.extend(graphs[0].iter_edges(src=edge.dst))
            worklist.extend(graphs[0].iter_edges(dst=edge.dst))

        elif edge.dst in known_nodes:
            #  Like above, but edge.src is unknown in this case.
            success = True
            counterparts_dict = {}
            for graph, mappings in work_mappings.items():

                for mapping in mappings:
                    already_mapped = already_mapped_dict[mapping]
                    #  Get the possible counter-parts.
                    candidates = _get_counterpart_candidates(graph, mapping, edge, already_mapped, known_src=False)
                    if len(candidates) == 0:
                        #  Can't extend this mapping, so this edge is not useful overall.
                        #  Exit out of all the loops.
                        success = False
                        break

                    else:
                        counterparts_dict[mapping] = candidates

                if not success:
                    break

            if not success:
                #  Can't do anything with this edge. Move on to the next item on the worklist.
                continue

            #  Can safely add this edge to the current supergraph. Adjust the meta-data being tracked.
            #  Specifically, extend all the mappings with the node corresponding to edge.src
            work_mappings = _get_new_work_mappings(work_mappings,
                                                   counterparts_dict,
                                                   already_mapped_dict,
                                                   edge.src)

            known_nodes.add(edge.src)
            known_edges.add(edge)
            worklist.extend(graphs[0].iter_edges(src=edge.src))
            worklist.extend(graphs[0].iter_edges(dst=edge.src))

    #  Similarly, try to extend the supergraph with graph-level tags and tagged edges as well.
    common_tags = set.intersection(*(set(g.iter_tags()) for g in graphs))
    common_tagged_edges: Set[TaggedEdge] = set()
    worklist_tagged = [e for e in graphs[0].iter_tagged_edges()
                       if e.src in known_nodes and e.dst in known_nodes]

    for tagged_edge in worklist_tagged:
        src = tagged_edge.src
        dst = tagged_edge.dst
        tag = tagged_edge.tag

        #  Check if this tagged edge is present in every graph for every mapping.
        for graph, mappings in work_mappings.items():
            if any(not graph.has_tagged_edge(src=m.m_node[src],
                                             dst=m.m_node[dst],
                                             tag=tag) for m in mappings):

                break

        else:
            common_tagged_edges.add(tagged_edge)

    #  At this point, we have all the nodes, edges, tags and tagged edges belonging to the supergraph.
    #  We now assemble the greatest common universal graph and the graph mapping w.r.t the query.
    universal_supergraph, mapping_wrt_exemplar = _create_symbolic_copy(Graph.from_nodes_and_edges(nodes=known_nodes,
                                                                                                  edges=known_edges))

    mapping_wrt_query = mapping_wrt_exemplar.slice(nodes=orig_known_nodes).apply_mapping(exemplar.reverse(),
                                                                                         only_keys=True)
    #  Add in the tags and tagged edges.
    universal_supergraph.add_tags(common_tags)
    universal_supergraph.add_tagged_edges(TaggedEdge(src=mapping_wrt_exemplar.m_node[e.src],
                                                     dst=mapping_wrt_exemplar.m_node[e.dst],
                                                     tag=e.tag)
                                          for e in common_tagged_edges)

    return universal_supergraph, mapping_wrt_query


def _get_new_work_mappings(work_mappings: Dict[Graph, List[GraphMapping]],
                           counterparts_dict: Dict[GraphMapping, List[Node]],
                           already_mapped_dict: Dict[GraphMapping, Set[Node]],
                           node_to_map: Node):
    new_work_mappings = {}
    for graph, mappings in work_mappings.items():
        new_mappings = []
        for mapping in mappings:
            counterparts = counterparts_dict[mapping]
            assert len(counterparts) > 0
            already_mapped = already_mapped_dict[mapping]
            for cand in counterparts:
                new_m = mapping.copy()
                new_m.m_node[node_to_map] = cand
                new_m.m_ent[node_to_map.entity] = cand.entity
                already_mapped_dict[new_m] = already_mapped | {cand}
                new_mappings.append(new_m)

        new_work_mappings[graph] = new_mappings[:1]
        for m in mappings:
            already_mapped_dict.pop(m)

    return new_work_mappings


def _get_counterpart_candidates(graph: Graph, mapping: GraphMapping, edge: Edge, already_mapped: Set[Node],
                                known_src: bool = True):
    if known_src:
        src = mapping.m_node[edge.src]
        dst = None
        entity = mapping.m_ent.get(edge.dst.entity, None)
    else:
        src = None
        dst = mapping.m_node[edge.dst]
        entity = mapping.m_ent.get(edge.src.entity, None)

    candidates: List[Node] = []
    for e in graph.iter_edges(src=src, dst=dst, label=edge.label):
        cand = e.dst if known_src else e.src
        if cand.label != edge.dst.label:
            continue

        if cand in already_mapped:
            continue

        if entity is not None and entity is not cand.entity:
            continue

        candidates.append(cand)

    return candidates


def _create_symbolic_copy(graph: Graph) -> Tuple[Graph, GraphMapping]:
    mapping = GraphMapping()
    for entity in graph.iter_entities():
        mapping.m_ent[entity] = Entity(value=SYMBOLIC_VALUE)

    for node in graph.iter_nodes():
        mapping.m_node[node] = Node(label=node.label, entity=mapping.m_ent[node.entity], value=SYMBOLIC_VALUE)

    new_graph = Graph.from_nodes_and_edges(nodes=set(mapping.m_node.values()),
                                           edges={Edge(src=mapping.m_node[e.src], dst=mapping.m_node[e.dst],
                                                       label=e.label) for e in graph.iter_edges()})

    return new_graph, mapping
