from typing import Set, Dict, List, Type

import attr

from gauss.graphs import Graph, TaggedEdge, Edge
from gauss.graphs.common.graphmapping import GraphMapping
from gauss.synthesis.graphs import Transformation
from gauss.synthesis.queries import Query
from gauss.synthesis.query_plans import QueryPlan, QueryPlans


@attr.s(cmp=False, repr=False)
class WaypointConstraint:
    depth: int = attr.ib()  # 0-indexed
    query: Query = attr.ib()
    query_plans: Set[QueryPlan] = attr.ib()
    mappings_dict: Dict[QueryPlan, List[GraphMapping]] = attr.ib()

    _checks: Dict[QueryPlan, List[GraphMapping]] = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        #  `mappings_dict` contains the *full* mappings. We need to slice it for this depth to constitute a valid check
        for plan, mappings in self.mappings_dict.items():
            unit = plan.units[self.depth]
            inp_nodes = set(unit.get_input_nodes())
            inp_connections = plan.all_connections.slice(nodes=inp_nodes)
            partial_mappings = [inp_connections.apply_mapping(m, only_values=True) for m in mappings]
            self._checks[plan] = partial_mappings

    def check(self, graph: Transformation) -> bool:
        for plan, partial_mappings in self._checks.items():
            t: Transformation = plan.units[self.depth]
            if any(t.is_subgraph(graph, partial_mapping=m) for m in partial_mappings):
                return True

        return False

    def step(self, graph: Transformation) -> 'WaypointConstraint':
        remaining_plans: Set[QueryPlan] = set()
        new_mappings_dict: Dict[QueryPlan, List[GraphMapping]] = {}
        for plan, partial_mappings in self._checks.items():
            full_mappings = self.mappings_dict[plan]
            new_mappings: List[GraphMapping] = []
            t: Transformation = plan.units[self.depth]

            for full_mapping, partial_mapping in zip(full_mappings, partial_mappings):
                for m in t.get_subgraph_mappings(graph, partial_mapping=partial_mapping):
                    new_mappings.append(full_mapping | m)

            if len(new_mappings) > 0:
                new_mappings_dict[plan] = new_mappings
                remaining_plans.add(plan)

        return WaypointConstraint(depth=self.depth + 1, query=self.query, query_plans=remaining_plans,
                                  mappings_dict=new_mappings_dict)

    def get_strengthening_constraint(self, input_graph: Graph) -> Graph:
        common_tags = None
        common_edges = None
        common_tagged_edges = None

        for plan, partial_mappings in self._checks.items():
            strengthening, s_mapping = plan.strengthenings[self.depth]
            s_edges = strengthening.get_all_edges()
            s_tagged_edges = set(strengthening.iter_tagged_edges())

            plan_tags = set(strengthening.iter_tags())
            plan_tagged_edges = None
            plan_edges = None
            for partial_mapping in partial_mappings:
                mapping_wrt_inp_graph = partial_mapping.apply_mapping(s_mapping, only_keys=True)
                for m in strengthening.get_subgraph_mappings(input_graph, partial_mapping=mapping_wrt_inp_graph):
                    if plan_tagged_edges is None:
                        plan_tagged_edges = {TaggedEdge(m.m_node[e.src], m.m_node[e.dst], e.tag)
                                             for e in s_tagged_edges}
                        plan_edges = {Edge(m.m_node[e.src], m.m_node[e.dst], e.label)
                                      for e in s_edges}
                    else:
                        plan_tagged_edges.intersection_update(TaggedEdge(m.m_node[e.src], m.m_node[e.dst], e.tag)
                                                              for e in s_tagged_edges)
                        plan_edges.intersection_update(Edge(m.m_node[e.src], m.m_node[e.dst], e.label)
                                                       for e in s_edges)

            if common_tags is None:
                common_tags = plan_tags or set()
                common_tagged_edges = plan_tagged_edges or set()
                common_edges = plan_edges or set()

            else:
                common_tags.intersection_update(plan_tags)
                common_tagged_edges.intersection_update(plan_tagged_edges)
                common_edges.intersection_update(plan_edges)

        nodes = {e.src for e in common_tagged_edges}
        nodes.update(e.dst for e in common_tagged_edges)
        nodes.update(e.src for e in common_edges)
        nodes.update(e.dst for e in common_edges)

        result = Graph.from_nodes_and_edges(nodes=nodes, edges=common_edges)
        result.add_tagged_edges(common_tagged_edges)
        result.add_tags(common_tags)
        return result


@attr.s
class Waypoints:
    depth: int = attr.ib()
    constraints: Dict[Query, WaypointConstraint] = attr.ib()

    def check(self, graph: Transformation) -> bool:
        return all(c.check(graph) for c in self.constraints.values())

    def step(self, graph: Transformation) -> 'Waypoints':
        cls: Type[Waypoints] = type(self)
        return cls(depth=self.depth + 1,
                   constraints={q: constraint.step(graph) for q, constraint in self.constraints.items()})

    @classmethod
    def initialize_from_query_plans(cls, query_plans: QueryPlans):
        depth: int = 0
        constraints = {query: WaypointConstraint(depth=depth,
                                                 query=query,
                                                 query_plans=plans,
                                                 mappings_dict={plan: [GraphMapping()] for plan in plans})
                       for query, plans in query_plans}
        return Waypoints(depth=depth, constraints=constraints)

    def get_strengthening_constraint(self, input_graph: Graph) -> Graph:
        strengthened_input_graph = Graph()
        for constraint in self.constraints.values():
            strengthened_input_graph.merge(constraint.get_strengthening_constraint(input_graph))

        return strengthened_input_graph
