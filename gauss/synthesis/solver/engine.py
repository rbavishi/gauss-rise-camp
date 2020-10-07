import time
from typing import Dict, List, Any, Tuple, Iterator, Optional

import attr
import autopep8

from gauss.graphs import Graph
from gauss.synthesis.config import EngineConfig
from gauss.synthesis.deduction.engine import DeductionEngine
from gauss.synthesis.domains import SynthesisDomain, EnumerationItem, Solution
from gauss.synthesis.graphs import Transformation, PlaceholderNode
from gauss.synthesis.problem import SynthesisProblem
from gauss.synthesis.queries import extract_queries
from gauss.synthesis.query_plans import QueryPlans
from gauss.synthesis.skeleton import Skeleton
from gauss.synthesis.solver.waypoints import Waypoints
from gauss.utilities.logutils import logger


@attr.s
class SolverContext:
    #  General information
    domain: SynthesisDomain = attr.ib()
    config: EngineConfig = attr.ib()
    problem: SynthesisProblem = attr.ib()

    #  Specific skeleton + plans information
    skeleton: Skeleton = attr.ib()
    query_plans: QueryPlans = attr.ib()

    #  Helpful data-structures to maintain during enumeration
    int_to_val: Dict[int, Any] = attr.ib()  # Skeleton argument integers to values
    int_to_graph: Dict[int, Graph] = attr.ib()  # Skeleton argument integers to graph abstractions of values
    waypoints: List[Optional[Waypoints]] = attr.ib()  # Waypoints for each depth.
    enumeration_items: List[Optional[EnumerationItem]] = attr.ib()
    graphs: List[Optional[Graph]] = attr.ib()

    def get_arguments(self, depth: int) -> Tuple[List[Any], List[Graph]]:
        #  Convert arguments represented as integers to actual values along with their graph abstractions
        component_name, arg_ints = self.skeleton[depth]
        return [self.int_to_val[i] for i in arg_ints], [self.int_to_graph[i] for i in arg_ints]

    def check_validity(self, graph: Transformation, depth: int) -> bool:
        return self.waypoints[depth].check(graph)

    def step(self, output: Any, graph: Transformation, output_graph: Graph,
             enumeration_item: EnumerationItem,
             depth: int):
        #  Depth is assumed to be 0-indexed.
        self.int_to_val[depth + 1] = output
        self.int_to_graph[depth + 1] = output_graph

        if depth < self.skeleton.length - 1:
            self.waypoints[depth + 1] = self.waypoints[depth].step(graph)

        self.enumeration_items[depth] = enumeration_item
        self.graphs[depth] = graph

    def prepare_solution(self, output: Any, output_graph: Graph) -> Solution:
        if self.problem.input_names is not None:
            int_to_names: Dict[int, str] = {-idx: name for idx, name in enumerate(self.problem.input_names, 1)}
        else:
            int_to_names: Dict[int, str] = {-idx: f"inp{idx}" for idx in range(1, len(self.problem.inputs) + 1)}

        int_to_names[self.skeleton.length] = self.problem.output_name

        graph = Graph()
        for g in self.graphs:
            graph.merge(g)

        #  Perform transitive closure w.r.t the nodes corresponding to the intermediate outputs
        #  and take the induced subgraph containing all nodes except those
        if self.skeleton.length > 1:
            join_nodes = set.union(*(set(self.int_to_graph[i].iter_nodes()) for i in range(1, self.skeleton.length)))
            self.domain.perform_transitive_closure(graph, join_nodes=join_nodes)
            graph = graph.induced_subgraph(keep_nodes=set(graph.iter_nodes()) - join_nodes)

        return self.domain.prepare_solution(self.problem.inputs,
                                            output,
                                            graph,
                                            self.problem.graph_inputs,
                                            output_graph,
                                            self.enumeration_items,
                                            arguments=[arg_ints for (comp_name, arg_ints) in self.skeleton],
                                            int_to_names=int_to_names,
                                            int_to_obj=self.int_to_val)

    @classmethod
    def build(cls,
              domain: SynthesisDomain,
              config: EngineConfig,
              problem: SynthesisProblem,
              skeleton: Skeleton,
              query_plans: QueryPlans):

        int_to_val = {-idx: val for idx, val in enumerate(problem.inputs, 1)}
        int_to_graph = {-idx: graph for idx, graph in enumerate(problem.graph_inputs, 1)}
        waypoints = [None] * skeleton.length
        waypoints[0] = Waypoints.initialize_from_query_plans(query_plans)

        return SolverContext(
            domain=domain,
            config=config,
            problem=problem,
            skeleton=skeleton,
            query_plans=query_plans,
            int_to_val=int_to_val,
            int_to_graph=int_to_graph,
            waypoints=waypoints,
            enumeration_items=[None] * skeleton.length,
            graphs=[None] * skeleton.length,
        )


@attr.s
class SynthesisEngine:
    _domain: SynthesisDomain = attr.ib()
    config: EngineConfig = attr.ib()
    _deduction_engine: DeductionEngine = attr.ib()

    _time_start = attr.ib(init=False)

    def solve(self, problem: SynthesisProblem):
        queries = extract_queries(problem)
        logger.opt(colors=True).debug(f"Found <green>{sum(len(v) for v in queries.values())}</green> queries "
                                      f"with <green>{len(queries)}</green> distinct transformations.")

        self._time_start = time.time()
        for length in range(self.config.min_length, self.config.max_length + 1):
            for sequence in self._deduction_engine.get_candidate_sequences(self._domain,
                                                                           problem, queries, length):
                for skeleton, query_plans in self._deduction_engine.get_query_plans(sequence, problem, queries):
                    logger.opt(colors=True).debug(f"Trying Skeleton <blue>{skeleton}</blue>")
                    for solution in self._solve_for_skeleton(problem, skeleton, query_plans):
                        yield solution
                        self._time_start = time.time()

    def _solve_for_skeleton(self, problem: SynthesisProblem, skeleton: Skeleton, query_plans: QueryPlans):
        context = SolverContext.build(
            domain=self._domain,
            config=self.config,
            problem=problem,
            skeleton=skeleton,
            query_plans=query_plans,
        )

        for final_output, output_graph in self._solve_for_skeleton_recursive(problem=problem,
                                                                             skeleton=skeleton,
                                                                             query_plans=query_plans,
                                                                             context=context,
                                                                             _depth=0):
            if problem.check_strict:
                checker = self._domain.check_equivalent
            else:
                checker = self._domain.check_partial

            if checker(problem.output, final_output):
                solution = context.prepare_solution(final_output, output_graph)
                yield solution

    def _solve_for_skeleton_recursive(self,
                                      problem: SynthesisProblem,
                                      skeleton: Skeleton,
                                      query_plans: QueryPlans,
                                      context: SolverContext,
                                      _depth: int = 0) -> Iterator[Tuple[Any, Graph]]:

        domain = self._domain
        component_name, arg_ints = skeleton[_depth]
        inputs, g_inputs = context.get_arguments(depth=_depth)
        inp_entities = [next(iter(g_inp.iter_entities())) for g_inp in g_inputs]
        inp_graph = Graph()
        for g_inp in g_inputs:
            inp_graph.merge(g_inp)

        #  Get the strengthening constraint for this depth.
        #  Specifically, for every query, get the intersection of the strengthenings of all the query plans for that
        #  query at this particular depth. Then take the union of all of these.
        #  In other words, this strengthening constraint is a graph containing the nodes, edges, tags and tagged edges
        #  that must be satisfied by the graph containing the inputs, that is `inp_graph` in this context.
        #  This constraint can then be used by the `enumerate` procedure to speed up the search.
        strengthening_constraint: Graph = context.waypoints[_depth].get_strengthening_constraint(inp_graph)
        enumeration_item: EnumerationItem
        for enumeration_item in domain.enumerate(component_name=component_name,
                                                 inputs=inputs,
                                                 g_inputs=g_inputs,
                                                 constants=problem.constants,
                                                 strengthening_constraint=strengthening_constraint):

            output = enumeration_item.output
            c_graph = enumeration_item.graph
            o_graph = enumeration_item.o_graph

            # for g in g_inputs:
            #     assert set(g.iter_nodes()).issubset(set(c_graph.iter_nodes()))

            if problem.timeout is not None and time.time() - self._time_start > problem.timeout:
                raise TimeoutError("Exceeded time limit.")

            out_entity = next(iter(o_graph.iter_entities()))
            c_graph.add_node(PlaceholderNode(entity=out_entity))
            c_graph = Transformation.build_from_graph(c_graph,
                                                      input_entities=inp_entities,
                                                      output_entity=out_entity)

            #  Check if the returned graph is consistent with the query plans.
            if not context.check_validity(c_graph, depth=_depth):
                continue

            #  Prepare for the next round.
            context.step(output=output, graph=c_graph, output_graph=o_graph,
                         enumeration_item=enumeration_item,
                         depth=_depth)

            if _depth == skeleton.length - 1:
                #  This was the last component, prepare the program and return it along with the final output and graph.
                yield output, o_graph

            else:
                #  Move on to the next component.
                yield from self._solve_for_skeleton_recursive(problem, skeleton, query_plans, context,
                                                              _depth=_depth + 1)
