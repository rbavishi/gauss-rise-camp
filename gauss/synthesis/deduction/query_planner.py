"""
Contains the logic for precomputation of query plans, amongst other things
"""
import collections
import functools
import itertools

import attr
from typing import List, Set, Dict, Optional, Tuple, FrozenSet, Iterator

from gauss.graphs import Graph, Entity, SYMBOLIC_VALUE, Node, Edge
from gauss.graphs.common.graphmapping import GraphMapping
from gauss.graphs.utils import create_symbolic_copy
from gauss.synthesis.config import EngineConfig
from gauss.synthesis.domains import SynthesisDomain
from gauss.synthesis.graphs import PlaceholderNode, Transformation, PLACEHOLDER_LABEL
from gauss.synthesis.problem import SynthesisProblem
from gauss.synthesis.queries import extract_paths, Path, Query, ArgumentNumberMapping
from gauss.synthesis.query_plans import QueryPlan, QueryPlans
from gauss.synthesis.skeleton import Skeleton
from gauss.synthesis.witness_set import WitnessSet, WitnessEntry
from gauss.utilities import logutils
from gauss.utilities.debug import debug_iter
from gauss.utilities.logutils import logger


@attr.s(cmp=False, repr=False)
class SequenceTrie:
    """
    A special data-structure to compactly store sequences of component-names. Supports intersection and union.
    """
    components: Set[str] = attr.ib()
    children: Optional[Dict[str, 'SequenceTrie']] = attr.ib(default=None)

    @classmethod
    def union(cls, t1: 'SequenceTrie', t2: 'SequenceTrie'):
        if t1.children is None:
            assert t2.children is None, "t1 and t2 should be of equal depths"
            return cls(t1.components | t2.components, None)

        result = SequenceTrie(t1.components | t2.components, {})
        only_t1 = t1.components - t2.components
        only_t2 = t2.components - t1.components
        common = t1.components & t2.components

        for c in only_t1:
            result.children[c] = t1.children[c]
        for c in only_t2:
            result.children[c] = t2.children[c]
        for c in common:
            result.children[c] = cls.union(t1.children[c], t2.children[c])

        return result

    @classmethod
    def intersect(cls, tries: List['SequenceTrie']) -> Optional['SequenceTrie']:
        common_components = set.intersection(*(trie.components for trie in tries))

        if len(common_components) == 0:
            return None

        if tries[0].children is None:
            assert all(t.children is None for t in tries), "All tries must be of the same depth"
            return SequenceTrie(components=common_components)

        new_children = {}
        retained_components: Set[str] = set()
        for component in common_components:
            child_tries = [trie.children[component] for trie in tries]
            intersected = cls.intersect(child_tries)
            if intersected is not None:
                new_children[component] = intersected
                retained_components.add(component)

        return SequenceTrie(retained_components, new_children)

    def get_sequences(self, _d: int = 0) -> List[List[str]]:
        if self.children is None:
            sequences = [[component] for component in self.components]

        else:
            sequences = []
            for component in self.components:
                child_sequences = self.children[component].get_sequences(_d=_d + 1)
                for seq in child_sequences:
                    seq.append(component)

                sequences.extend(child_sequences)

        if _d > 0:
            return sequences

        #  Reverse the lists and then return
        return [list(reversed(seq)) for seq in sequences]


@attr.s(cmp=False, repr=False)
class UnitMetaPlan:
    @attr.s(cmp=False, repr=False)
    class ComponentEntry:
        name: str = attr.ib()
        argument_mappings: Set[ArgumentNumberMapping] = attr.ib(factory=set)

    transformation: Transformation = attr.ib()
    component_entries: Dict[str, ComponentEntry] = attr.ib(factory=dict)
    strengthenings: Dict[str, Tuple[Transformation, GraphMapping]] = attr.ib(factory=dict)
    empty: bool = attr.ib(default=False)

    def deepcopy(self):
        copy, mapping = self.transformation.deepcopy()
        strengthenings = {k: (t, m.apply_mapping(mapping, only_keys=True)) for k, (t, m) in self.strengthenings.items()}
        return UnitMetaPlan(transformation=copy,
                            component_entries=self.component_entries.copy(),
                            strengthenings=strengthenings,
                            empty=self.empty), mapping


@attr.s(cmp=False, repr=False)
class MetaQueryPlan:
    @attr.s(cmp=False, repr=False)
    class BlueprintItem:
        depth: int = attr.ib()
        unit: UnitMetaPlan = attr.ib()
        canonical_mapping: GraphMapping = attr.ib()  # Mapping w.r.t the canonical transformation at that depth.
        sub_plan: Optional['MetaQueryPlan'] = attr.ib()  # None if terminal unit.
        # Mappings from inputs and output of the unit to the inputs of the sub-plan
        border_mapping: Optional[GraphMapping] = attr.ib()

    transformation: Transformation = attr.ib()
    blueprint: Dict[int, Dict[str, List[BlueprintItem]]] = attr.ib()

    #  A flattened list of blueprint items.
    blueprint_items: Dict[int, Set[BlueprintItem]] = attr.ib(init=False, factory=dict)
    #  A canonical transformation for every depth.
    canonical_transformations: Dict[int, Transformation] = attr.ib(init=False, factory=dict)

    @classmethod
    def initialize_from_unit_plan(cls, unit: UnitMetaPlan) -> 'MetaQueryPlan':
        copy, mapping = unit.transformation.deepcopy()
        blueprint: Dict[int, Dict[str, List[MetaQueryPlan.BlueprintItem]]]
        blueprint = collections.defaultdict(lambda: collections.defaultdict(list))
        for c in unit.component_entries:
            blueprint[1][c].append(MetaQueryPlan.BlueprintItem(depth=1,
                                                               unit=unit,
                                                               canonical_mapping=mapping.reverse(),
                                                               sub_plan=None,
                                                               border_mapping=None))

        result = MetaQueryPlan(transformation=unit.transformation.deepcopy()[0], blueprint=blueprint)
        result.canonical_transformations[1] = copy
        return result


@attr.s
class QueryPlanner:
    _domain: SynthesisDomain = attr.ib()
    _config: EngineConfig = attr.ib()
    _unit_plans: Dict[Transformation, UnitMetaPlan] = attr.ib(init=False, factory=dict)
    _meta_plans: Dict[Transformation, MetaQueryPlan] = attr.ib(init=False, factory=dict)
    _sequence_tries: Dict[int, Dict[MetaQueryPlan, Optional[SequenceTrie]]] = attr.ib(init=False, default=None)

    def get_candidate_sequences(self,
                                problem: SynthesisProblem,
                                queries: Dict[Transformation, List[Query]],
                                length: int = 1) -> List[List[str]]:

        #  Simply take the intersection of the sequence tries for all the requested transformations.
        meta_plans = [self._meta_plans[t] for t in queries]
        if any(self._sequence_tries[length][p] is None for p in meta_plans):
            return []

        tries = [self._sequence_tries[length][p] for p in meta_plans]

        intersected_trie = SequenceTrie.intersect(tries)
        if intersected_trie is None:
            return []

        return intersected_trie.get_sequences()

    def get_query_plans(self,
                        sequence: List[str],
                        problem: SynthesisProblem,
                        queries: Dict[Transformation, List[Query]]) -> Iterator[Tuple[Skeleton, QueryPlans]]:

        skeletons_to_canonical_query_plans: Dict[Skeleton, Dict[Query, Set[QueryPlan]]]
        skeletons_to_canonical_query_plans = collections.defaultdict(lambda: collections.defaultdict(set))

        #  Find the canonical query plans for all transformations. Simultaneously compute the skeletons
        #  that can achieve all the transformations.
        viable_skeletons: Optional[Set[Skeleton]] = None
        for t, query_list in queries.items():
            s_to_qp = self._get_canonical_query_plans(sequence, t)
            viable_skeletons_for_query_list: Optional[Set[Skeleton]] = None
            for query in query_list:
                #  We need to find a skeleton that works for every individual query.
                viable_skeletons_for_query: Set[Skeleton] = set()
                for skeleton, query_plans in s_to_qp.items():
                    #  Need to adjust the skeleton to reflect the argument number mapping of the query.
                    processed_skeleton: Skeleton = query.arg_number_mapping.apply_skeleton(skeleton)
                    viable_skeletons_for_query.add(processed_skeleton)
                    skeletons_to_canonical_query_plans[processed_skeleton][query].update(query_plans)

                if viable_skeletons_for_query_list is None:
                    viable_skeletons_for_query_list = viable_skeletons_for_query
                else:
                    viable_skeletons_for_query_list.intersection_update(viable_skeletons_for_query)

            if viable_skeletons is None:
                viable_skeletons = viable_skeletons_for_query_list
            else:
                viable_skeletons.intersection_update(viable_skeletons_for_query_list)

            #  No skeleton found which can satisfy all the queries / transformations. End the iterator
            if len(viable_skeletons) == 0:
                return

        #  Amongst the viable skeletons, for a set of skeletons which have the same canonical query plan for each query,
        #  only keep one of them to save time as failure of one guarantees failure of others as well.
        key_to_skeleton_map: Dict[FrozenSet[Tuple[Query, FrozenSet[QueryPlan]]], Skeleton] = {}
        for skeleton in viable_skeletons:
            key = frozenset((query, frozenset(plans))
                            for query, plans in skeletons_to_canonical_query_plans[skeleton].items())
            if key in key_to_skeleton_map:
                continue

            key_to_skeleton_map[key] = skeleton

        chosen_skeletons: Set[Skeleton] = set(key_to_skeleton_map.values())

        #  Yield the chosen skeletons one by one along with adapted query plans for each individual query.
        for skeleton in chosen_skeletons:
            plans = QueryPlans()
            query_plan_dict = skeletons_to_canonical_query_plans[skeleton]

            for query, plan_list in query_plan_dict.items():
                #  Make a distinct copy of all query plans for all queries and perform constant propagation.
                adapted_plans = [self._adapt_query_plan(plan, query) for plan in plan_list]
                plans.record_plans(query, adapted_plans)

            yield skeleton, plans

    @classmethod
    def build(cls, domain: SynthesisDomain, config: EngineConfig, witness_set: WitnessSet) -> 'QueryPlanner':
        q_planner = QueryPlanner(domain=domain, config=config)
        logger_color = logger.opt(colors=True)

        #  ----------------------------------------------------------------------------------------------------------  #
        #  Stage 1 : Build unit plans
        #  ----------------------------------------------------------------------------------------------------------  #
        logger.debug("Extracting meta query-plans of length 1 from individual components...")
        for component_name in domain.get_available_components():
            for witness_entry in witness_set.entries[component_name]:
                q_planner._extract_unit_plans(component_name, witness_entry)

        #  Log some information for post-mortem analysis if necessary
        #  Total plans found.
        logger_color.debug(f"Found <green>{len(q_planner._unit_plans)}</green> unit plans in total.")

        #  Plans found per component.
        components_to_units: Dict[str, List[UnitMetaPlan]] = collections.defaultdict(list)
        for unit in q_planner._unit_plans.values():
            for c in unit.component_entries:
                components_to_units[c].append(unit)

        with logutils.temporary_add(f"{config.path}/logs/query_planner/unit_plans.log",
                                    level="TRACE",
                                    only_sink=True) as logger_:
            logger_ = logger_.opt(raw=True)
            for component_name, units in components_to_units.items():
                logger_color.debug(f"Found <green>{len(units)}</green> unit plans from "
                                  f"<blue>{component_name}</blue>.")
                logger_.opt(colors=True).debug(f"Found <green>{len(units)}</green> unit plans from "
                                              f"<blue>{component_name}</blue>.")
                for unit in units:
                    logger_.trace(f"Component: {component_name}\n")
                    logger_.trace("-----------------\n")
                    logger_.trace("Transformation\n")
                    logger_.trace("-----------------\n")
                    logger_.trace(unit.transformation.to_str(domain))
                    logger_.trace("\n-----------------\n")
                    logger_.trace("Argument Mappings\n")
                    logger_.trace("-----------------\n")
                    logger_.trace(unit.component_entries[component_name].argument_mappings)
                    logger_.trace("\n========xxx========\n\n")

        #  ----------------------------------------------------------------------------------------------------------  #
        #  Stage 2 : Strengthen Unit Plans
        #  ----------------------------------------------------------------------------------------------------------  #
        logger_color.debug("Strengthening Unit Plans...")
        for unit_plan in debug_iter(list(q_planner._unit_plans.values()), desc='Strengthening Unit Plans'):
            query: Transformation = unit_plan.transformation
            for component_name in unit_plan.component_entries.keys():
                #  The witness set examples are guaranteed to have a placeholder node for each entity,
                #  so the use of placeholder nodes in transformations should not cause issues.
                if unit_plan.empty:
                    strengthened, mapping = query.deepcopy()
                    strengthened = Graph.from_graph(strengthened)

                else:
                    examples: List[Transformation] = witness_set.get_transformations(component_name)
                    strengthened, mapping = query.get_greatest_common_universal_supergraph(examples)

                inp_entities = set(query.get_input_entities())
                m_reverse = mapping.reverse()
                keep_nodes = [n for n in strengthened.iter_nodes()
                              if m_reverse.m_ent.get(n.entity, None) in inp_entities]

                strengthened = strengthened.induced_subgraph(keep_nodes=keep_nodes)

                unit_plan.strengthenings[component_name] = (strengthened, mapping)

        logger_color.debug(f"Strengthened <green>{len(q_planner._unit_plans)}</green> unit plans.")

        #  ----------------------------------------------------------------------------------------------------------  #
        #  Stage 3 : Combining unit plans to obtain query plans upto max-depth
        #  Note that these are not explicitly constructed. Rather, a recursive formulation is established
        #  to construct the query plans quickly during test-time.
        #  ----------------------------------------------------------------------------------------------------------  #

        #  Initialize the meta-plans from these unit plans
        logger.debug("Initializing meta-plans from unit-plans...")
        for transformation, unit_plan in q_planner._unit_plans.items():
            q_planner._meta_plans[transformation] = MetaQueryPlan.initialize_from_unit_plan(unit_plan.deepcopy()[0])

        logger_color.debug(f"Evolving meta query-plans upto a maximum length of "
                          f"<blue>{config.max_length}</blue>")

        #  Setup a mapping from the label of the output node of transformations
        #  to the meta plan for that transformation. This helps in quickly finding
        #  appropriate unit plans to extend a plan with.
        nlabel_to_unit_plan: Dict[int, Set[UnitMetaPlan]] = collections.defaultdict(set)
        for transformation, unit_plan in q_planner._unit_plans.items():
            #  Guaranteed to be a single node with that entity.
            output_node = next(transformation.get_output_nodes())
            nlabel_to_unit_plan[output_node.label].add(unit_plan)

        #  At every depth, the worklist will be the set of query plans with an entry for depth-1 in the blueprint.
        worklist: Set[MetaQueryPlan] = set(q_planner._meta_plans.values())
        for depth in range(2, config.max_length + 1):
            for unit in worklist:
                q_planner._evolve_meta_plan(unit, depth, nlabel_to_unit_plan)

            worklist = {plan for plan in q_planner._meta_plans.values() if depth in plan.blueprint}
            logger_color.debug(f"Found <green>{len(worklist)}</green> transformations in total "
                              f"at depth <blue>{depth}</blue>.")
            if len(worklist) == 0:
                break

        logger_color.debug(f"Found <green>{len(q_planner._meta_plans)}</green> transformations and meta "
                          f"query plans in total with <blue>max_depth={config.max_length}</blue>.")

        #  TODO : Log the transformations to a file.

        #  ----------------------------------------------------------------------------------------------------------  #
        #  Stage 4 : Setup auxiliary data-structures
        #  ----------------------------------------------------------------------------------------------------------  #

        #  Flattened blue-print items enable access to all the items agnostic of the component name.
        for plan in q_planner._meta_plans.values():
            plan.blueprint_items = collections.defaultdict(set)
            for depth, bp_dict in plan.blueprint.items():
                for items_list in bp_dict.values():
                    plan.blueprint_items[depth].update(items_list)

        #  Sequence tries help in quickly computing candidate sequences given a synthesis problem.
        logger_color.debug("Constructing Sequence Tries...")
        q_planner._compute_sequence_tries()
        logger_color.debug(f"Sequence Tries constructed for every depth.")

        return q_planner

    def _extract_unit_plans(self, component_name: str, witness_entry: WitnessEntry):
        graph = witness_entry.graph
        input_entities = witness_entry.get_input_entities()
        output_entity = witness_entry.get_output_entity()

        placeholder_dict = {}
        #  A placeholder node can represent any node belonging to an entity.
        #  This helps coalesce equivalent query plans.
        for ent in itertools.chain(input_entities, [output_entity]):
            placeholder_dict[ent] = PlaceholderNode(entity=ent)

        path_dict: Dict[Entity, List[Path]] = extract_paths(graph, input_entities, output_entity)

        #  Find queries by taking exactly one path, and placeholder nodes for the
        #  input entities not present in the path.
        for path_ent, paths in path_dict.items():
            remaining_entities = [ent for ent in input_entities if ent is not path_ent]
            for path in paths:
                path_nodes, path_edges = path
                nodes = list(path_nodes) + [placeholder_dict[ent] for ent in remaining_entities]
                edges = path_edges

                #  Get the corresponding subgraph.
                subgraph = Graph.from_nodes_and_edges(nodes=set(nodes), edges=set(edges))
                self._record_unit_meta_query_plan(component_name, subgraph, input_entities, output_entity)

        #  Include empty transformations to help with evolution (the second stage).
        #  An empty transformation plays the role of a *wildcard* plan, that is, any transformation is valid.
        #  We add all empty transformations with input nodes spanning all distinct input node types
        #  (including placeholders) and the output being the placeholder node.
        label_canonical_node: Dict[Entity, Dict[int, Node]] = collections.defaultdict(dict)
        for ent in input_entities:
            for node in graph.iter_nodes(entity=ent):
                label_canonical_node[ent][node.label] = node

        canonical_nodes: List[List[Node]] = [list(v.values()) for v in label_canonical_node.values()]
        canonical_nodes.append([placeholder_dict[output_entity]])
        for subgraph_nodes in itertools.product(*canonical_nodes):
            subgraph = Graph.from_nodes_and_edges(nodes=subgraph_nodes, edges=[])
            self._record_unit_meta_query_plan(component_name, subgraph, input_entities, output_entity,
                                              empty=True)

    def _record_unit_meta_query_plan(self,
                                     component_name: str,
                                     subgraph: Graph,
                                     input_entities: List[Entity],
                                     output_entity: Entity,
                                     empty: bool = False):

        #  First extract the symbolic transformation
        symbolic_copy, mapping = create_symbolic_copy(subgraph)
        mapped_input_entities = [mapping.m_ent[i] for i in input_entities]
        mapped_output_entity = mapping.m_ent[output_entity]
        transformation = Transformation.build_from_graph(symbolic_copy,
                                                         input_entities=mapped_input_entities,
                                                         output_entity=mapped_output_entity)

        if transformation not in self._unit_plans:
            #  Transformation never seen before. Create a fresh unit plan.
            arg_number_mapping = ArgumentNumberMapping({i: i for i in range(len(transformation.get_input_entities()))})
            unit_plan = UnitMetaPlan(transformation=transformation, empty=empty)
            unit_plan.component_entries[component_name] = UnitMetaPlan.ComponentEntry(component_name,
                                                                                      {arg_number_mapping})
            self._unit_plans[transformation] = unit_plan

        else:
            #  Update the existing unit plan with the current component name and argument mapping(s).
            unit_plan: UnitMetaPlan = self._unit_plans[transformation]
            canonical_transform = unit_plan.transformation
            idx_input_entities = {ent: idx for idx, ent in enumerate(transformation.get_input_entities())}
            canonical_inp_entities = canonical_transform.get_input_entities()
            if component_name not in unit_plan.component_entries:
                entry = unit_plan.component_entries[component_name] = UnitMetaPlan.ComponentEntry(component_name)
            else:
                entry = unit_plan.component_entries[component_name]

            for m in canonical_transform.get_subgraph_mappings(transformation):
                arg_num_mapping = ArgumentNumberMapping({i: idx_input_entities[m.m_ent[ent]]
                                                         for i, ent in enumerate(canonical_inp_entities)})
                entry.argument_mappings.add(arg_num_mapping)

    def _evolve_meta_plan(self,
                          plan: MetaQueryPlan,
                          depth: int,
                          nlabel_to_unit_plan: Dict[int, Set[UnitMetaPlan]]):

        #  Replace an input node of plan with the output of a unit meta-plan (contained in nlabel_to_meta_plan).
        #  Thus we extend an existing plan with the output of exactly one component, thus increasing the program
        #  depth by exactly one.

        plan_transformation: Transformation = plan.canonical_transformations[depth - 1]
        all_input_nodes: Set[Node] = set(plan_transformation.get_input_nodes())
        for inp_node in all_input_nodes:
            #  Even if it is a placeholder node, it can only be extended via the empty transform. This makes sense
            #  as no matter what the extension is, it will never "influence" the final output, as there is no path
            #  between a placeholder node and the output node. This helps reduce the size of the collection of
            #  meta query-plans by a large margin.

            remaining_inputs = all_input_nodes - {inp_node}
            for extender in nlabel_to_unit_plan[int(inp_node.label)]:
                #  We can map the other inputs to the inputs of the extender plan. They can also be distinct
                #  inputs of their own. The total number of inputs should, however, be less than max_inputs.
                #  For the remaining inputs, if they are a placeholder node, they can be mapped to *any* of the
                #  input nodes of the extender plan, regardless of the label.
                nlabel_to_inp_node: Dict[int, Set[Node]] = collections.defaultdict(set)
                extender_inputs: Set[Node] = set(extender.transformation.get_input_nodes())
                #  Guaranteed to be a single output node by construction.
                extender_output: Node = next(extender.transformation.get_output_nodes())

                for inp in extender_inputs:
                    nlabel_to_inp_node[inp.label].add(inp)

                nlabel_to_inp_node[extender_output.label].add(extender_output)

                mapping_possibilities: Dict[Node, Set[Node]] = {inp_node: {extender_output}}
                for inp in remaining_inputs:
                    if inp.label == PLACEHOLDER_LABEL:
                        mapping_possibilities[inp] = extender_inputs | {extender_output, inp}
                    else:
                        mapping_possibilities[inp] = nlabel_to_inp_node[inp.label] | {inp}

                node_list = list(all_input_nodes)
                for border_node_mapping in itertools.product(*[mapping_possibilities[n] for n in node_list]):
                    border_node_mapping: Dict[Node, Node] = dict(zip(node_list, border_node_mapping))

                    border_mapping = GraphMapping(m_ent={k.entity: v.entity for k, v in border_node_mapping.items()},
                                                  m_node=border_node_mapping.copy())

                    #  Create a deepcopy of the extender for safety
                    copied_extender, copy_mapping = extender.deepcopy()
                    copied_extender_inputs: Set[Node] = set(copied_extender.transformation.get_input_nodes())
                    copied_extender_output: Node = next(copied_extender.transformation.get_output_nodes())
                    border_mapping = border_mapping.apply_mapping(copy_mapping, only_values=True)
                    assert border_mapping.m_node != border_node_mapping

                    #  The new inputs are the inputs of extender, plus the nodes of the current plan
                    #  which were not bound to any of the inputs of extender.
                    #  We also decide the order of the nodes/entities right now.
                    new_input_nodes: List[Node] = []
                    for inp in plan_transformation.get_input_nodes():
                        mapped = border_mapping.m_node[inp]
                        if mapped is inp:
                            new_input_nodes.append(inp)
                        elif mapped is copied_extender_output:
                            new_input_nodes.extend(i for i in copied_extender.transformation.get_input_nodes()
                                                   if i not in new_input_nodes)
                        elif mapped not in new_input_nodes:
                            assert mapped in copied_extender_inputs
                            new_input_nodes.append(mapped)

                    new_input_entities = [n.entity for n in new_input_nodes]
                    #  Every entity is associated with one node so the following should hold true.
                    assert len(new_input_entities) == len(set(new_input_entities))

                    if len(new_input_entities) > self._config.max_inputs:
                        continue

                    new_output_entity = plan_transformation.get_output_entity()

                    #  Obtain the transformation by establishing common edges between the node pairs in
                    #  border_node_mapping, taking the transitive closure w.r.t equality, and finally the
                    #  induced subgraph by removing the input nodes of the current plan.
                    joint_graph = Graph.from_graph(copied_extender.transformation)
                    joint_graph.merge(plan_transformation)

                    final_border_mapping = GraphMapping()
                    for k, v in border_mapping.m_node.items():
                        if k is not v:
                            final_border_mapping.m_node[k] = v
                            final_border_mapping.m_ent[k.entity] = v.entity
                            for edge in plan_transformation.iter_edges(src=k):
                                joint_graph.add_edge(Edge(v, edge.dst, edge.label))
                            for edge in plan_transformation.iter_edges(dst=k):
                                joint_graph.add_edge(Edge(edge.src, v, edge.label))

                    join_nodes: Set[Node] = set(all_input_nodes)
                    join_nodes.difference_update(new_input_nodes)
                    join_nodes.add(copied_extender_output)
                    self._domain.perform_transitive_closure(joint_graph, join_nodes=join_nodes)

                    keep_nodes = set(joint_graph.iter_nodes())
                    keep_nodes.difference_update(join_nodes)
                    new_transformation_subgraph = joint_graph.induced_subgraph(keep_nodes=keep_nodes)
                    new_transformation = Transformation.build_from_graph(new_transformation_subgraph,
                                                                         input_entities=new_input_entities,
                                                                         output_entity=new_output_entity)

                    #  Record the transformation and how it was obtained.
                    if new_transformation not in self._meta_plans:
                        #  The transformation was never seen before.
                        blueprint = collections.defaultdict(lambda: collections.defaultdict(list))
                        meta_plan = MetaQueryPlan(transformation=new_transformation.deepcopy()[0],
                                                  blueprint=blueprint)
                        self._meta_plans[new_transformation] = meta_plan
                    else:
                        meta_plan = self._meta_plans[new_transformation]

                    if depth not in meta_plan.canonical_transformations:
                        copy, mapping = new_transformation.deepcopy()
                        meta_plan.canonical_transformations[depth] = copy
                        # mapping = mapping.slice(nodes=set(copied_extender.transformation.iter_nodes()))
                        mapping = mapping.reverse()

                    else:
                        canonical = meta_plan.canonical_transformations[depth]
                        mapping = next(new_transformation.get_subgraph_mappings(canonical))
                        # mapping = mapping.slice(nodes=set(copied_extender.transformation.iter_nodes()))
                        mapping = mapping.reverse()

                    bp_item = MetaQueryPlan.BlueprintItem(depth=depth,
                                                          unit=copied_extender,
                                                          canonical_mapping=mapping,
                                                          sub_plan=plan,
                                                          border_mapping=final_border_mapping)

                    for c in copied_extender.component_entries:
                        meta_plan.blueprint[depth][c].append(bp_item)

    def _compute_sequence_tries(self):
        #  First compute tries for depth 1, then depth 2 and so on, using the tries from the previous depths.
        self._sequence_tries = {}
        for depth in range(1, self._config.max_length + 1):
            self._sequence_tries[depth] = {}
            for meta_plan in self._meta_plans.values():
                if len(meta_plan.blueprint[depth]) == 0:
                    self._sequence_tries[depth][meta_plan] = None

                elif depth == 1:
                    components = set.union(*(set(bp_item.unit.component_entries.keys())
                                             for bp_item in meta_plan.blueprint_items[1]))
                    self._sequence_tries[1][meta_plan] = SequenceTrie(components=components)

                else:
                    components = set.union(*(set(bp_item.unit.component_entries.keys())
                                             for bp_item in meta_plan.blueprint_items[depth]))
                    children = {}
                    #  The children are the union of all the tries corresponding to subplans for a particular component.
                    for c in components:
                        child_tries: Set[SequenceTrie] = set()
                        for bp_item in meta_plan.blueprint_items[depth]:
                            if c in bp_item.unit.component_entries:
                                child_tries.add(self._sequence_tries[depth - 1][bp_item.sub_plan])

                        children[c] = functools.reduce(SequenceTrie.union, child_tries)

                    self._sequence_tries[depth][meta_plan] = SequenceTrie(components=components,
                                                                          children=children)

    def _get_canonical_query_plans(self,
                                   sequence: List[str],
                                   transformation: Transformation) -> Dict[Skeleton, Set[QueryPlan]]:

        meta_plan = self._meta_plans[transformation]
        blueprint_item_lists = self._get_blueprint_item_lists(sequence,
                                                              meta_plan,
                                                              _d=len(sequence))
        canonical_transformation = meta_plan.canonical_transformations[len(sequence)]
        mapping = next(canonical_transformation.get_subgraph_mappings(transformation))

        skeletons_to_plans: Dict[Skeleton, Set[QueryPlan]] = collections.defaultdict(set)

        for blueprint_item_list in blueprint_item_lists:
            #  Breakdown the overall transformation in terms of the unit plans contained in the blueprint items.
            #  Store the connections between them as a graph mapping.
            connections = GraphMapping()
            connections.update(mapping)
            graph = Graph()
            for item in blueprint_item_list:
                graph.merge(item.unit.transformation)
                connections = connections.apply_mapping(item.canonical_mapping, only_keys=True)

                if item.border_mapping:
                    connections.update(item.border_mapping)
                    connections = connections.apply_mapping(connections, only_values=True)

            #  Assemble the query plan
            query_plan = QueryPlan(transformation,
                                   units=[item.unit.transformation for item in blueprint_item_list],
                                   all_connections=connections,
                                   strengthenings=[item.unit.strengthenings[component_name]
                                                   for component_name, item in zip(sequence, blueprint_item_list)])

            #  Obtain the skeletons for which this query plan would work.
            #  External inputs are negative integers. See gauss.synthesis.skeleton for details.
            ent_to_idx = {ent: -idx for idx, ent in enumerate(transformation.get_input_entities(), 1)}
            possible_arg_ints_lists = []
            for component_name, (idx, item) in zip(sequence, enumerate(blueprint_item_list, 1)):
                #  Get the mapped entities to the inputs of this unit's transformation, and look up their idx values.
                arg_ints = [ent_to_idx[connections.m_ent[ent]] for ent in item.unit.transformation.get_input_entities()]

                #  Get all the permutations as well.
                arg_ints_list = [arg_num_mapping.apply_list(arg_ints)
                                 for arg_num_mapping in item.unit.component_entries[component_name].argument_mappings]

                possible_arg_ints_lists.append(arg_ints_list)
                ent_to_idx[item.unit.transformation.get_output_entity()] = idx

            #  The skeletons are then simply the all the combinations
            for arg_ints_list in itertools.product(*possible_arg_ints_lists):
                skeleton = Skeleton(list(zip(sequence, arg_ints_list)))
                skeletons_to_plans[skeleton].add(query_plan)

        return skeletons_to_plans

    def _get_blueprint_item_lists(self,
                                  sequence: List[str],
                                  plan: MetaQueryPlan,
                                  _d: int) -> List[List[MetaQueryPlan.BlueprintItem]]:
        if _d == 1:
            return [[i] for i in plan.blueprint[1][sequence[-1]]]

        item_lists = []
        for item in plan.blueprint[_d][sequence[-_d]]:
            child_item_lists = self._get_blueprint_item_lists(sequence, item.sub_plan, _d=_d - 1)
            for i in child_item_lists:
                #  Append in reverse to avoid making copies of lists.
                i.append(item)

            item_lists.extend(child_item_lists)

        if _d == len(sequence):
            #  Undo the reverse appends.
            return [list(reversed(i)) for i in item_lists]
        else:
            return item_lists

    def _adapt_query_plan(self, plan: QueryPlan, query: Query):
        #  The given plan is assumed to be a canonical query plan.
        #  Also, the transformation in plan should be the same as the transformation in query. This should be
        #  guaranteed by construction of the query plan.

        #  Create a fresh copy of the plan where the transformation is the actual subgraph.
        adapted_plan = plan.deepcopy().adapt(new_transformation=query.subgraph,
                                             mapping_old_to_new=query.mapping)

        equality_label = self._domain.get_equality_edge_label()

        if equality_label is None:
            return None

        #  Propagate known values amongst the nodes with the equality edge
        influence: Dict[Node, Set[Node]] = collections.defaultdict(set)
        for k, v in adapted_plan.all_connections.m_node.items():
            influence[v].add(k)

        seen = set()
        worklist = collections.deque(adapted_plan.transformation.iter_nodes())
        while len(worklist) > 0:
            node = worklist.popleft()
            if node in seen:
                continue

            seen.add(node)
            if node.value is SYMBOLIC_VALUE:
                continue

            #  Connected nodes inherit the value
            for n in influence[node]:
                if n.value is SYMBOLIC_VALUE:
                    n.value = node.value
                    worklist.append(n)

            #  Equality edges with src and dst as node also propagate the values
            for unit in adapted_plan.units:
                for e in unit.iter_edges(src=node, label=equality_label):
                    if e.dst.value is SYMBOLIC_VALUE:
                        e.dst.value = node.value
                        worklist.append(e.dst)

                for e in unit.iter_edges(dst=node, label=equality_label):
                    if e.src.value is SYMBOLIC_VALUE:
                        e.src.value = node.value
                        worklist.append(e.src)

        #  We may have wrecked the internal data-structures of the unit transformations by changing values directly.
        #  Create shallow copies which force a rebuild
        adapted_plan.units = [Transformation.build_from_graph(Graph.from_nodes_and_edges(unit.get_all_nodes(),
                                                                                         unit.get_all_edges()),
                                                              unit.get_input_entities(),
                                                              unit.get_output_entity())
                              for unit in adapted_plan.units]

        return adapted_plan
