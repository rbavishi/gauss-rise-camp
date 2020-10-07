"""
Routines related to instantiation and usage of deduction engines.
"""
from typing import List, Dict, Tuple, Iterator

import attr

from gauss.synthesis.config import EngineConfig
from gauss.synthesis.deduction.query_planner import QueryPlanner
from gauss.synthesis.domains import SynthesisDomain
from gauss.synthesis.graphs import Transformation
from gauss.synthesis.problem import SynthesisProblem
from gauss.synthesis.queries import Query
from gauss.synthesis.query_plans import QueryPlans
from gauss.synthesis.skeleton import Skeleton
from gauss.synthesis.witness_set import WitnessEntry, WitnessSet
from gauss.utilities.logutils import logger


@attr.s
class DeductionEngine:
    _config: EngineConfig = attr.ib()
    _query_planner: QueryPlanner = attr.ib()

    def get_candidate_sequences(self,
                                domain: SynthesisDomain,
                                problem: SynthesisProblem,
                                queries: Dict[Transformation, List[Query]],
                                length: int) -> Iterator[List[str]]:

        yield from domain.rank_sequences(self._query_planner.get_candidate_sequences(problem, queries, length))

    def get_query_plans(self,
                        sequence: List[str],
                        problem: SynthesisProblem,
                        queries: Dict[Transformation, List[Query]]) -> Iterator[Tuple[Skeleton, QueryPlans]]:

        yield from self._query_planner.get_query_plans(sequence, problem, queries)

    @classmethod
    def build(cls, domain: SynthesisDomain, config: EngineConfig) -> 'DeductionEngine':
        #  First setup the witness set
        component_names = list(domain.get_available_components())
        logger.opt(colors=True).debug(f"Building witness set using components <blue>{component_names}</blue>...")
        entries: Dict[str, List[WitnessEntry]] = {}

        for component_name in component_names:
            entries[component_name] = [domain.generate_witness_entry(component_name, seed=idx)
                                       for idx in range(config.num_examples_per_component)]

        witness_set = WitnessSet(entries=entries)
        logger.info(f"Finished building witness set.")
        for component_name, entries in witness_set.entries.items():
            logger.opt(colors=True).debug(f"Found <green>{len(entries)}</green> examples for "
                                          f"component <blue>{component_name}</blue>")

        #  Now pre-compute query plans for all the possible unit queries using the witness set
        logger.debug(f"Initializing query planner...")
        query_planner = QueryPlanner.build(domain, config, witness_set)
        logger.info(f"Query planner initialized.")

        #  Assemble the deduction engine
        logger.info("Assembling deduction engine...")

        return DeductionEngine(config=config, query_planner=query_planner)
