import collections

import attr
from typing import List, Collection, Set, Dict, Tuple, Iterator

from gauss.graphs.common.graphmapping import GraphMapping
from gauss.synthesis.graphs import Transformation
from gauss.synthesis.queries import Query


@attr.s(cmp=False, repr=False)
class QueryPlan:
    transformation: Transformation = attr.ib()
    units: List[Transformation] = attr.ib()
    all_connections: GraphMapping = attr.ib()
    strengthenings: List[Tuple[Transformation, GraphMapping]] = attr.ib()

    def deepcopy(self):
        mapping = GraphMapping()
        new_units = []
        new_strengthenings = []
        for unit, (strengthened_transform, s_mapping) in zip(self.units, self.strengthenings):
            copy, unit_mapping = unit.deepcopy()
            mapping.update(unit_mapping)
            new_units.append(copy)
            new_strengthenings.append((strengthened_transform, s_mapping.apply_mapping(unit_mapping, only_keys=True)))

        return QueryPlan(transformation=self.transformation,
                         units=new_units,
                         all_connections=self.all_connections.apply_mapping(mapping),
                         strengthenings=new_strengthenings)

    def adapt(self, new_transformation: Transformation, mapping_old_to_new: GraphMapping):
        return QueryPlan(transformation=new_transformation,
                         units=self.units[:],
                         all_connections=self.all_connections.apply_mapping(mapping_old_to_new, only_values=True),
                         strengthenings=self.strengthenings[:])


@attr.s(cmp=False, repr=False)
class QueryPlans:
    _plan_dict: Dict[Query, Set[QueryPlan]] = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):
        self._plan_dict = collections.defaultdict(set)

    def record_plans(self, query: Query, plans: Collection[QueryPlan]):
        self._plan_dict[query].update(plans)

    def __iter__(self) -> Iterator[Tuple[Query, Set[QueryPlan]]]:
        yield from self._plan_dict.items()
