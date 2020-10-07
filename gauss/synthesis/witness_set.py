"""
The basic witness set implementation
"""
import itertools

import attr
from typing import List, Any, Dict

from gauss.graphs import Graph, Entity
from gauss.synthesis.graphs import Transformation, PlaceholderNode


@attr.s
class WitnessEntry:
    inputs: List[Any] = attr.ib()
    output: Any = attr.ib()
    program: str = attr.ib()
    graph: Graph = attr.ib()
    extra: Dict = attr.ib(factory=dict)

    transformation: Transformation = attr.ib(init=False, default=None)

    _input_entities: List[Entity] = attr.ib(init=False, factory=list)
    _output_entity: Entity = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):
        entities = list(self.graph.iter_entities())
        for inp in self.inputs:
            for ent in entities:
                if inp is ent.value:
                    self._input_entities.append(ent)
                    break

            else:
                raise AssertionError(f"No entity found for input {inp}.")

        try:
            self._output_entity = next(ent for ent in entities if ent.value is self.output)

        except StopIteration:
            raise AssertionError(f"No entity found for output {self.output}.")

        #  Add placeholder nodes for every input and output entity. This facilitates graph algorithms.
        #  TODO : Is there a better way? Should we introduce the concept of wildcard nodes in subgraph isomorphism?
        #  TODO : Seems like that would entail significant engineering, not to mention rethinking the API.
        #  TODO : Keeping this as low priority.
        for ent in itertools.chain(self._input_entities, [self._output_entity]):
            self.graph.add_node(PlaceholderNode(entity=ent))

        self.transformation = Transformation.build_from_graph(self.graph,
                                                              input_entities=self._input_entities,
                                                              output_entity=self._output_entity)

    def get_input_entities(self) -> List[Entity]:
        return self._input_entities

    def get_output_entity(self) -> Entity:
        return self._output_entity


@attr.s
class WitnessSet:
    entries: Dict[str, List[WitnessEntry]] = attr.ib(factory=dict)

    def get_graphs(self, component_name: str) -> List[Graph]:
        return [i.graph for i in self.entries[component_name]]

    def get_transformations(self, component_name: str) -> List[Transformation]:
        return [i.transformation for i in self.entries[component_name]]
