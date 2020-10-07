from abc import ABC, abstractmethod
from typing import List, Set, Optional, Dict, Any, Tuple, Iterator

import attr

from gauss.graphs import Graph, Node
from gauss.synthesis.witness_set import WitnessEntry


@attr.s
class EnumerationItem:
    output: Any = attr.ib()
    graph: Graph = attr.ib()
    o_graph: Graph = attr.ib()


@attr.s
class Solution:
    code: str = attr.ib()
    output: Any = attr.ib()


@attr.s
class SynthesisDomain(ABC):
    @abstractmethod
    def get_available_components(self) -> List[str]:
        """
        Returns:
            List[str]: The names of components available in this domain as a list of strings.
        """

    @abstractmethod
    def generate_witness_entry(self, component_name: str, seed: int) -> WitnessEntry:
        """
        Generates an entry for the witness set for the given component (referenced by its name)
        Args:
            component_name (str): The name of the component to generate the entry for.
            seed (int): A seed that can be used to direct the generation (if a random process is involved).

        Returns:
            WitnessEntry: The generated entry.
        """

    @abstractmethod
    def enumerate(self,
                  component_name: str,
                  inputs: List[Any],
                  g_inputs: List[Graph],
                  **kwargs) -> Iterator[EnumerationItem]:
        """
        Enumerate all possible programs (with output and graph abstraction) for the given component and
        inputs (along with the graph abstractions of the inputs).

        Args:
            component_name:
            inputs:
            g_inputs:
            **kwargs:

        Returns:

        """

    @abstractmethod
    def prepare_solution(self,
                         inputs: List[Any],
                         output: Any,
                         graph: Graph,
                         graph_inputs: List[Graph],
                         graph_output: Graph,
                         enumeration_items: List[EnumerationItem],
                         arguments: List[List[int]],
                         int_to_names: Dict[int, str],
                         int_to_obj: Dict[int, Any]) -> Solution:
        pass

    @abstractmethod
    def check_equivalent(self, target: Any, actual: Any, **kwargs):
        """
        Check if the actual output is equivalent to the target output.
        Args:
            target:
            actual:

        Returns:

        """

    @abstractmethod
    def check_partial(self, target: Any, actual: Any, **kwargs):
        """
        Check if the actual output *subsumes* the target output.
        Args:
            target:
            actual:

        Returns:

        """

    def get_equality_edge_label(self) -> Optional[int]:
        return None

    @abstractmethod
    def perform_transitive_closure(self, graph: Graph, join_nodes: Optional[Set[Node]] = None) -> None:
        """
        Perform a transitive closure on the graph w.r.t the join nodes, if any.

        Args:
            graph: The graph to perform the transitive closure on.
            join_nodes: The set of nodes to restrict joins to, if any.

        Returns:
            (None) : The closure is performed in-place.
        """

    def get_node_label_map(self) -> Dict[int, str]:
        """
        Return a mapping from integers to node labels. Helpful for debugging if implemented.
        Returns:

        """
        return {}

    def get_edge_label_map(self) -> Dict[int, str]:
        """
        Return a mapping from integers to node labels. Helpful for debugging if implemented.
        Returns:

        """
        return {}

    def rank_sequences(self, sequences: List[List[str]]) -> List[List[str]]:
        """
        Decide the order of exploration. By default, no ranking (passthrough).

        Args:
            sequences:

        Returns:

        """
        return sequences


@attr.s
class SynthesisUI(ABC):
    @abstractmethod
    def get_available_operations(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def perform_operation(self,
                          operation: str,
                          inputs: List[Tuple[Dict, Any]],
                          obj_store: Dict[str, Any],
                          trace_store: Dict[str, Any],
                          kwargs: Dict[str, Any]):
        pass

    @abstractmethod
    def process_ui_interaction(self,
                               inputs: Dict[str, Any],
                               interactions: List[Dict]) -> Tuple[Any, Graph, Dict[str, Graph]]:
        pass
