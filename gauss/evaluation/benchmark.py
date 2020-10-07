from abc import ABC, abstractmethod
from typing import List, Any

import attr

from gauss.synthesis.problem import SynthesisProblem
from gauss.synthesis.skeleton import Skeleton


@attr.s
class Benchmark(ABC):
    b_id: str = attr.ib()  # ID of the benchmark
    inputs: List[Any] = attr.ib()  # Original inputs.
    intermediates: List[Any] = attr.ib()  # Intermediate outputs.
    output: Any = attr.ib()  # Target output (complete).
    constants: List[Any] = attr.ib()  # Any user-supplied constants.
    program: str = attr.ib()  # The ground-truth program
    skeleton: Skeleton = attr.ib()  # Skeleton of the ground-truth program

    @abstractmethod
    def construct_synthesis_problem(self) -> SynthesisProblem:
        """
        Return the corresponding SynthesisProblem by constructing the intent graph, amongst other things.
        Returns:
            SynthesisProblem: The problem corresponding to the benchmark.
        """
