from typing import List, Any, Optional

import attr

from gauss.graphs import Graph


@attr.s
class SynthesisProblem:
    inputs: List[Any] = attr.ib()
    output: Any = attr.ib()
    graph: Graph = attr.ib()
    graph_inputs: List[Graph] = attr.ib()

    constants: List[Any] = attr.ib(factory=list)
    check_strict: bool = attr.ib(default=True)

    #  Code formatting directives
    input_names: Optional[List[str]] = attr.ib(default=None)
    output_name: Optional[str] = attr.ib(default=None)

    #  Timeouts
    timeout: Optional[float] = attr.ib(converter=float, default=None)
