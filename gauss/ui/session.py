from abc import ABC, abstractmethod
from typing import Any, List

import attr

from gauss.synthesis.domains import SynthesisUI, SynthesisDomain
from gauss.synthesis.solver.engine import SynthesisEngine


@attr.s(cmp=False, repr=False)
class UISession(ABC):
    session_id: str = attr.ib()
    domain: SynthesisDomain = attr.ib()
    domain_ui: SynthesisUI = attr.ib()
    engine: SynthesisEngine = attr.ib()
    inputs: List[Any] = attr.ib()

    output_widget = attr.ib()

    def __attrs_post_init__(self):
        pass

    @abstractmethod
    def display(self):
        pass
