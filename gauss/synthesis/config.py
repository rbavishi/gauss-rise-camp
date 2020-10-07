"""
Engine configurations help fine-tune the behavior per synthesis domain.
"""
import os
import shutil

import attr

from gauss.utilities.logutils import logger


@attr.s
class EngineConfig:
    """
    This class exposes the various knobs that can be set per synthesis domain to fine-tune the behavior
    of the synthesis engine.
    """

    path: str = attr.ib(converter=os.path.abspath)
    min_length: int = attr.ib(default=1)
    max_length: int = attr.ib(default=1)
    min_inputs: int = attr.ib(default=1)
    max_inputs: int = attr.ib(default=1)
    num_examples_per_component: int = attr.ib(default=1)

    _overwrite: bool = attr.ib(default=False, repr=False)

    def __attrs_post_init__(self):
        if os.path.exists(self.path):
            if not self._overwrite:
                with logger.catch(reraise=True):
                    raise AssertionError(f"Path to save the engine ({self.path}) already exists. "
                                         f"Pass (overwrite=True) to the config constructor to reset.")
            else:
                shutil.rmtree(self.path)
                os.makedirs(self.path, exist_ok=True)
        else:
            os.makedirs(self.path, exist_ok=True)
