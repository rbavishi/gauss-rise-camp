"""
Some useful utilities for logging, performance debugging etc.
"""

from typing import Optional, TypeVar, Iterable

import colorama
import tqdm
from colorama import Fore as CF, Style as CS

from gauss.utilities import logutils

colorama.init()

T = TypeVar("T")


def debug_iter(iterator: Iterable[T],
               *args,
               display_current: Optional[str] = None,
               position: int = None, **kwargs) -> Iterable[T]:
    if logutils.get_stderr_level() not in ["TRACE", "DEBUG"]:
        yield from iterator
        return

    print(CF.MAGENTA, end='\r')
    kwargs.pop("dynamic_ncols", None)
    if display_current is not None:
        with tqdm.tqdm(iterator, *args, **kwargs, dynamic_ncols=True, position=position) as t:
            for result in t:
                t.set_postfix({display_current: result})
                yield result
                t.set_postfix({})

    else:
        yield from tqdm.tqdm(iterator, *args, **kwargs, dynamic_ncols=True, position=position)

    if position is None or position == 0:
        print(CS.RESET_ALL, end='\r')
