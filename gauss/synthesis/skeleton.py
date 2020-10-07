from typing import Tuple, List, Iterator

import attr


@attr.s(cmp=False, repr=False)
class Skeleton:
    """
    Essentially a program without arguments, or a high-level sketch.
    Arguments are integers with the following convention -
        1) Negative integers denote inputs to the whole program,
           -1 being the first input, -2 being the second and so on.
        2) Positive integers denote partial outputs,
            1 being the result of the first function, 2 being the result of the second and so on.
    """

    program: List[Tuple[str, List[int]]] = attr.ib()
    length: int = attr.ib(init=False, default=0)

    def __attrs_post_init__(self):
        self.length = len(self.program)

    def __iter__(self) -> Iterator[Tuple[str, List[int]]]:
        yield from self.program

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        return self.program[item]

    def __hash__(self):
        return hash(str(self.program))

    def __eq__(self, other):
        if not isinstance(other, Skeleton):
            return False

        return self.program == other.program

    def __len__(self):
        return self.length

    def __str__(self):
        return str(self.program)

    def __repr__(self):
        return repr(self.program)

    @property
    def components(self):
        return [i[0] for i in self.program]
