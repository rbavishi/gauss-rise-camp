from typing import Any

import attr


@attr.s(cmp=False, repr=False)
class Entity:
    value: Any = attr.ib()


@attr.s(cmp=False, repr=False)
class Node:
    label: int = attr.ib(converter=int)
    entity: Entity = attr.ib()
    value: Any = attr.ib()


class SymbolicValue:
    pass


#  Using a class to maintain consistency with pickling.
SYMBOLIC_VALUE = SymbolicValue
DEFAULT_ENTITY = Entity(value=SYMBOLIC_VALUE)


@attr.s(cmp=False, repr=False)
class Edge:
    src: Node = attr.ib()
    dst: Node = attr.ib()
    label: int = attr.ib(converter=int)

    def __hash__(self):
        return hash((self.src, self.dst, self.label))

    def __eq__(self, other):
        if isinstance(other, Edge):
            #  Two edges are equal if they are the same type,
            #  and have the same src and dst nodes and share the edge label.
            if type(self) != type(other):
                return False

            return self.src is other.src and self.dst is other.dst and self.label == other.label

        elif isinstance(other, tuple) and len(other) == 3:
            return (self.src, self.dst, self.label) == other

        return False


@attr.s(cmp=False, repr=False)
class TaggedEdge:
    src: Node = attr.ib()
    dst: Node = attr.ib()
    tag: str = attr.ib()

    def __hash__(self):
        return hash((self.src, self.dst, self.tag))

    def __eq__(self, other):
        if isinstance(other, TaggedEdge):
            #  Two tagged edges are equal if they are the same type,
            #  and have the same src and dst nodes and share the tag.
            if type(self) != type(other):
                return False

            return self.src is other.src and self.dst is other.dst and self.tag == other.tag

        elif isinstance(other, tuple) and len(other) == 3:
            return (self.src, self.dst, self.tag) == other

        return False
