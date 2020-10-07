from typing import Dict, Set

import attr

from gauss.graphs import Entity, Node


@attr.s(cmp=False)
class GraphMapping:
    m_ent: Dict[Entity, Entity] = attr.ib(factory=dict)
    m_node: Dict[Node, Node] = attr.ib(factory=dict)

    def copy(self):
        return GraphMapping(m_ent=self.m_ent.copy(), m_node=self.m_node.copy())

    def reverse(self) -> 'GraphMapping':
        return GraphMapping({v: k for k, v in self.m_ent.items()},
                            {v: k for k, v in self.m_node.items()})

    def apply_mapping(self, mapping: 'GraphMapping',
                      only_keys: bool = False, only_values: bool = False) -> 'GraphMapping':
        if only_keys:
            return GraphMapping({mapping.m_ent.get(k, k): v for k, v in self.m_ent.items()},
                                {mapping.m_node.get(k, k): v for k, v in self.m_node.items()})
        elif only_values:
            return GraphMapping({k: mapping.m_ent.get(v, v) for k, v in self.m_ent.items()},
                                {k: mapping.m_node.get(v, v) for k, v in self.m_node.items()})
        else:
            return GraphMapping({mapping.m_ent.get(k, k): mapping.m_ent.get(v, v) for k, v in self.m_ent.items()},
                                {mapping.m_node.get(k, k): mapping.m_node.get(v, v) for k, v in self.m_node.items()})

    def slice(self, nodes: Set[Node]):
        m_node = {k: v for k, v in self.m_node.items() if k in nodes}
        m_ent = {k.entity: v.entity for k, v in m_node.items()}
        return GraphMapping(m_ent, m_node)

    def update(self, other: 'GraphMapping'):
        self.m_ent.update(other.m_ent)
        self.m_node.update(other.m_node)

    def __or__(self, other):
        result = GraphMapping({**self.m_ent, **other.m_ent}, {**self.m_node, **other.m_node})
        return result


@attr.s(cmp=False, repr=False)
class GraphMultiMapping:
    m_ent: Dict[Entity, Set[Entity]] = attr.ib(factory=dict)
    m_node: Dict[Node, Set[Node]] = attr.ib(factory=dict)
