"""
Enumeration strategies for rlang generators
"""
import collections
import itertools
import random

import attr
from atlas.operators import operator, OpInfo
from atlas.strategies import DfsStrategy
from typing import Collection, List, Any, Set, Dict, Union

from atlas.utils.stubs import stub

from gauss.graphs import Node, Graph

#  Stubs to shut up the errors
from gauss.synthesis.query_plans import QueryPlans


@stub
def SelectNode(*args, **kwargs):
    pass


@stub
def SelectConst(*args, **kwargs):
    pass


@stub
def SubsetNode(*args, **kwargs):
    pass


@stub
def OrderedSubsetNode(*args, **kwargs):
    pass


@stub
def FreshColumn(*args, **kwargs):
    pass


_FRESH_COL_CNT = 1000000


class DfsGraphStrategy(DfsStrategy):
    """ Basic DFS enumeration strategy adapted for graphs. """

    def __init__(self):
        super().__init__()
        self.placeholders: Dict[str, str] = {}

    @operator
    def SelectNode(self, domain: Collection[Node], **kwargs):
        for n in domain:
            yield n.value

    @operator
    def SelectConst(self, domain: Collection[Any], **kwargs):
        """ Assumes a finite and constant domain on every call """
        yield from domain

    @operator
    def SubsetNode(self, domain: Collection[Node], lengths: List[int] = None,
                   min_len: int = None, max_len: int = None, allow_empty: bool = False, **kwargs):

        domain = [n.value for n in domain]
        if min_len is None:
            min_len = 0 if allow_empty else 1
        if max_len is None:
            max_len = len(domain)

        if lengths is None:
            lengths = list(range(min_len, max_len + 1))

        for length in lengths:
            yield from itertools.combinations(domain, length)

    @operator
    def OrderedSubsetNode(self, domain: Collection[Node], lengths: List[int] = None,
                          min_len: int = None, max_len: int = None, allow_empty: bool = False, **kwargs):

        domain = [n.value for n in domain]
        if min_len is None:
            min_len = 0 if allow_empty else 1
        if max_len is None:
            max_len = len(domain)

        if lengths is None:
            lengths = list(range(min_len, max_len + 1))

        for length in lengths:
            yield from itertools.permutations(domain, length)

    @operator
    def FreshColumn(self, **kwargs):
        global _FRESH_COL_CNT
        _FRESH_COL_CNT += 1
        result = f"_COL__{_FRESH_COL_CNT}"
        self.placeholders[result] = kwargs.get('prefix', result)
        yield result


class RandomizedGraphStrategy(DfsGraphStrategy):
    """ Randomized version of the above. Just shuffles the domain before enumerating. """

    @operator
    def SelectNode(self, domain: Collection[Node], **kwargs):
        domain = [n.value for n in domain]
        random.shuffle(domain)
        yield from domain

    @operator
    def SelectConst(self, domain: Collection[Any], **kwargs):
        """ Assumes a finite and constant domain on every call """
        domain = list(domain)
        random.shuffle(domain)
        yield from domain

    @operator
    def SubsetNode(self, domain: Collection[Node], lengths: List[int] = None,
                   min_len: int = None, max_len: int = None, allow_empty: bool = False, **kwargs):

        domain = [n.value for n in domain]
        random.shuffle(domain)

        if min_len is None:
            min_len = 0 if allow_empty else 1
        if max_len is None:
            max_len = len(domain)

        if lengths is None:
            lengths = list(range(min_len, max_len + 1))

        for _ in range(100):
            length = random.choice(lengths)
            random.shuffle(domain)
            selected = domain[:length]

            yield selected

    @operator
    def OrderedSubsetNode(self, domain: Collection[Node], lengths: List[int] = None,
                          min_len: int = None, max_len: int = None, allow_empty: bool = False, **kwargs):

        domain = [n.value for n in domain]
        random.shuffle(domain)

        if min_len is None:
            min_len = 0 if allow_empty else 1
        if max_len is None:
            max_len = len(domain)

        if lengths is None:
            lengths = list(range(min_len, max_len + 1))

        for _ in range(100):
            length = random.choice(lengths)
            random.shuffle(domain)
            selected = domain[:length]

            yield selected


@attr.s
class Intelligence:
    all_uids: Set[str] = attr.ib()
    selected: Dict[str, Set[Union[Node, str]]] = attr.ib()
    not_selected: Dict[str, Set[Union[Node, str]]] = attr.ib()

    @classmethod
    def build(cls, strengthening_constraint: Graph) -> 'Intelligence':
        all_uids: Set[str] = set()
        selected: Dict[str, Set[Union[Node, str]]] = collections.defaultdict(set)
        not_selected: Dict[str, Set[Union[Node, str]]] = collections.defaultdict(set)

        for tag in strengthening_constraint.iter_tags():
            #  Tags correspond to SelectConst invocations
            #  Format is (SELECTED/NOT_SELECTED)@(val)@(uid)
            #  Values are guaranteed to be strings in the RLang domain.
            selected_bool, value, uid = tag.split('@')
            selected_bool = selected_bool == "SELECTED"
            all_uids.add(uid)
            if selected_bool:
                selected[uid].add(value)
            else:
                not_selected[uid].add(value)

        for tagged_edge in strengthening_constraint.iter_tagged_edges():
            #  Tagged edges should be self edges for the RLang domain.
            #  Format is (SELECTED/NOT_SELECTED)@(uid)
            node = tagged_edge.src
            selected_bool, uid = tagged_edge.tag.split('@')
            selected_bool = selected_bool == "SELECTED"
            all_uids.add(uid)

            if selected_bool:
                selected[uid].add(node)
            else:
                not_selected[uid].add(node)

        return Intelligence(all_uids=all_uids,
                            selected=selected,
                            not_selected=not_selected)

    def process(self, uid: str, domain: List[Union[str, Node]]):
        selected = set()
        not_selected = set()
        unknown = set()

        selected_set = self.selected[uid]
        not_selected_set = self.not_selected[uid]

        for d in domain:
            in_selected = d in selected_set
            in_not_selected = d in not_selected_set
            #  A value can both be selected and not_selected.
            #  This indicates a conflict and should stop enumeration.
            if in_selected:
                selected.add(d)
            if in_not_selected:
                not_selected.add(d)

            if (not in_selected) and (not in_not_selected):
                unknown.add(d)

        return selected, unknown, not_selected

    def __contains__(self, item):
        if isinstance(item, str):
            uid = item
            return uid in self.all_uids

        return False


class IntelligentEnumerationStrategy(DfsGraphStrategy):
    def __init__(self, strengthening_constraint: Graph):
        super().__init__()
        self.intelligence = Intelligence.build(strengthening_constraint)

    @operator
    def SelectNode(self, domain: Collection[Node], op_info: OpInfo = None, **kwargs):
        if op_info.uid in self.intelligence:
            selected, unknown, not_selected = self.intelligence.process(op_info.uid, list(domain))

            #  All of these conditions imply a conflict.
            if len(selected) > 1 or (len(unknown) + len(selected)) < 1 or not selected.isdisjoint(not_selected):
                return

            if len(selected) == 1:
                yield next(iter(selected)).value
                return

            for n in unknown:
                yield n.value

            return

        for n in domain:
            yield n.value

    @operator
    def SelectConst(self, domain: Collection[str], op_info: OpInfo = None, **kwargs):
        """ Assumes a finite and constant domain on every call """
        if op_info.uid in self.intelligence:
            selected, unknown, not_selected = self.intelligence.process(op_info.uid, list(domain))

            #  All of these conditions imply a conflict.
            if len(selected) > 1 or (len(unknown) + len(selected)) < 1 or not selected.isdisjoint(not_selected):
                return

            if len(selected) == 1:
                yield from selected
                return

            yield from unknown
            return

        yield from domain

    @operator
    def SubsetNode(self, domain: Collection[Node], lengths: List[int] = None,
                   min_len: int = None, max_len: int = None, allow_empty: bool = False,
                   op_info: OpInfo = None, **kwargs):

        domain = list(domain)
        if min_len is None:
            min_len = 0 if allow_empty else 1
        if max_len is None:
            max_len = len(domain)

        if lengths is None:
            lengths = list(range(min_len, max_len + 1))

        if op_info.uid in self.intelligence:
            selected, unknown, not_selected = self.intelligence.process(op_info.uid, list(domain))

            #  All of these conditions imply a conflict.
            if len(selected) + len(unknown) < min(lengths) or len(selected) > max(lengths) or \
                    not selected.isdisjoint(not_selected):
                return

            for length in lengths:
                if length < len(selected):
                    continue

                if length == len(selected):
                    yield [n.value for n in selected]
                    continue

                for s_unknown in itertools.combinations(unknown, length - len(selected)):
                    s = [n.value for n in itertools.chain(selected, s_unknown)]
                    yield s

            return

        domain = [n.value for n in domain]
        for length in lengths:
            yield from itertools.combinations(domain, length)

    @operator
    def OrderedSubsetNode(self, domain: Collection[Node], lengths: List[int] = None,
                          min_len: int = None, max_len: int = None, allow_empty: bool = False,
                          op_info: OpInfo = None, **kwargs):

        domain = list(domain)
        if min_len is None:
            min_len = 0 if allow_empty else 1
        if max_len is None:
            max_len = len(domain)

        if lengths is None:
            lengths = list(range(min_len, max_len + 1))

        if op_info.uid in self.intelligence:
            selected, unknown, not_selected = self.intelligence.process(op_info.uid, list(domain))

            #  All of these conditions imply a conflict.
            if len(selected) + len(unknown) < min(lengths) or len(selected) > max(lengths) or \
                    not selected.isdisjoint(not_selected):
                return

            for length in lengths:
                if length < len(selected):
                    continue

                if length == len(selected):
                    yield [n.value for n in selected]
                    continue

                for s_unknown in itertools.combinations(unknown, length - len(selected)):
                    s = [n.value for n in itertools.chain(selected, s_unknown)]
                    yield from itertools.permutations(s)

            return

        domain = [n.value for n in domain]

        for length in lengths:
            yield from itertools.permutations(domain, length)
