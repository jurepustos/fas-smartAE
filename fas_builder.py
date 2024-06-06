from collections import defaultdict
from copy import copy
from typing import Self

from fas_graph import Edge, Node


class OrderingFASEdges:
    def __init__(self):
        self.removed_edges: list[Edge] = []
        self.smartAE_restored: list[Edge] = []


class OrderingFASBuilder:
    def __init__(self):
        self.fas_edges: list[Edge] = []
        self.merged_edges: dict[Edge, list[Edge]] = {}
        self.forward = OrderingFASEdges()
        self.backward = OrderingFASEdges()

    def merge(self, other: Self):
        self.forward.removed_edges.extend(other.forward.removed_edges)
        self.forward.smartAE_restored.extend(other.forward.smartAE_restored)
        self.backward.removed_edges.extend(other.backward.removed_edges)
        self.backward.smartAE_restored.extend(other.backward.smartAE_restored)

    def build_fas(self, node_labels: dict[Node, str]) -> list[Edge]:
        fas = []
        raise NotImplementedError
        return fas


class FASBuilder:
    def __init__(self):
        self.fas_edges: list[Edge] = []
        self.merged_edges = defaultdict(list)
        self.orderings: dict[OrderingFASBuilder] = defaultdict(OrderingFASBuilder)

    def add_fas_edges(self, edges: list[Edge]):
        self.fas_edges.extend(edges)

    def add_merged_edges(self, merges: dict[Edge, list[Edge]]):
        for edge, merged_edges in merges.items():
            self.merged_edges[edge].extend(merged_edges)

    def ordering(self, name: str) -> OrderingFASBuilder:
        return self.orderings[name]

    def add_ordering(self, name: str, ordering: OrderingFASBuilder):
        self.orderings[name] = ordering
        self.fas_edges = self.fas_edges
        self.merged_edges = self.merged_edges

    def merge(self, other: Self):
        self.fas_edges.extend(other.fas_edges)
        self.merged_edges.extend(other.merged_edges)
        for name, ordering_fas in other.orderings.items():
            self.orderings[name].merge(ordering_fas)
