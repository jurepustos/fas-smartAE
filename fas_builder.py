from collections import defaultdict
from copy import copy
from typing import Iterable, Self

from sortedcontainers import SortedList

from fas_graph import Edge, Node


class OrderingFASBuilder:
    def __init__(self: Self):
        self.removed_edges: list[Edge] = []
        self.smartAE_restored: SortedList[Edge] = SortedList()

    def merge(self, other: Self):
        self.removed_edges.extend(other.removed_edges)
        self.smartAE_restored.update(other.smartAE_restored)

    def add_removed_edges(self, edges: Iterable[Edge]):
        self.removed_edges.extend(edges)

    def add_smartAE_restored(self, edges: Iterable[Edge]):
        self.smartAE_restored.update(edges)


class FASBuilder:
    def __init__(self, node_labels: list[str]):
        self.fas_edges: list[tuple[str, str]] = []
        self.merged_edges: defaultdict[
            tuple[str, str], list[tuple[str, str]]
        ] = defaultdict(list)
        self.node_labels = node_labels
        self.ordering_instances: defaultdict[str, list[tuple[str, str]]] = (
            defaultdict(list)
        )

    def add_fas_edges(self, edges: list[Edge]):
        self.fas_edges.extend(
            (self.node_labels[u], self.node_labels[v]) for u, v in edges
        )

    def get_ordering_names(self) -> list[str]:
        return list(self.ordering_instances.keys())

    def add_merged_edges(self, merges: dict[Edge, list[Edge]]):
        for (u, v), merged_edges in merges.items():
            self.merged_edges[self.node_labels[u], self.node_labels[v]].extend(
                (self.node_labels[u], self.node_labels[v])
                for u, v in merged_edges
            )

    def add_ordering(self, name: str, ordering: OrderingFASBuilder):
        instance: list[tuple[str, str]] = []
        for u, v in ordering.removed_edges:
            if (u, v) not in ordering.smartAE_restored:
                instance.append((self.node_labels[u], self.node_labels[v]))

        self.ordering_instances[name] = instance

    def merge(self, other: Self):
        self.fas_edges.extend(other.fas_edges)
        self.merged_edges.update(other.merged_edges)
        for name, ordering_fas in other.ordering_instances.items():
            self.ordering_instances[name].extend(ordering_fas)

    def build_fas(self, name: str) -> list[tuple[str, str]]:
        instance = self.ordering_instances[name]

        fas = copy(self.fas_edges)
        merged_edges = copy(self.merged_edges)
        for u, v in instance:
            # if the current edge is merged from reductions, unmerge it
            if (u, v) in merged_edges:
                unmerged_u, unmerged_v = self.merged_edges[(u, v)].pop()
                if len(self.merged_edges[(u, v)]) == 0:
                    del self.merged_edges[(u, v)]
                fas.append((unmerged_u, unmerged_v))
            else:
                fas.append((u, v))
        return fas
