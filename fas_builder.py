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
        self.merged_edges: defaultdict[Edge, list[tuple[str, str]]] = (
            defaultdict(list)
        )
        self.node_labels = node_labels
        self.orderings: defaultdict[str, OrderingFASBuilder] = defaultdict(
            OrderingFASBuilder
        )

    def add_fas_edges(self, edges: list[Edge]):
        self.fas_edges.extend(
            (self.node_labels[u], self.node_labels[v]) for u, v in edges
        )

    def add_merged_edges(self, merges: dict[Edge, list[Edge]]):
        for edge, merged_edges in merges.items():
            self.merged_edges[edge].extend(
                (self.node_labels[u], self.node_labels[v])
                for u, v in merged_edges
            )

    def add_ordering(self, name: str, ordering: OrderingFASBuilder):
        self.orderings[name] = ordering

    def merge(self, other: Self):
        self.fas_edges.extend(other.fas_edges)
        self.merged_edges.update(other.merged_edges)
        for name, ordering_fas in other.orderings.items():
            self.orderings[name].merge(ordering_fas)

    def build_fas(self, name: str) -> list[tuple[str, str]]:
        ordering_builder = self.orderings[name]

        fas = copy(self.fas_edges)
        merged_edges = copy(self.merged_edges)
        for source, target in ordering_builder.removed_edges:
            # skip if smartAE restored the edge
            if (source, target) not in ordering_builder.smartAE_restored:
                # if the current edge is merged from reductions, unmerge it
                if (source, target) in merged_edges:
                    unmerged_source, unmerged_target = self.merged_edges[
                        (source, target)
                    ].pop()
                    fas.append((unmerged_source, unmerged_target))
                else:
                    fas.append(
                        (self.node_labels[source], self.node_labels[target])
                    )
        return fas
