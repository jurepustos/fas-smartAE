from collections import defaultdict
from copy import copy
from typing import Self

from sortedcontainers import SortedList

from fas_graph import Edge, Node


class OrderingFASEdges:
    def __init__(
        self,
        fas_edges: list[tuple[str, str]],
        merged_edges: dict[Edge, list[tuple[str, str]]],
        node_labels: list[str],
    ):
        self.fas_edges = fas_edges
        self.merged_edges = merged_edges
        self.node_labels = node_labels
        self.removed_edges: list[tuple[str, str]] = []
        self.smartAE_restored: SortedList[tuple[str, str]] = SortedList()

    def add_removed_edges(self, edges: list[Edge]):
        for source, target in edges:
            self.removed_edges.append(
                (self.node_labels[source], self.node_labels[target])
            )

    def add_smartAE_restored(self, edges: list[Edge]):
        for source, target in edges:
            self.smartAE_restored.add(
                (self.node_labels[source], self.node_labels[target])
            )

    def merge(self, other: Self):
        self.removed_edges.extend(other.removed_edges)
        self.smartAE_restored.update(other.smartAE_restored)

    def build_fas(self):
        fas = copy(self.fas_edges)
        for source, target in self.removed_edges:
            # skip if smartAE restored the edge
            if (source, target) not in self.smartAE_restored:
                # if the current edge is merged from reductions, unmerge it
                if (source, target) in self.merged_edges:
                    uunmerged_source, unmerged_target = self.merged_edges[
                        (source, target)
                    ].pop()
                    fas.append((uunmerged_source, unmerged_target))
                else:
                    fas.append((source, target))
        return fas


class OrderingFASBuilder:
    def __init__(
        self,
        fas_edges: list[tuple[str, str]],
        merged_edges: dict[Edge, list[tuple[str, str]]],
        node_labels: list[str],
    ):
        self.fas_edges = fas_edges
        self.merged_edges = merged_edges
        self.node_labels = node_labels
        self.forward = OrderingFASEdges(
            self.fas_edges, self.merged_edges, self.node_labels
        )
        self.backward = OrderingFASEdges(
            self.fas_edges, self.merged_edges, self.node_labels
        )

    def merge(self, other: Self):
        self.forward.merge(other.forward)
        self.backward.merge(other.backward)


class FASBuilder:
    def __init__(self, node_labels: list[str]):
        self.fas_edges: list[tuple[str, str]] = []
        self.merged_edges: defaultdict[Edge, list[tuple[str, str]]] = (
            defaultdict(list)
        )
        self.node_labels = node_labels
        self.orderings: defaultdict[str, OrderingFASBuilder] = defaultdict(
            lambda: OrderingFASBuilder(
                self.fas_edges, self.merged_edges, self.node_labels
            )
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

    def ordering(self, name: str) -> OrderingFASBuilder:
        self.orderings[name] = OrderingFASBuilder(
            self.fas_edges, self.merged_edges, self.node_labels
        )

        return self.orderings[name]

    def merge(self, other: Self):
        self.fas_edges.extend(other.fas_edges)
        self.merged_edges.update(other.merged_edges)
        for name, ordering_fas in other.orderings.items():
            self.orderings[name].merge(ordering_fas)
