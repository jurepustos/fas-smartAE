from collections import defaultdict
from typing import Self

from sortedcontainers import SortedList

from fas_graph import Edge, Node


class OrderingFASEdges:
    def __init__(self, fas_edges: list[Edge], merged_edges: dict[Edge, list[Edge]]):
        self.fas_edges = fas_edges
        self.merged_edges = merged_edges
        self.removed_edges: list[Edge] = []
        self.smartAE_restored: SortedList[Edge] = SortedList()

    def build_fas(self, node_labels: list[str]):
        fas = [
            (node_labels[source], node_labels[target])
            for source, target in self.fas_edges
        ]
        for source, target in self.removed_edges:
            # skip if smartAE restored the edge
            if (source, target) not in self.smartAE_restored:
                # if the current edge is merged from reductions, unmerge it
                if (source, target) in self.merged_edges:
                    uunmerged_source, unmerged_target = self.merged_edges[
                        (source, target)
                    ].pop()
                    fas.append(
                        ([node_labels[uunmerged_source], node_labels[unmerged_target]])
                    )
                else:
                    fas.append(([node_labels[source], node_labels[target]]))
        return fas


class OrderingFASBuilder:
    def __init__(self, fas_edges: list[Edge], merged_edges: dict[Edge, list[Edge]]):
        self.fas_edges = fas_edges
        self.merged_edges = merged_edges
        self.forward = OrderingFASEdges(self.fas_edges, self.merged_edges)
        self.backward = OrderingFASEdges(self.fas_edges, self.merged_edges)

    def merge(self, other: Self):
        self.forward.removed_edges.extend(other.forward.removed_edges)
        self.forward.smartAE_restored.update(other.forward.smartAE_restored)
        self.backward.removed_edges.extend(other.backward.removed_edges)
        self.backward.smartAE_restored.update(other.backward.smartAE_restored)


class FASBuilder:
    def __init__(self):
        self.fas_edges: list[Edge] = []
        self.merged_edges = defaultdict(list)
        self.orderings: dict[OrderingFASBuilder] = defaultdict(
            lambda: OrderingFASBuilder(self.fas_edges, self.merged_edges)
        )

    def add_fas_edges(self, edges: list[Edge]):
        self.fas_edges.extend(edges)

    def add_merged_edges(self, merges: dict[Edge, list[Edge]]):
        for edge, merged_edges in merges.items():
            self.merged_edges[edge].merge(merged_edges)

    def ordering(self, name: str) -> OrderingFASBuilder:
        return self.orderings[name]

    def merge(self, other: Self):
        self.fas_edges.extend(other.fas_edges)
        self.merged_edges.update(other.merged_edges)
        for name, ordering_fas in other.orderings.items():
            self.orderings[name].merge(ordering_fas)
