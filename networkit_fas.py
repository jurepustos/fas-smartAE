import re
from collections import defaultdict
from copy import copy
from typing import Iterator, Self

from networkit.components import StronglyConnectedComponents
from networkit.graph import Graph
from networkit.graphtools import GraphTools

from fas_graph import Edge, FASGraph, Node


class NetworkitGraph(FASGraph):
    def __init__(
        self,
        networkit_graph: Graph,
        node_labels: list[str] | None = None,
        self_loops: list[Node] | None = None,
    ):
        self.graph = networkit_graph
        self.self_loops = self_loops if self_loops is not None else []
        self.node_labels = (
            node_labels
            if node_labels is not None
            else list(networkit_graph.iterNodes())
        )
        self.acyclic: bool | None = None
        self.added_backward_edge: Edge | None = None
        self.prev_topological_sort: list[Node] | None = None
        self.prev_inv_topological_sort: list[Node] | None = None
        self.topological_sort: list[Node] | None = None
        self.inv_topological_sort: list[Node] | None = None

    def get_node_labels(self) -> list[str]:
        return self.node_labels

    def get_nodes(self) -> list[Node]:
        return list(self.graph.iterNodes())

    def get_num_nodes(self) -> Node:
        return self.graph.numberOfNodes()

    def get_num_edges(self) -> Node:
        return self.graph.numberOfEdges()

    def get_out_degree(self, node: Node) -> int:
        return self.graph.degreeOut(node)

    def get_in_degree(self, node: Node) -> int:
        return self.graph.degreeIn(node)

    def iter_out_neighbors(self, node: Node) -> Iterator[Node]:
        return self.graph.iterNeighbors(node)

    def iter_in_neighbors(self, node: Node) -> Iterator[Node]:
        return self.graph.iterInNeighbors(node)

    def get_self_loops(self) -> list[Node]:
        return self.self_loops

    def remove_runs(self) -> dict[Edge, list[Edge]]:
        merged_edges: defaultdict[Edge, list[Edge]] = defaultdict(list)
        for u in self.get_nodes():
            # we need to collect them into a list because the internal neighbors
            # list changes during the algorithm, leading to potential crashes
            u_neighbors = list(self.iter_out_neighbors(u))
            for v in u_neighbors:
                while (
                    u != v
                    and self.get_in_degree(v) == 1
                    and self.get_out_degree(v) == 1
                ):
                    w = next(self.iter_out_neighbors(v))
                    if u != w:
                        self.remove_run(merged_edges, u, v, w)
                    v = w

        return merged_edges

    def remove_run(
        self,
        merged_edges: defaultdict[Edge, list[Edge]],
        u: Node,
        v: Node,
        w: Node,
    ):
        # uv was uxv in a previous step
        if (u, v) in merged_edges:
            merged_edges[(u, w)].extend(merged_edges[(u, v)])
            # since v only has u and w as neighbors
            del merged_edges[(u, v)]
        merged_edges[(u, w)].append((u, v))
        # merge weights
        uv_weight = self.graph.weight(u, v)
        vw_weight = self.graph.weight(v, w)
        uw_added_weight = min(uv_weight, vw_weight)
        self.graph.removeNode(v)
        self.graph.increaseWeight(u, w, uw_added_weight)

    def iter_2cycles(self) -> Iterator[Edge]:
        for u in self.get_nodes():
            u_neighbors = list(self.iter_out_neighbors(u))
            for v in u_neighbors:
                if u < v:
                    v_neighbors = list(self.iter_out_neighbors(v))
                    for w in v_neighbors:
                        if u == w:
                            yield u, v
                            yield v, u

    def remove_2cycles(self) -> list[Edge]:
        FAS: list[Edge] = []
        for a, b in self.iter_2cycles():
            ab_weight = self.get_edge_weight(a, b)
            ba_weight = self.get_edge_weight(b, a)
            if self.get_out_degree(b) == 1 and ab_weight >= ba_weight:
                FAS.extend([(b, a)] * ba_weight)
                self.graph.removeNode(b)
            elif self.get_in_degree(b) == 1 and ab_weight <= ba_weight:
                FAS.extend([(a, b)] * ab_weight)
                self.graph.removeNode(b)

        return FAS

    def iter_strongly_connected_components(self) -> Iterator[Self]:
        """
        Returns a list of strongly connected components
        """
        cc = StronglyConnectedComponents(self.graph)
        cc.run()
        for component_nodes in cc.getComponents():
            subgraph = GraphTools.subgraphFromNodes(self.graph, component_nodes)
            mapping = GraphTools.getContinuousNodeIds(subgraph)
            compact_subgraph = GraphTools.getCompactedGraph(subgraph, mapping)
            labels = len(mapping) * [""]
            for orig, mapped in mapping.items():
                labels[mapped] = self.node_labels[orig]

            yield self.__class__(compact_subgraph, node_labels=labels)

    def is_acyclic(self) -> bool:
        if self.acyclic is not None:
            return self.acyclic

        try:
            self.topological_sort = GraphTools.topologicalSort(self.graph)
        except RuntimeError:
            self.acyclic = False
            return False

        assert self.topological_sort is not None
        self.inv_topological_sort = len(self.topological_sort) * [0]
        for i, node in enumerate(self.topological_sort):
            self.inv_topological_sort[node] = i
        self.acyclic = True
        return True

    def get_edge_weight(self, source: Node, target: Node) -> int:
        return int(self.graph.weight(source, target))

    def edge_between_components(self, source: Node, target: Node) -> bool:
        cc = StronglyConnectedComponents(self.graph)
        cc.run()
        if cc.numberOfComponents() == self.get_num_nodes():
            self.acyclic = True
        else:
            self.acyclic = False
        partition = cc.getPartition()
        return not partition.inSameSubset(source, target)

    def edge_preserves_topology(self, source: Node, target: Node) -> bool:
        if not self.acyclic or self.inv_topological_sort is None:
            # we only know the answer if we know the graph is acyclic
            return False

        return (
            self.inv_topological_sort[target]
            >= self.inv_topological_sort[source]
        )

    def add_edges(self, edges: list[tuple[Node, Node]]):
        for source, target in edges:
            self.add_edge(source, target)

    def add_edge(self, source: Node, target: Node):
        # check if the edge violates the topological ordering,
        # making the graph cyclic.
        if self.acyclic and not self.edge_preserves_topology(source, target):
            self.acyclic = None
            if self.added_backward_edge is None:
                self.added_backward_edge = source, target
                self.prev_topological_sort = self.topological_sort
                self.prev_inv_topological_sort = self.inv_topological_sort
                self.topological_sort = None
                self.topological_sort = None
            else:
                # we only store acyclicity information for one violating edge back
                self.added_backward_edge = None
                self.topological_sort = None
                self.inv_topological_sort = None
                self.prev_topological_sort = None
                self.prev_inv_topological_sort = None

        self.graph.increaseWeight(source, target, 1)

    def remove_edge(self, source: Node, target: Node):
        if self.added_backward_edge == (source, target):
            self.added_backward_edge = None
            self.topological_sort = self.prev_topological_sort
            self.inv_topological_sort = self.prev_inv_topological_sort
            self.prev_topological_sort = None
            self.prev_inv_topological_sort = None
            self.acyclic = True

        if self.graph.weight(source, target) > 1:
            self.graph.increaseWeight(source, target, -1)
        else:
            self.graph.removeEdge(source, target)
            if not self.acyclic:
                self.acyclic = None

    def remove_edges(self, edges: list[tuple[Node, Node]]):
        for source, target in edges:
            self.remove_edge(source, target)

    @classmethod
    def load_from_edge_list(cls, filename: str) -> tuple[Self, dict[str, Node]]:
        """
        Load the graph from an edge-list representation.
        The resulting graph does not have isolated vertices.
        """
        graph = Graph(weighted=True, directed=True)
        labels: dict[str, Node] = {}
        self_loops: list[Node] = []

        with open(filename, "r") as file:
            for line in file:
                nodes = [word for word in line.strip().split()]
                source = nodes[0]

                # line is a comment
                if source[0] == "#":
                    continue

                if source not in labels:
                    labels[source] = graph.addNode()

                target = nodes[1]
                if target not in labels:
                    labels[target] = graph.addNode()
                if target == source:
                    self_loops.append(labels[source])
                graph.increaseWeight(labels[source], labels[target], 1)

        inverse_labels: list[str] = graph.numberOfNodes() * [""]
        for label, node in labels.items():
            inverse_labels[node] = label

        return cls(
            graph, node_labels=inverse_labels, self_loops=self_loops
        ), labels

    @classmethod
    def load_from_dot(cls, filename: str):
        graph = Graph(directed=True, weighted=True)
        labels: dict[str, Node] = {}
        self_loops: list[Node] = []

        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                # Skip empty lines and non-edge definitions
                if (
                    not line
                    or line.startswith("digraph")
                    or line.startswith("label")
                    or line.startswith("{")
                    or line.startswith("}")
                ):
                    continue

                # Match node definition lines
                node_match = re.match(r'\s*(\d+)\s*\[label="(\d+)"\];', line)
                if node_match:
                    node_id, node_label = map(int, node_match.groups())
                    if node_id not in labels:
                        labels[node_id] = graph.addNode()
                    continue

                # Match edge definition lines
                edge_match = re.match(
                    r'\s*(\d+)\s*->\s*(\d+)\s*\[label="\(\w=(\d+),\w=(\d+)\)"\];',
                    line,
                )
                if edge_match:
                    source, target, weight, _ = map(int, edge_match.groups())
                    if source not in labels:
                        labels[source] = graph.addNode()
                    if target not in labels:
                        labels[target] = graph.addNode()
                    if source == target:
                        self_loops.append(labels[source])
                    graph.increaseWeight(labels[source], labels[target], weight)

        inverse_labels = ["" for _ in range(graph.numberOfNodes())]
        for label, node in labels.items():
            inverse_labels[node] = str(label)

        return NetworkitGraph(
            graph, node_labels=inverse_labels, self_loops=self_loops
        ), labels

    @classmethod
    def load_from_adjacency_list(
        cls, filename: str
    ) -> tuple[Self, dict[str, Node]]:
        """
        Load the graph from an adjacency-list representation.
        """
        graph = Graph(directed=True, weighted=True)
        labels: dict[str, Node] = {}
        self_loops: list[Node] = []

        with open(filename, "r") as file:
            for line in file:
                nodes = list(line.strip().split())
                source = nodes[0]
                # line is a comment
                if source[0] == "#":
                    continue

                if source not in labels:
                    labels[source] = graph.addNode()

                for target in nodes[1:]:
                    if target not in labels:
                        labels[target] = graph.addNode()
                    if target == source:
                        self_loops.append(labels[source])
                    graph.increaseWeight(labels[source], labels[target], 1)

        inverse_labels: list[str] = graph.numberOfNodes() * [""]
        for label, node in labels.items():
            inverse_labels[node] = label

        return cls(
            graph, node_labels=inverse_labels, self_loops=self_loops
        ), labels

    def __copy__(self):
        copied_graph = NetworkitGraph(
            copy(self.graph),
            node_labels=copy(self.node_labels),
        )
        copied_graph.topological_sort = copy(self.topological_sort)
        copied_graph.inv_topological_sort = copy(self.inv_topological_sort)
        return copied_graph
