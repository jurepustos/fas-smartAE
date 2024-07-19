import re
from copy import copy
from functools import partial
from typing import Iterator, Self

from networkit.components import BiconnectedComponents, StronglyConnectedComponents
from networkit.graph import Graph
from networkit.graphtools import GraphTools
from networkit.structures import Partition
from networkit.traversal import Traversal

from fas_graph import Edge, FASGraph, Node


class NetworkitGraph(FASGraph):
    def __init__(
        self,
        networkit_graph: Graph,
        node_labels: list[str] | None = None,
        self_loops: list[Edge] | None = None,
    ):
        self.graph = networkit_graph
        self.self_loops = self_loops
        self.node_labels = (
            node_labels
            if node_labels is not None
            else list(networkit_graph.iterNodes())
        )
        self._acyclic: bool | None = None
        self._scc_partition: Partition | None = None
        self._added_backward_edge: Edge | None = None
        self._prev_topological_sort: list[Node] | None = None
        self._prev_inv_topological_sort: list[Node] | None = None
        self._topological_sort: list[Node] | None = None
        self._inv_topological_sort: list[Node] | None = None

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

    def remove_self_loops(self) -> list[Edge]:
        return self.self_loops if self.self_loops is not None else []

    def remove_runs(self) -> dict[Edge, Edge]:
        merged_edges: dict[Edge, Edge] = {}
        for u in self.get_nodes():
            # we need to collect them into a list because the internal neighbors
            # list changes during the algorithm, leading to potential crashes
            u_neighbors = list(self.iter_out_neighbors(u))
            for v in u_neighbors:
                last_merged_pair = None
                while (
                    u != v
                    and self.get_in_degree(v) == 1
                    and self.get_out_degree(v) == 1
                ):
                    w = next(self.iter_out_neighbors(v))
                    if u != w:
                        if last_merged_pair is not None and last_merged_pair[0] == (
                            u,
                            v,
                        ):
                            (x, y) = last_merged_pair[1]
                            last_merged_pair = (u, w), (x, y)
                        else:
                            last_merged_pair = (u, w), (u, v)
                        self.remove_run(u, v, w)
                    v = w

                if last_merged_pair is not None:
                    (u, v), (x, y) = last_merged_pair
                    merged_edges[(u, v)] = (x, y)

        return merged_edges

    def remove_run(
        self,
        u: Node,
        v: Node,
        w: Node,
    ):
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

    def iter_components(self) -> Iterator[Self]:
        """
        Returns a list of strongly connected components
        """
        scc = StronglyConnectedComponents(self.graph)
        scc.run()
        for component_nodes in scc.getComponents():
            if len(component_nodes) >= 2:
                component = GraphTools.subgraphFromNodes(self.graph, component_nodes)
                for bcc in self._iter_biconnected_components_of(component):
                    yield bcc

    def _iter_biconnected_components_of(self, subgraph: Graph) -> Iterator[Self]:
        undirected_subgraph = GraphTools.toUndirected(subgraph)
        bcc = BiconnectedComponents(undirected_subgraph)
        bcc.run()
        for component_nodes in bcc.getComponents():
            if len(component_nodes) >= 2:
                component = GraphTools.subgraphFromNodes(self.graph, component_nodes)
                mapping = GraphTools.getContinuousNodeIds(component)
                compact_component = GraphTools.getCompactedGraph(component, mapping)
                labels = len(mapping) * [""]
                for orig, mapped in mapping.items():
                    labels[mapped] = self.node_labels[orig]

                yield self.__class__(compact_component, node_labels=labels)

    def is_acyclic(self) -> bool:
        if self._acyclic is not None:
            return self._acyclic

        try:
            self._topological_sort = GraphTools.topologicalSort(self.graph)
        except RuntimeError:
            self._acyclic = False
            return False

        assert self._topological_sort is not None
        self._inv_topological_sort = len(self._topological_sort) * [0]
        for i, node in enumerate(self._topological_sort):
            self._inv_topological_sort[node] = i
        self._acyclic = True
        return True

    def get_edge_weight(self, source: Node, target: Node) -> int:
        return int(self.graph.weight(source, target))

    def edge_between_components(self, source: Node, target: Node) -> bool:
        if self._scc_partition is None:
            cc = StronglyConnectedComponents(self.graph)
            cc.run()
            if cc.numberOfComponents() == self.get_num_nodes():
                self._acyclic = True
            else:
                self._acyclic = False
            self._scc_partition = cc.getPartition()

        assert self._scc_partition is not None
        return not self._scc_partition.inSameSubset(source, target)

    def edge_preserves_topology(self, source: Node, target: Node) -> bool:
        if not self._acyclic or self._inv_topological_sort is None:
            # we only know the answer if we know the graph is acyclic
            return False

        return self._inv_topological_sort[target] >= self._inv_topological_sort[source]

    def add_edges(self, edges: list[tuple[Node, Node]]):
        for source, target in edges:
            self.add_edge(source, target)

    def add_edge(self, source: Node, target: Node):
        # check if the edge violates the topological ordering,
        # making the graph cyclic.
        if self._acyclic and not self.edge_preserves_topology(source, target):
            self._acyclic = None
            self._scc_partition = None
            if self._added_backward_edge is None:
                self._added_backward_edge = source, target
                self._prev_topological_sort = self._topological_sort
                self._prev_inv_topological_sort = self._inv_topological_sort
                self._topological_sort = None
                self._topological_sort = None
            else:
                # we only store acyclicity information for one violating edge back
                self._added_backward_edge = None
                self._topological_sort = None
                self._inv_topological_sort = None
                self._prev_topological_sort = None
                self._prev_inv_topological_sort = None

        self.graph.increaseWeight(source, target, 1)

    def remove_edge(self, source: Node, target: Node):
        if self._added_backward_edge == (source, target):
            self._added_backward_edge = None
            self._topological_sort = self._prev_topological_sort
            self._inv_topological_sort = self._prev_inv_topological_sort
            self._prev_topological_sort = None
            self._prev_inv_topological_sort = None
            self._acyclic = True

        if self._scc_partition is not None and not self.edge_between_components(
            source, target
        ):
            self._scc_partition = None

        if self.graph.weight(source, target) > 1:
            self.graph.increaseWeight(source, target, -1)
        else:
            self.graph.removeEdge(source, target)
            if not self._acyclic:
                self._acyclic = None

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
        self_loops: list[Edge] = []

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
                    self_loops.append((labels[source], labels[source]))
                graph.increaseWeight(labels[source], labels[target], 1)

        inverse_labels: list[str] = graph.numberOfNodes() * [""]
        for label, node in labels.items():
            inverse_labels[node] = label

        return cls(graph, node_labels=inverse_labels, self_loops=self_loops), labels

    @classmethod
    def load_from_dot(cls, filename: str):
        graph = Graph(directed=True, weighted=True)
        labels: dict[str, Node] = {}
        self_loops: list[Edge] = []

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
                    node_id, node_label = node_match.groups()
                    if node_id not in labels:
                        labels[node_id] = graph.addNode()
                    continue

                # Match edge definition lines
                edge_match = re.match(
                    r'\s*(\d+)\s*->\s*(\d+)\s*\[label="\(\w=(\d+),\w=(\d+)\)"\];',
                    line,
                )
                if edge_match:
                    source, target, weight, _ = edge_match.groups()
                    if source not in labels:
                        labels[source] = graph.addNode()
                    if target not in labels:
                        labels[target] = graph.addNode()
                    if source == target:
                        self_loops.append((labels[source], labels[source]))
                    else:
                        graph.increaseWeight(labels[source], labels[target], weight)

        inverse_labels = ["" for _ in range(graph.numberOfNodes())]
        for label, node in labels.items():
            inverse_labels[node] = str(label)

        return NetworkitGraph(
            graph, node_labels=inverse_labels, self_loops=self_loops
        ), labels

    @classmethod
    def load_from_adjacency_list(cls, filename: str) -> tuple[Self, dict[str, Node]]:
        """
        Load the graph from an adjacency-list representation.
        """
        graph = Graph(directed=True, weighted=True)
        labels: dict[str, Node] = {}
        self_loops: list[Edge] = []

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
                        self_loops.append((labels[source], labels[source]))
                    else:
                        graph.increaseWeight(labels[source], labels[target], 1)

        inverse_labels: list[str] = graph.numberOfNodes() * [""]
        for label, node in labels.items():
            inverse_labels[node] = label

        return cls(graph, node_labels=inverse_labels, self_loops=self_loops), labels

    def __copy__(self):
        copied_graph = NetworkitGraph(
            copy(self.graph),
            node_labels=copy(self.node_labels),
        )
        copied_graph._acyclic = copy(self._acyclic)
        copied_graph._added_backward_edge = copy(self._added_backward_edge)
        copied_graph._topological_sort = copy(self._topological_sort)
        copied_graph._inv_topological_sort = copy(self._inv_topological_sort)
        copied_graph._prev_topological_sort = copy(self._prev_topological_sort)
        copied_graph._prev_inv_topological_sort = copy(self._prev_inv_topological_sort)
        return copied_graph
