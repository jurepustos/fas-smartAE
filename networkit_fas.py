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
        self.node_labels = node_labels
        self.topological_sort: list[Node] | None = None
        self.inv_topological_sort: list[Node] | None = None

    def get_node_labels(self) -> dict[Node, str]:
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

    def get_self_loop_nodes(self) -> list[Node]:
        return self.self_loops

    def find_2cycles(self) -> list[Edge]:
        twoCycles: list[Edge] = []
        for u in self.get_nodes():
            for v in self.iter_out_neighbors(u):
                if u < v:
                    for w in self.iter_out_neighbors(v):
                        if u == w:
                            twoCycles.append((u, v))
        return twoCycles

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
        self, merged_edges: defaultdict[Edge, list[Edge]], u: Node, v: Node, w: Node
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

    def remove_2cycles(self) -> list[Edge]:
        FAS: list[Edge] = []
        cy2 = self.find_2cycles()

        for a, b in cy2:
            ab_weight = self.get_edge_weight(a, b)
            ba_weight = self.get_edge_weight(b, a)
            if self.get_out_degree(b) == 1 and ab_weight <= ba_weight:
                FAS.extend([(a, b)] * ab_weight)
                self.graph.removeNode(b)
            elif self.get_in_degree(b) == 1 and ab_weight >= ba_weight:
                FAS.extend([(b, a)] * ba_weight)
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
            labels = {}
            for orig_node, mapped_node in mapping.items():
                labels[mapped_node] = self.node_labels[orig_node]

            yield NetworkitGraph(compact_subgraph, node_labels=labels)

    def is_acyclic(self) -> bool:
        if self.topological_sort is not None:
            return True

        try:
            self.topological_sort = GraphTools.topologicalSort(self.graph)
        except RuntimeError:
            return False

        self.inv_topological_sort = len(self.topological_sort) * [0]
        for i, node in enumerate(self.topological_sort):
            self.inv_topological_sort[node] = i
        return True

    def edge_preserves_topology(self, source: Node, target: Node) -> bool:
        if self.inv_topological_sort is None:
            # we only know the answer if we know the graph is acyclic
            return False

        return self.inv_topological_sort[target] >= self.inv_topological_sort[source]

    def get_edge_weight(self, source: Node, target: Node) -> int:
        return int(self.graph.weight(source, target))

    def add_edges(self, edges: list[tuple[Node, Node]]):
        for source, target in edges:
            self.add_edge(source, target)

    def add_edge(self, source: Node, target: Node):
        # check if the edge violates the topological ordering,
        # making the graph cyclic.
        if not self.edge_preserves_topology(source, target):
            self.topological_sort = None
            self.inv_topological_sort = None

        self.graph.increaseWeight(source, target, 1)

    def remove_edge(self, source: Node, target: Node):
        if self.graph.weight(source, target) > 1:
            self.graph.increaseWeight(source, target, -1)
        else:
            self.graph.removeEdge(source, target)

    def remove_edges(self, edges: list[tuple[Node, Node]]):
        for source, target in edges:
            self.remove_edge(source, target)

    @classmethod
    def load_from_edge_list(cls, filename: str):
        """
        Load the graph from an edge-list representation.
        The resulting graph does not have isolated vertices.
        """
        graph = Graph(weighted=True, directed=True)
        labels: dict[str, Node] = {}
        self_loops: list[Node] = []

        with open(filename, "r") as file:
            for line in file:
                nodes = [int(word) for word in line.strip().split()]
                source = nodes[0]

                # line is a comment
                if source == "#":
                    continue

                if source not in labels:
                    labels[source] = graph.addNode()

                target = nodes[1]
                if target not in labels:
                    labels[target] = graph.addNode()
                if target == source:
                    self_loops.append(source)
                graph.increaseWeight(labels[source], labels[target], 1)

        inverse_labels: list[str] = graph.numberOfNodes() * [""]
        for label, node in labels.items():
            inverse_labels[node] = label

        return NetworkitGraph(graph, node_labels=inverse_labels, self_loops=self_loops)

    @classmethod
    def load_from_adjacency_list(cls, filename: str):
        """
        Load the graph from an adjacency-list representation.
        """
        graph = Graph(directed=True, weighted=True)
        labels: dict[str, Node] = {}
        self_loops: list[Node] = []

        with open(filename, "r") as file:
            for line in file:
                nodes = list(map(int, line.strip().split()))
                source = nodes[0]
                # line is a comment
                if source == "#":
                    continue
                if source == "#":
                    continue

                if source not in labels:
                    labels[source] = graph.addNode()

                for target in nodes[1:]:
                    if target not in labels:
                        labels[target] = graph.addNode()
                    if target == source:
                        self_loops.append(source)
                    graph.increaseWeight(labels[source], labels[target], 1)

        inverse_labels: list[str] = graph.numberOfNodes() * [""]
        for label, node in labels.items():
            inverse_labels[node] = label

        return NetworkitGraph(graph, node_labels=inverse_labels, self_loops=self_loops)

    def __copy__(self):
        copied_graph = NetworkitGraph(
            copy(self.graph),
            node_labels=copy(self.node_labels),
        )
        copied_graph.topological_sort = copy(self.topological_sort)
        copied_graph.inv_topological_sort = copy(self.inv_topological_sort)
        return copied_graph
