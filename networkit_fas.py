from copy import copy
from typing import Iterator, Self

from networkit.components import StronglyConnectedComponents
from networkit.graph import Graph
from networkit.graphio import EdgeListReader
from networkit.graphtools import GraphTools

from fas_graph import FASGraph


class NetworkitGraph(FASGraph):
    def __init__(
        self,
        networkit_graph: Graph,
        node_labels: dict[int, str] | None = None,
    ):
        self.graph = networkit_graph
        self.node_labels = node_labels
        self.acyclic = None
        self.topological_sort: list[int] | None = None
        self.inv_topological_sort: list[int] | None = None

    def get_node_labels(self) -> dict[int, str]:
        return self.node_labels

    def get_nodes(self) -> list[int]:
        return list(self.graph.iterNodes())

    def get_num_nodes(self) -> int:
        return self.graph.numberOfNodes()

    def get_num_edges(self) -> int:
        return self.graph.numberOfEdges()

    def get_out_degree(self, node: int) -> int:
        return self.graph.degreeOut(node)

    def get_in_degree(self, node: int) -> int:
        return self.graph.degreeIn(node)

    def iter_out_neighbors(self, node: int) -> Iterator[int]:
        return self.graph.iterNeighbors(node)

    def iter_in_neighbors(self, node: int) -> Iterator[int]:
        return self.graph.iterInNeighbors(node)

    def remove_neighbors_sink(self, node: int):
        removed = []
        for neighbor in self.iter_in_neighbors(node):
            if self.get_out_degree(neighbor) == 1:
                removed.append(neighbor)
                self.graph.removeNode(neighbor)

        for i in removed:
            self.remove_neighbors_sink(i)
        return

    def remove_sinks(self):
        sinks = [
            node for node in self.graph.iterNodes() if self.graph.degreeOut(node) == 0
        ]
        while sinks:
            sink = sinks.pop()
            for neighbor in self.iter_in_neighbors(sink):
                if self.get_out_degree(neighbor) == 1:
                    sinks.append(neighbor)
            self.graph.removeNode(sink)

    def remove_sources(self):
        sources = [
            node for node in self.graph.iterNodes() if self.graph.degreeIn(node) == 0
        ]
        while sources:
            source = sources.pop()
            for neighbor in self.iter_out_neighbors(source):
                if self.get_in_degree(neighbor) == 1:
                    sources.append(neighbor)
            self.graph.removeNode(source)

    def find_2cycles(self):
        twoCycles = []
        for u in self.get_nodes():
            for v in self.graph.iterNeighbors(u):
                for w in self.graph.iterNeighbors(v):
                    if u == w and u < v:
                        twoCycles.append((u, v))
        return twoCycles

    def remove_runs(self):
        for u in self.get_nodes():
            for v in self.iter_out_neighbors(u):
                while (
                    u != v
                    and self.get_in_degree(v) == 1
                    and self.get_out_degree(v) == 1
                ):
                    w = next(self.iter_out_neighbors(v))
                    if u < w:
                        self.graph.removeNode(v)
                        if not self.graph.hasEdge(u, w):
                            self.graph.addEdge(u, w)
                    v = w

    def remove_2cycles(self) -> tuple[list[tuple[int, int], dict[int, int]]:
        # TODO: return all guaranteed FAS vertices as well as a description of node merges
        FAS = []
        cy2 = self.find_2cycles()

        for pair in cy2:
            a = pair[0]
            b = pair[1]
            if self.get_out_degree(b) == 1:
                FAS.append((b, a))
                self.graph.removeNode(b)
            elif self.get_in_degree(b) == 1:
                FAS.append((a, b))
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
        if self.acyclic is not None:
            return self.acyclic

        try:
            self.topological_sort = GraphTools.topologicalSort(self.graph)
            self.inv_topological_sort = len(self.topological_sort) * [0]
            for i, node in enumerate(self.topological_sort):
                self.inv_topological_sort[node] = i
        except RuntimeError:
            self.acyclic = False
            return False

        self.acyclic = True
        return True

    def edge_preserves_acyclicity(self, edge: tuple[int, int]) -> bool:
        if not self.acyclic:
            # we only care if we know the graph is acyclic
            return False

        source, target = edge
        return self.inv_topological_sort[target] >= self.inv_topological_sort[source]

    def add_edges(self, edges: list[tuple[int, int]]):
        for edge in edges:
            self.add_edge(edge)

    def add_edge(self, edge: tuple[int, int]):
        # check if the edge violates the topological ordering,
        # making the graph cyclic.
        if not self.edge_preserves_acyclicity(edge):
            self.topological_sort = None
            self.inv_topological_sort = None
            self.acyclic = False

        source, target = edge
        is_new = self.graph.addEdge(source, target, checkMultiEdge=True)
        if not is_new:
            self.graph.increaseWeight(source, target, 1)

    def remove_edge(self, edge: tuple[int, int]):
        # removal of an edge can make the graph acyclic
        if not self.acyclic:
            self.acyclic = None

        source, target = edge
        if self.graph.weight(source, target) > 1:
            self.graph.increaseWeight(source, target, -1)
        else:
            self.graph.removeEdge(source, target)

    def remove_edges(self, edges: list[tuple[int, int]]):
        for edge in edges:
            self.remove_edge(edge)

    @classmethod
    def load_from_edge_list(cls, filename: str):
        """
        Load the graph from an edge-list representation.
        The resulting graph does not have isolated vertices.
        """
        reader = EdgeListReader("\t", 0, directed=True, continuous=False)
        graph = reader.read(filename)
        labels = reader.getNodeMap()
        inverse_labels = {}
        for label, node in labels.items():
            inverse_labels[node] = label
        graph.removeMultiEdges()
        graph.removeSelfLoops()
        return NetworkitGraph(graph, node_labels=inverse_labels)

    @classmethod
    def load_from_adjacency_list(cls, filename: str):
        """
        Load the graph from an adjacency-list representation.
        """
        graph = Graph(directed=True)
        labels = {}

        with open(filename, "r") as file:
            for line in file:
                nodes = list(map(int, line.strip().split()))
                source = nodes[0]
                if source not in labels:
                    labels[source] = graph.addNode()

                for target in nodes[1:]:
                    if target not in labels:
                        labels[target] = graph.addNode()
                    graph.addEdge(source, target, checkMultiEdges=True)
        graph.removeSelfLoops()

        inverse_labels = {}
        for label, node in labels.items():
            inverse_labels[node] = label

        return NetworkitGraph(graph, node_labels=inverse_labels)

    def __copy__(self):
        copied_graph = NetworkitGraph(
            copy(self.graph),
            node_labels=copy(self.node_labels),
        )
        copied_graph.acyclic = self.acyclic
        copied_graph.topological_sort = copy(self.topological_sort)
        copied_graph.inv_topological_sort = copy(self.inv_topological_sort)
        return copied_graph
