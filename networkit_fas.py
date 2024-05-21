import traceback
from copy import copy
from typing import Iterator, Self

from networkit.components import StronglyConnectedComponents
from networkit.graph import Graph
from networkit.graphio import EdgeListReader
from networkit.graphtools import GraphTools
from networkit.traversal import Traversal

from fas_graph import FASGraph


class NetworkitGraph(FASGraph):
    def __init__(self, networkit_graph: Graph):
        self.graph = networkit_graph

    def get_nodes(self) -> list[int]:
        return list(self.graph.iterNodes())

    def get_num_nodes(self) -> int:
        return self.graph.numberOfNodes()

    def get_out_degree(self, node: int) -> int:
        return self.graph.degreeOut(node)

    def get_in_degree(self, node: int) -> int:
        return self.graph.degreeIn(node)

    def iter_out_neighbors(self, node: int) -> Iterator[int]:
        return self.graph.iterNeighbors(node)

    def iter_in_neighbors(self, node: int) -> Iterator[int]:
        return self.graph.iterInNeighbors(node)

    def remove_sinks(self):
        sinks = [
            node for node in self.graph.iterNodes() if self.graph.degreeOut(node) == 0
        ]
        for sink in sinks:
            self.graph.removeNode(sink)

    def remove_sources(self):
        sources = [
            node for node in self.graph.iterNodes() if self.graph.degreeIn(node) == 0
        ]
        for source in sources:
            self.graph.removeNode(source)

    def iter_strongly_connected_components(self) -> Iterator[Self]:
        """
        Returns a list of strongly connected components
        """
        cc = StronglyConnectedComponents(self.graph)
        cc.run()
        for component_nodes in cc.getComponents():
            yield self.__class__(
                GraphTools.subgraphFromNodes(self.graph, component_nodes)
            )

    def is_acyclic(self) -> bool:
        visited = {node: False for node in self.graph.iterNodes()}

        def exploreNode(u):
            if visited[u]:
                raise Exception('Cycle detected')
            visited[u] = True

        for node in self.graph.iterNodes():
            if not visited[node]:
                try:
                    Traversal.DFSfrom(self.graph, node, exploreNode)
                except Exception:
                    return False
        return True

    def add_edge(self, edge: tuple[int, int]):
        self.graph.addEdge(*edge)

    def remove_edge(self, edge: tuple[int, int]):
        self.graph.removeEdge(*edge)

    def remove_edges(self, edges: list[tuple[int, int]]):
        for edge in edges:
            self.graph.removeEdge(*edge)

    @classmethod
    def load_from_edge_list(cls, filename: str):
        """
        Load the graph from an edge-list representation.
        The resulting graph does not have isolated vertices.
        """
        reader = EdgeListReader("\t", 0, directed=True)
        graph = reader.read(filename)
        graph.removeMultiEdges()
        graph.removeSelfLoops()
        return cls(graph)

    def __copy__(self):
        return NetworkitGraph(copy(self.graph))
