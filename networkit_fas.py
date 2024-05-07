from networkit.graph import Graph
from networkit.graphio import EdgeListReader
from networkit.components import StronglyConnectedComponents
from networkit.graphtools import GraphTools
from networkit.traversal import Traversal
from sortedcontainers import SortedList
from typing import Iterator, Self
from copy import copy

from .fas_graph import FASGraph


class NetworkitGraph(FASGraph[int, tuple[int, int]]):
    def __init__(self, networkit_graph: Graph):
        self.graph = networkit_graph

    def get_nodes(self) -> list[int]:
        return list(self.graph.iterNodes())

    def remove_sinks(self):
        sinks = [node for node in self.graph.iterNodes()
                 if self.graph.degreeOut(node) == 0]
        for sink in sinks:
            self.graph.removeNode(sink)

    def remove_sources(self):
        sources = [node for node in self.graph.iterNodes()
                   if self.graph.degreeIn(node) == 0]
        for source in sources:
            self.graph.removeNode(source)

    def iter_strongly_connected_components(self) -> Iterator[Self]:
        """
        Returns a list of strongly connected components
        """
        cc = StronglyConnectedComponents(self.graph)
        cc.run()
        for component_nodes in cc.getComponents():
            yield GraphTools.subgraphFromNodes(self.graph, component_nodes)

    def get_forward_edges_from(self, ordering: list[int],
                               start_index: int) -> list[tuple[int, int]]:
        forward_edges = []
        start_neighbors = SortedList(graph.iterOutNeighbors(ordering[start_index]))
        start = ordering[start_index]
        for node_index in range(start_index+1, len(ordering)):
            node = ordering[node_index]
            if node in start_neighbors:
                forward_edges.append((start, node))
        return forward_edges

    def get_backward_edges_from(self, ordering: list[int],
                                start_index: int) -> list[tuple[int, int]]:
        backward_edges = []
        start_neighbors = SortedList(graph.iterInNeighbors(ordering[start_index]))
        start = ordering[start_index]
        for node_index in range(start_index+1, len(ordering)):
            node = ordering[node_index]
            if node in start_neighbors:
                backward_edges.append((node, start))
        return backward_edges

    def is_acyclic(self) -> bool:
        try:
            Traversal.DFSfrom(self.graph, GraphTools.randomNode(self.graph),
                              acyclic_dfs_callback(self.graph))
            return True
        except CycleDetectedError:
            return False

    def add_edge(self, edge: tuple[int, int]):
        self.graph.addEdge(edge)

    def remove_edge(self, edge: tuple[int, int]):
        self.graph.removeEdge(edge)

    def remove_edges(self, edges: list[tuple[int, int]]):
        self.graph.removeEdges(edges)

    @classmethod
    def load_from_edge_list(cls, filename: str):
        """
        Load the graph from an edge-list representation.
        The resulting graph does not have isolated vertices.
        """
        reader = EdgeListReader(directed=True, separator='\t')
        return cls(reader.read(filename))

    def __copy__(self):
        return NetworkitGraph(copy(self.graph))


class CycleDetectedError(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "A cycle was detected"
        super().__init__(message)


def acyclic_dfs_callback(graph: Graph):
    active_path = {node: False for node in graph.iterNodes()}
    prev_node = None

    def callback_inner(node):
        nonlocal prev_node
        # remove the previously visited node from the active path
        if prev_node is not None and active_path[prev_node]:
            active_path[prev_node] = False

        # check if a neighbor is in the active path
        for neighbor in graph.iterOutNeighbors():
            if active_path[neighbor]:
                raise CycleDetectedError

        # add current node to the active path
        active_path[node] = True
        # set the previous node to the current node, so that it is removed
        # from the active path when backtracking
        prev_node = node

    return callback_inner
