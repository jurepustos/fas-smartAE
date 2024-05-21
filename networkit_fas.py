import graphlib
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

    def remove_2cycles(self):
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
            yield self.__class__(
                GraphTools.subgraphFromNodes(self.graph, component_nodes)
            )

    def is_acyclic(self) -> bool:
        sorter = graphlib.TopologicalSorter()

        for node in self.graph.iterNodes():
            sorter.add(node)
        for source, target in self.graph.iterEdges():
            sorter.add(source, target)
        try:
            sorter.static_order()
            return True
        except graphlib.CycleError:
            return False

    def is_acyclic2(self) -> bool:
        visited = {node: False for node in self.graph.iterNodes()}

        def exploreNode(u):
            if visited[u]:
                raise Exception("Cycle detected")
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
        for neighbor in graph.iterNeighbors(node):
            if active_path[neighbor]:
                raise CycleDetectedError

        # add current node to the active path
        active_path[node] = True
        # set the previous node to the current node, so that it is removed
        # from the active path when backtracking
        prev_node = node

    return callback_inner
