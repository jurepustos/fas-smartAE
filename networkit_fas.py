import traceback
from copy import copy
from typing import Iterator, Self
import graphlib

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
    
    def get_edge(self, source: int, target: int) -> tuple[int, int]:
        return (source, target)

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
    
    @classmethod
    def load_from_adjacency_list(cls, filename: str):
        """
        Load the graph from an adjacency-list representation.
        """
        graph = Graph(directed=True)

        with open(filename, 'r') as file:
            line_count = sum(1 for line in file)

        graph.addNodes(line_count +1)
        #print(graph.numberOfNodes())

        with open(filename, 'r') as file:
            for line in file:
                nodes = list(map(int, line.strip().split()))
                source = nodes[0]

                for target in nodes[1:]:
                    print(source, target)
                    graph.addEdge(source, target)
        graph.removeMultiEdges()
        graph.removeSelfLoops()

        #TODO: remove isolated nodes
        return cls(graph)
    


    def __copy__(self):
        return NetworkitGraph(copy(self.graph))
    

    def is_acyclic(self) -> bool:
        if self.graph.numberOfNodes() == 0 or self.graph.numberOfEdges() == 0:
            return True

        try:
            Traversal.DFSfrom(
                self.graph,
                next(self.graph.iterNodes()),
                acyclic_dfs_callback(self.graph),
            )
            return True
        except RuntimeError:
            return False
    

    def is_acyclic_topologically(self) -> bool:
        sorter = graphlib.TopologicalSorter()

        for node in self.graph.iterNodes():
            sorter.add(node)
        for source, target in self.graph.iterEdges():
            sorter.add(source, target) 
        try:
            sorted_order = [*sorter.static_order()]
            #print(sorted_order)
            return True
        except graphlib.CycleError:
            #print("Error: Graph has a cycle")
            return False
        

    def find_2cycles(self):
        twoCycles = []
        for u in self.get_nodes():
            for v in self.graph.iterNeighbors(u):
                for w in self.graph.iterNeighbors(v):
                    if u == w and u < v:
                        twoCycles.append((u, v))
        return twoCycles


    def remove_runs(self):   #remove runs larger than 2
        for u in self.get_nodes():
            for v in self.graph.iterNeighbors(u):
                while self.get_in_degree(v) == 1 and self.get_out_degree(v) == 1:
                    w = next(self.iter_out_neighbors(v))
                    if u < w:
                        self.graph.removeNode(v)
                        if not self.graph.hasEdge(u, w):
                            self.graph.addEdge(u, w)
                    v = w
        return


    def remove_2cycles(self):
        FAS = []
        cy2 = self.find_2cycles()

        for pair in cy2:
            a = pair[0]
            b = pair[1]
            
            if self.get_out_degree(b) == 1:
                edge = (b, a)
                FAS.append(edge)
                self.graph.removeNode(b)
            elif self.get_in_degree(b) == 1: 
                edge = (a, b)
                FAS.append(edge)
                self.graph.removeNode(b)    
        
        return FAS


class CycleDetectedError(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "A cycle was detected"
        super().__init__(message)


def acyclic_dfs_callback(graph: Graph):
    visited = {node: False for node in graph.iterNodes()}

    def callback_inner(node):
        # add current node to the active path
        visited[node] = True

        # check if a neighbor is in the active path
        for neighbor in graph.iterNeighbors(node):
            if visited[neighbor]:
                raise CycleDetectedError

    return callback_inner