from graph_tool import Graph, Vertex, Edge
from graph_tool.topology import label_components, is_DAG

from typing import Iterator, Self

from .fas_graph import FASGraph


class GTGraph(FASGraph[Vertex, Edge]):
    def __init__(self, gt_graph: Graph):
        self.graph = gt_graph

    def get_nodes(self) -> Iterator[Vertex]:
        return self.graph.vertices()

    def get_edge(self, source: Vertex, target: Vertex) -> Edge | None:
        return self.graph.edge(source, target)

    def get_out_degree(self, vertex: Vertex):
        return vertex.out_degree()

    def get_in_degree(self, vertex: Vertex):
        return vertex.in_degree()

    def iter_out_neighbors(self, vertex: Vertex) -> Iterator[Vertex]:
        return self.graph.iter_out_neighbors(vertex)

    def iter_in_neighbors(self, vertex: Vertex) -> Iterator[Vertex]:
        return self.graph.iter_in_neighbors(vertex)

    def iter_strongly_connected_components(self) -> Iterator[Self]:
        comp_vprop, hist = label_components(self.graph)
        for comp_index in range(self.graph.num_vertices()):
            if hist[comp_index] >= 2:
                vprop = self.graph.new_vp("bool", vals=comp_vprop.a == comp_index)
                self.graph.set_vertex_filter(vprop)
                comp = Graph(self.graph, prune=True)
                self.graph.clear_filters()
                yield GTGraph(comp)

    def remove_sinks(self):
        out_degree_vp = self.graph.degree_property_map("out")
        vp = self.graph.new_vp("bool", vals=out_degree_vp.a > 0)
        self.graph.set_vertex_filter(vp)
        self.graph.purge_vertices()
        self.graph.clear_filters()

    def remove_sources(self):
        in_degree_vp = self.graph.degree_property_map("in")
        vp = self.graph.new_vp("bool", vals=in_degree_vp.a > 0)
        self.graph.set_vertex_filter(vp)
        self.graph.purge_vertices()
        self.graph.clear_filters()

    def is_acyclic(self):
        return is_DAG(self.graph)

    def add_edge(self, edge: Edge):
        self.graph.add_edge(edge.source, edge.target)

    def remove_edge(self, edge: Edge):
        self.graph.remove_edge(edge)

    def remove_edges(self, edges: list[Edge]):
        for edge in edges:
            self.graph.remove(edge)

    def load_from_edge_list(cls, filename: str):
        edges = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                if line[0] == '#':
                    continue

                edge = tuple(line.split(' ')[:2])
                edges.append(edge)

                line = f.readline()

        graph = Graph(edges, fast_edge_removal=True)
        return GTGraph(graph)

    def __copy__(self):
        return GTGraph(self.graph.copy())
