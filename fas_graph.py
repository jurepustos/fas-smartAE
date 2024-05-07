from abc import ABC, abstractmethod, abstractclassmethod
from typing import TypeVar, Self, Iterator, Generic
from copy import copy

Edge = TypeVar('Edge')
Node = TypeVar('Node')


class FASGraph(ABC, Generic[Node, Edge]):
    @abstractmethod
    def get_nodes(self) -> Iterator[Node]:
        raise NotImplementedError

    @abstractmethod
    def get_out_degree(self, node: Node) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_in_degree(self, node: Node) -> int:
        raise NotImplementedError

    @abstractmethod
    def iter_strongly_connected_components(self) -> Iterator[Self]:
        raise NotImplementedError

    @abstractmethod
    def remove_sinks(self):
        raise NotImplementedError

    @abstractmethod
    def remove_sources(self):
        raise NotImplementedError

    def get_forward_edges(self, ordering: list[int]) \
            -> tuple[list[Edge], Self]:
        forward_edges = []
        forward_graph = copy(self)
        for i in range(len(ordering)):
            edges = forward_graph.get_forward_edges_from(ordering, i)
            forward_edges.extend(edges)
            forward_graph.remove_edges(forward_edges)
            # TODO: instead of checking acyclicity, we could use SCCs instead
            if forward_graph.is_acyclic(ordering):
                break

        return forward_edges, forward_graph

    @abstractmethod
    def get_forward_edges_from(self, ordering: list[Node],
                               start_index: int) -> list[Edge]:
        raise NotImplementedError

    def get_backward_edges(self, ordering: list[int]) \
            -> tuple[list[Edge], Self]:
        backward_edges = []
        backward_graph = copy(self)
        for i in range(len(ordering)):
            edges = backward_graph.get_backward_edges_from(ordering, i)
            backward_edges.extend(edges)
            backward_graph.remove_edges(backward_edges)
            # TODO: instead of checking acyclicity, we could use SCCs instead
            if backward_graph.is_acyclic(ordering):
                break

        return backward_edges, backward_graph

    @abstractmethod
    def get_backward_edges_from(self, ordering: list[Node],
                                start_index: int) -> list[Edge]:
        raise NotImplementedError

    @abstractmethod
    def is_acyclic(self):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, edge: Edge):
        raise NotImplementedError

    @abstractmethod
    def remove_edge(self, edge: Edge):
        raise NotImplementedError

    @abstractmethod
    def remove_edges(self, edges: list[Edge]):
        raise NotImplementedError

    @abstractclassmethod
    def load_from_edge_list(cls, filename: str):
        raise NotImplementedError
