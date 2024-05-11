from abc import ABC, abstractmethod, abstractclassmethod
from typing import TypeVar, Self, Iterator, Generic
from copy import copy

from sortedcontainers import SortedList

Edge = TypeVar('Edge')
Node = TypeVar('Node')


class FASGraph(ABC, Generic[Node, Edge]):
    @abstractmethod
    def get_nodes(self) -> Iterator[Node]:
        raise NotImplementedError

    @abstractmethod
    def get_edge(self, source: Node, target: Node) -> Edge | None:
        raise NotImplementedError

    @abstractmethod
    def get_out_degree(self, node: Node) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_in_degree(self, node: Node) -> int:
        raise NotImplementedError

    @abstractmethod
    def iter_out_neighbors(self, node: Node) -> Iterator[Node]:
        raise NotImplementedError

    @abstractmethod
    def iter_in_neighbors(self, node: Node) -> Iterator[Node]:
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
