from abc import ABC, abstractmethod
from typing import Iterator, Self


class FASGraph(ABC):
    @abstractmethod
    def get_node_labels(self) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def get_nodes(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def get_num_nodes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_out_degree(self, node: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_in_degree(self, node: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def iter_out_neighbors(self, node: int) -> Iterator[int]:
        raise NotImplementedError

    @abstractmethod
    def iter_in_neighbors(self, node: int) -> Iterator[int]:
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
    def remove_runs(self):
        raise NotImplementedError

    @abstractmethod
    def remove_2cycles(self) -> list[tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def is_acyclic(self):
        raise NotImplementedError

    @abstractmethod
    def edge_preserves_acyclicity(self, edge: tuple[int, int]) -> bool:
        """
        Returns True if adding the edge is guaranteed to keep the graph acyclic.
        Otherwise, return False. In particular, if the graph is currently not acyclic, 
        return False.
        """
        raise NotImplementedError

    @abstractmethod
    def add_edges(self, edges: list[tuple[int, int]]):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, edge: tuple[int, int]):
        raise NotImplementedError

    @abstractmethod
    def remove_edge(self, edge: tuple[int, int]):
        raise NotImplementedError

    @abstractmethod
    def remove_edges(self, edges: list[tuple[int, int]]):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_edge_list(cls, filename: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_adjacency_list(cls, filename: str):
        raise NotImplementedError
