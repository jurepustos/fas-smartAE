import enum
import random
import sys
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from copy import copy
from queue import SimpleQueue
from typing import Callable, TextIO

from sortedcontainers import SortedList

from fas_builder import FASBuilder, OrderingFASBuilder
from fas_graph import FASGraph, Node


class Mode(enum.Enum):
    FAST = enum.auto()
    NORMAL = enum.auto()
    PARALLEL = enum.auto()

    def __str__(self):
        match self:
            case Mode.FAST:
                return "fast"
            case Mode.NORMAL:
                return "normal"
            case Mode.PARALLEL:
                return "parallel"

    @classmethod
    def from_str(self, string: str):
        match string:
            case "fast":
                return Mode.FAST
            case "normal":
                return Mode.NORMAL
            case "parallel":
                return Mode.PARALLEL
        raise AssertionError("Valid modes are fast, normal and parallel")


class Direction(enum.Enum):
    FORWARD = enum.auto()
    BACKWARD = enum.auto()

    def __str__(self):
        match self:
            case Direction.FORWARD:
                return "forward"
            case Direction.BACKWARD:
                return "backward"


class Ordering(enum.Enum):
    OUT_DEGREE = enum.auto()
    IN_DEGREE = enum.auto()
    RANDOM = enum.auto()
    DEGREE_DIFFERENCE = enum.auto()
    DEGREE_RATIO = enum.auto()

    def _degree_difference_sorter(
        self, graph: FASGraph
    ) -> Callable[[Node], int]:
        def sorter(node: Node) -> int:
            in_degree = graph.get_in_degree(node)
            out_degree = graph.get_out_degree(node)
            return abs(in_degree - out_degree)

        return sorter

    def _degree_ratio_sorter(self, graph: FASGraph) -> Callable[[Node], float]:
        def sorter(node: Node) -> float:
            in_degree = graph.get_in_degree(node)
            out_degree = graph.get_out_degree(node)
            if out_degree == 0 and in_degree == 0:
                return 0
            if out_degree > 0:
                ratio = in_degree / out_degree
            else:
                ratio = float("inf")
            if in_degree > 0:
                inv_ratio = out_degree / in_degree
            else:
                inv_ratio = float("inf")
            return max(ratio, inv_ratio)

        return sorter

    def get_asc_sorter(self, graph: FASGraph) -> Callable[[Node], int | float]:
        match self:
            case Ordering.OUT_DEGREE:
                return graph.get_out_degree
            case Ordering.IN_DEGREE:
                return graph.get_in_degree
            case Ordering.DEGREE_DIFFERENCE:
                return self._degree_difference_sorter(graph)
            case Ordering.DEGREE_RATIO:
                return self._degree_ratio_sorter(graph)
            case Ordering.RANDOM:
                return lambda _: random.random()

    def __str__(self):
        match self:
            case Ordering.OUT_DEGREE:
                return "out"
            case Ordering.IN_DEGREE:
                return "in"
            case Ordering.DEGREE_DIFFERENCE:
                return "deg_diff"
            case Ordering.DEGREE_RATIO:
                return "deg_ratio"
            case Ordering.RANDOM:
                return "random"


DIRECTIONS = [Direction.FORWARD, Direction.BACKWARD]
ORDERINGS = [
    Ordering.IN_DEGREE,
    Ordering.OUT_DEGREE,
    Ordering.DEGREE_DIFFERENCE,
    Ordering.DEGREE_RATIO,
]


def feedback_arc_set(
    graph: FASGraph,
    use_smartAE: bool = True,
    reduce: bool = True,
    mode: Mode = Mode.NORMAL,
    quality: bool = False,
    threads: int = 1,
    log_file: TextIO = sys.stderr,
) -> dict[str, list[tuple[str, str]]]:
    """
    Searches for a minimal Feedback Arc Set of the input graph
    and returns an approximate answer as a list of edges.
    """
    fas_builder = FASBuilder(graph.get_node_labels())
    if reduce:
        fas_builder.add_merged_edges(graph.remove_runs())
        fas_builder.add_fas_edges(graph.remove_2cycles())

    print("Computing strongly connected components", file=log_file)
    components = [
        comp
        for comp in graph.iter_strongly_connected_components()
        if comp.get_num_nodes() >= 2
    ]
    del graph
    print("Finished computing strongly connected components", file=log_file)
    for i, component in enumerate(components):
        num_nodes = component.get_num_nodes()
        num_edges = component.get_num_edges()
        print(
            f"Component {i}: {num_nodes} nodes, {num_edges} edges",
            file=log_file,
        )
        if num_nodes >= 5000:
            # to write to output with any kind of bigger component
            log_file.flush()
        component_fas_builder = FASBuilder(component.get_node_labels())

        match mode:
            case Mode.FAST:
                component_fas_builder = fast_mode(
                    component,
                    use_smartAE=use_smartAE,
                    reduce=reduce,
                    quality=quality,
                    log_file=log_file,
                )
            case Mode.NORMAL:
                component_fas_builder = normal_mode(
                    component,
                    use_smartAE=use_smartAE,
                    reduce=reduce,
                    quality=quality,
                    log_file=log_file,
                )
            case Mode.PARALLEL:
                component_fas_builder = parallel_mode(
                    component,
                    use_smartAE=use_smartAE,
                    reduce=reduce,
                    threads=threads,
                    quality=quality,
                    log_file=log_file,
                )

        fas_builder.merge(component_fas_builder)

    ordering_names = fas_builder.get_ordering_names()
    if len(ordering_names) > 0:
        instances = {
            name: fas_builder.build_fas(name) for name in ordering_names
        }
    else:
        # all components have single edges
        instances = {"edges": fas_builder.fas_edges}

    return instances


def fast_mode(
    graph: FASGraph,
    use_smartAE: bool = True,
    reduce: bool = True,
    quality: bool = False,
    log_file: TextIO = sys.stderr,
) -> FASBuilder:
    fas_builder = FASBuilder(graph.get_node_labels())
    ordering = Ordering.OUT_DEGREE
    nodes = graph.get_nodes()
    nodes.sort(key=ordering.get_asc_sorter(graph), reverse=True)
    for direction in DIRECTIONS:
        name = f"{ordering}_desc_{direction}"
        print(f"\tComputing {name}", file=log_file)
        fas_builder.add_ordering(
            name,
            compute_fas(
                copy(graph),
                nodes,
                direction,
                use_smartAE=use_smartAE,
                quality=quality,
            ),
        )
        print(f"\tFinished {name}", file=log_file)

    return fas_builder


def normal_mode(
    graph: FASGraph,
    use_smartAE: bool = True,
    reduce: bool = True,
    quick: bool = False,
    quality: bool = False,
    log_file: TextIO = sys.stderr,
) -> FASBuilder:
    fas_builder = FASBuilder(graph.get_node_labels())

    for ordering in ORDERINGS:
        nodes = graph.get_nodes()
        nodes.sort(key=ordering.get_asc_sorter(graph))
        for direction in DIRECTIONS:
            name = f"{ordering}_asc_{direction}"
            print(f"\tComputing {name}", file=log_file)
            fas_builder.add_ordering(
                name,
                compute_fas(
                    copy(graph),
                    nodes,
                    direction,
                    use_smartAE=use_smartAE,
                    quality=quality,
                ),
            )
            print(f"\tFinished {name}", file=log_file)

        nodes.reverse()
        for direction in DIRECTIONS:
            name = f"{ordering}_desc_{direction}"
            print(f"\tComputing {name}", file=log_file)
            fas_builder.add_ordering(
                name,
                compute_fas(
                    copy(graph),
                    nodes,
                    direction,
                    use_smartAE=use_smartAE,
                    quality=quality,
                ),
            )
            print(f"\tFinished {name}", file=log_file)

    random.shuffle(nodes)
    for direction in DIRECTIONS:
        name = f"random_{direction}"
        print(f"\tComputing {name}", file=log_file)
        fas_builder.add_ordering(
            name,
            compute_fas(
                copy(graph),
                nodes,
                direction,
                use_smartAE=use_smartAE,
                quality=quality,
            ),
        )
        print(f"\tFinished {name}", file=log_file)
    return fas_builder


def parallel_mode(
    graph: FASGraph,
    use_smartAE: bool = True,
    reduce: bool = True,
    threads: int = 1,
    quality: bool = False,
    log_file: TextIO = sys.stderr,
) -> FASBuilder:
    fas_builder = FASBuilder(graph.get_node_labels())

    def finish_callback(name: str):
        def callback(future: Future[OrderingFASBuilder]):
            fas_builder.add_ordering(f"{name}", future.result())
            print(f"\tFinished {name}", file=log_file)

        return callback

    name = ""
    with ProcessPoolExecutor(
        max_workers=threads,
    ) as executor:
        for ordering in ORDERINGS:
            nodes = graph.get_nodes()
            nodes.sort(key=ordering.get_asc_sorter(graph))
            for direction in DIRECTIONS:
                name = f"{ordering}_asc_{direction}"
                future = executor.submit(
                    compute_fas,
                    copy(graph),
                    nodes,
                    direction,
                    use_smartAE=use_smartAE,
                    quality=quality,
                )
                future.add_done_callback(finish_callback(f"{name}"))

            nodes.reverse()
            for direction in DIRECTIONS:
                name = f"{ordering}_desc_{direction}"
                future = executor.submit(
                    compute_fas,
                    copy(graph),
                    nodes,
                    direction,
                    use_smartAE=use_smartAE,
                    quality=quality,
                )
                future.add_done_callback(finish_callback(f"{name}"))

        random.shuffle(nodes)
        for direction in DIRECTIONS:
            name = f"random_{direction}"
            future = executor.submit(
                compute_fas,
                copy(graph),
                nodes,
                direction,
                use_smartAE=use_smartAE,
                quality=quality,
            )
            future.add_done_callback(finish_callback(f"{name}"))

    return fas_builder


def compute_fas(
    graph: FASGraph,
    ordering: list[int],
    direction: Direction,
    use_smartAE: bool = True,
    quality: bool = False,
) -> OrderingFASBuilder:
    """
    Computes a minimal FAS for the given graph and node ordering.
    """
    builder = OrderingFASBuilder()

    edges = remove_direction_edges(
        graph, ordering, direction=direction, quality=quality
    )

    builder.add_removed_edges(edges)

    # reduce the size of the FAS with the smartAE heuristic
    if use_smartAE:
        builder.add_smartAE_restored(smart_ae(graph, edges))

    return builder


def remove_direction_edges(
    graph: FASGraph,
    ordering: list[int],
    direction: Direction,
    quality: bool = False,
) -> list[tuple[int, int]]:
    direction_edges = []
    for i in range(len(ordering)):
        edges = get_direction_edges_from(graph, ordering, i, direction)
        if quality:
            for source, target in edges:
                edges_to_remove = []
                if not graph.edge_between_components(source, target):
                    edges_to_remove.append((source, target))
                graph.remove_edges(edges_to_remove)
                direction_edges.extend(edges_to_remove)
            if graph.is_acyclic():
                break
        else:
            graph.remove_edges(edges)
            direction_edges.extend(edges)
            if graph.is_acyclic():
                break

    return direction_edges


def get_direction_edges_from(
    graph: FASGraph, ordering: list[int], start_index: int, direction: Direction
) -> list[tuple[int, int]]:
    backward_edges = []
    start_node = ordering[start_index]
    match direction:
        case Direction.FORWARD:
            start_neighbors = SortedList(graph.iter_out_neighbors(start_node))
        case Direction.BACKWARD:
            start_neighbors = SortedList(graph.iter_in_neighbors(start_node))
    for node_index in range(start_index + 1, len(ordering)):
        node = ordering[node_index]
        if node in start_neighbors:
            match direction:
                case Direction.FORWARD:
                    for _ in range(graph.get_edge_weight(start_node, node)):
                        backward_edges.append((start_node, node))
                case Direction.BACKWARD:
                    for _ in range(graph.get_edge_weight(node, start_node)):
                        backward_edges.append((node, start_node))
    return backward_edges


def smart_ae(
    graph: FASGraph, fas: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """
    For details of this algorithm, consult the paper by Cavallaro et al.
    """
    added_edges = []
    while len(fas):
        added_count = 0
        processed_edge_indices: list[int] = []
        n = len(fas)
        i = 0
        while i + added_count < n:
            edge = fas[i + added_count]
            processed_edge_indices.append(i + added_count)

            graph.add_edge(*edge)
            if graph.is_acyclic():
                added_edges.append(edge)
                added_count += 1
            else:
                graph.remove_edge(*edge)
            i += 1

        deleted_count = 0
        for edge_index in processed_edge_indices:
            del fas[edge_index - deleted_count]
            deleted_count += 1

    return added_edges
