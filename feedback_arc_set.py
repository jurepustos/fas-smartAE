import enum
import random
import sys
import time
from concurrent.futures import Executor, Future
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from copy import copy, deepcopy
from functools import partial
from typing import Callable, Iterator, TextIO

from sortedcontainers import SortedList

from fas_builder import FASBuilder, OrderingFASBuilder
from fas_graph import FASGraph, Node


class Mode(enum.Enum):
    FAST = enum.auto()
    QUALITY = enum.auto()

    def __str__(self):
        match self:
            case Mode.FAST:
                return "fast"
            case Mode.QUALITY:
                return "quality"

    @classmethod
    def from_str(self, string: str):
        match string:
            case "fast":
                return Mode.FAST
            case "quality":
                return Mode.QUALITY
        raise AssertionError("Valid modes are fast and quality")


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

    def _degree_difference_sorter(self, graph: FASGraph) -> Callable[[Node], int]:
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


def print_output(
    graph: FASGraph,
    *values: str,
    sep: str | None = None,
    end: str | None = None,
    file: TextIO | None = None,
    flush: bool = False,
):
    if graph.get_num_nodes() >= 1000:
        print(*values, sep=sep, end=end, file=file, flush=flush)


def reduce_graph(graph: FASGraph) -> FASBuilder:
    fas_builder = FASBuilder(graph.get_node_labels())
    fas_builder.add_fas_edges(graph.remove_self_loops())
    num_nodes = None
    num_edges = None
    while graph.get_num_nodes() != num_nodes and graph.get_num_edges() != num_edges:
        num_nodes = graph.get_num_nodes()
        num_edges = graph.get_num_edges()
        fas_builder.add_merged_edges(graph.remove_runs())
        fas_builder.add_fas_edges(graph.remove_2cycles())
    return fas_builder


def feedback_arc_set(
    graph: FASGraph,
    use_smartAE: bool = True,
    reduce: bool = True,
    mode: Mode = Mode.FAST,
    threads: int = 1,
    log_file: TextIO = sys.stderr,
) -> dict[str, list[tuple[str, str]]]:
    """
    Searches for a minimal Feedback Arc Set of the input graph
    and returns an approximate answer as a list of edges.
    """
    fas_builder = FASBuilder(graph.get_node_labels())
    if reduce:
        fas_builder.merge(reduce_graph(graph))

    components_iter = (
        comp for comp in graph.iter_components() if comp.get_num_nodes() >= 2
    )
    if threads > 1:
        component_builders = parallel_components(
            components_iter,
            threads,
            use_smartAE=use_smartAE,
            log_file=log_file,
            mode=mode,
        )
        for builder in component_builders:
            fas_builder.merge(builder)
    else:
        component_builders = sequential_components(
            components_iter,
            use_smartAE=use_smartAE,
            log_file=log_file,
            mode=mode,
        )
        for builder in component_builders:
            fas_builder.merge(builder)

    ordering_names = fas_builder.get_ordering_names()
    if len(ordering_names) > 0:
        instances = {name: fas_builder.build_fas(name) for name in ordering_names}
    else:
        # all components have single edges
        instances = {"edges": fas_builder.fas_edges}

    return instances


def sequential_components(
    it_components: Iterator[FASGraph],
    use_smartAE: bool,
    log_file: TextIO,
    mode: Mode,
) -> list[FASBuilder]:
    comp_start_time = time.time()
    print("Computing components", file=log_file)
    components = list(it_components)
    print("Finished computing components", file=log_file)
    comp_end_time = time.time()
    print(
        f"Component calculation time: {comp_end_time - comp_start_time} s",
        file=log_file,
        flush=True,
    )

    component_builders: list[FASBuilder] = []
    for i, component in enumerate(components):
        num_nodes = component.get_num_nodes()
        num_edges = component.get_num_edges()
        print_output(
            component,
            f"Component {i}: {num_nodes} nodes, {num_edges} edges",
            file=log_file,
            flush=True,
        )

        component_builders.append(
            sequential_orderings(
                component,
                use_smartAE=use_smartAE,
                log_file=log_file,
                mode=mode,
            )
        )
        if num_nodes >= 1000:
            # to write to output with any kind of bigger component
            log_file.flush()

    return component_builders


def sequential_orderings(
    graph: FASGraph,
    use_smartAE: bool,
    log_file: TextIO,
    mode: Mode,
) -> FASBuilder:
    fas_builder = FASBuilder(graph.get_node_labels())

    nodes = graph.get_nodes()
    for ordering in ORDERINGS:
        nodes.sort(key=ordering.get_asc_sorter(graph))
        for direction in DIRECTIONS:
            name = f"{ordering}_asc_{direction}"
            print_output(graph, f"\tComputing {name}", file=log_file, flush=True)
            fas_builder.add_ordering(
                name,
                compute_fas(
                    copy(graph),
                    copy(nodes),
                    direction,
                    use_smartAE=use_smartAE,
                    mode=mode,
                ),
            )
            print_output(graph, f"\tFinished {name}", file=log_file)

        nodes.reverse()
        for direction in DIRECTIONS:
            name = f"{ordering}_desc_{direction}"
            print_output(graph, f"\tComputing {name}", file=log_file, flush=True)
            fas_builder.add_ordering(
                name,
                compute_fas(
                    copy(graph),
                    copy(nodes),
                    direction,
                    use_smartAE=use_smartAE,
                    mode=mode,
                ),
            )
            print_output(graph, f"\tFinished {name}", file=log_file)

    random.shuffle(nodes)
    for direction in DIRECTIONS:
        name = f"random_{direction}"
        print_output(graph, f"\tComputing {name}", file=log_file, flush=True)
        fas_builder.add_ordering(
            name,
            compute_fas(
                copy(graph),
                copy(nodes),
                direction,
                use_smartAE=use_smartAE,
                mode=mode,
            ),
        )
        print_output(graph, f"\tFinished {name}", file=log_file)
    return fas_builder


def parallel_components(
    it_components: Iterator[FASGraph],
    threads: int,
    use_smartAE: bool,
    log_file: TextIO,
    mode: Mode,
) -> list[FASBuilder]:
    component_builders: list[FASBuilder] = []
    with ProcessPoolExecutor(max_workers=threads) as executor:
        for i, component in enumerate(it_components):
            num_nodes = component.get_num_nodes()
            num_edges = component.get_num_edges()
            print_output(
                component,
                f"Component {i}: {num_nodes} nodes, {num_edges} edges",
                file=log_file,
                flush=True,
            )

            if num_edges >= 1000:
                component_builders.append(
                    parallel_orderings(
                        copy(component),
                        executor,
                        use_smartAE=use_smartAE,
                        log_file=log_file,
                        mode=mode,
                    )
                )
            else:
                component_builders.append(
                    sequential_orderings(
                        component,
                        use_smartAE=use_smartAE,
                        log_file=log_file,
                        mode=mode,
                    )
                )

    return component_builders


def finish_callback(
    name: str, fas_builder: FASBuilder, future: Future[OrderingFASBuilder]
):
    fas_builder.add_ordering(f"{name}", future.result())


def parallel_orderings(
    graph: FASGraph,
    executor: Executor,
    use_smartAE: bool,
    log_file: TextIO,
    mode: Mode,
) -> FASBuilder:
    fas_builder = FASBuilder(graph.get_node_labels())

    name = ""
    nodes = graph.get_nodes()
    for ordering in ORDERINGS:
        nodes.sort(key=ordering.get_asc_sorter(graph))
        for direction in DIRECTIONS:
            name = f"{ordering}_asc_{direction}"
            future = executor.submit(
                compute_fas,
                copy(graph),
                copy(nodes),
                direction,
                use_smartAE=use_smartAE,
                mode=mode,
            )
            future.add_done_callback(partial(finish_callback, f"{name}", fas_builder))

        nodes.reverse()
        for direction in DIRECTIONS:
            name = f"{ordering}_desc_{direction}"
            future = executor.submit(
                compute_fas,
                copy(graph),
                copy(nodes),
                direction,
                use_smartAE=use_smartAE,
                mode=mode,
            )
            future.add_done_callback(partial(finish_callback, f"{name}", fas_builder))

    random.shuffle(nodes)
    for direction in DIRECTIONS:
        name = f"random_{direction}"
        future = executor.submit(
            compute_fas,
            copy(graph),
            copy(nodes),
            direction,
            use_smartAE=use_smartAE,
            mode=mode,
        )
        future.add_done_callback(partial(finish_callback, f"{name}", fas_builder))

    return fas_builder


def compute_fas(
    graph: FASGraph,
    ordering: list[int],
    direction: Direction,
    use_smartAE: bool,
    mode: Mode,
) -> OrderingFASBuilder:
    """
    Computes a minimal FAS for the given graph and node ordering.
    """
    builder = OrderingFASBuilder()

    edges = remove_direction_edges(graph, ordering, direction=direction, mode=mode)

    builder.add_removed_edges(edges)

    # reduce the size of the FAS with the smartAE heuristic
    if use_smartAE:
        builder.add_smartAE_restored(smart_ae(graph, edges))

    return builder


def remove_direction_edges(
    graph: FASGraph,
    ordering: list[int],
    direction: Direction,
    mode: Mode,
) -> list[tuple[int, int]]:
    direction_edges = []
    for i in range(len(ordering)):
        edges = get_direction_edges_from(graph, ordering, i, direction)
        match mode:
            case Mode.QUALITY:
                edges_to_remove = []
                for source, target in edges:
                    if not graph.edge_between_components(source, target):
                        edges_to_remove.append((source, target))
                graph.remove_edges(edges_to_remove)
                direction_edges.extend(edges_to_remove)
                if graph.is_acyclic():
                    break
            case Mode.FAST:
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


def smart_ae(graph: FASGraph, fas: list[tuple[int, int]]) -> list[tuple[int, int]]:
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
