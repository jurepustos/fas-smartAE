import random
import sys
from copy import copy

from sortedcontainers import SortedList

from fas_builder import FASBuilder, OrderingFASBuilder
from fas_graph import FASGraph, Node
from typing import TextIO

DIRECTIONS = ["forward", "backward"]


def feedback_arc_set(
    graph: FASGraph,
    use_smartAE: bool = True,
    reduce: bool = True,
    random_ordering: bool = True,
    greedy_orderings: bool = True,
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

    for i, component in enumerate(graph.iter_strongly_connected_components()):
        print(
            f"Component {i}: {component.get_num_nodes()} nodes, {component.get_num_edges()} edges",
            file=log_file,
        )
        component_fas_builder = FASBuilder(component.get_node_labels())

        if component.get_num_nodes() < 2:
            continue

        component_nodes = component.get_nodes()
        component_nodes.sort(key=component.get_out_degree)
        for direction in DIRECTIONS:
            print(f"\tComputing out_asc_{direction}", file=log_file)
            component_fas_builder.add_ordering(
                f"out_asc_{direction}",
                compute_fas(
                    component,
                    component_nodes,
                    direction,
                    use_smartAE=use_smartAE,
                ),
            )
            print(f"\tFinished out_asc_{direction}", file=log_file)

        component_nodes.reverse()
        for direction in DIRECTIONS:
            print(f"\tComputing out_desc_{direction}", file=log_file)
            component_fas_builder.add_ordering(
                f"out_desc_{direction}",
                compute_fas(
                    component,
                    component_nodes,
                    direction,
                    use_smartAE=use_smartAE,
                ),
            )
            print(f"\tFinished out_desc_{direction}", file=log_file)

        component_nodes = component.get_nodes()
        component_nodes.sort(key=component.get_out_degree)
        for direction in DIRECTIONS:
            print(f"\tComputing in_asc_{direction}", file=log_file)
            component_fas_builder.add_ordering(
                f"in_asc_{direction}",
                compute_fas(
                    component,
                    component_nodes,
                    direction,
                    use_smartAE=use_smartAE,
                ),
            )
            print(f"\tFinished in_asc_{direction}", file=log_file)

        component_nodes.reverse()
        for direction in DIRECTIONS:
            print(f"\tComputing in_desc_{direction}", file=log_file)
            component_fas_builder.add_ordering(
                f"in_desc_{direction}",
                compute_fas(
                    component,
                    component_nodes,
                    direction,
                    use_smartAE=use_smartAE,
                ),
            )
            print(f"\tFinished in_desc_{direction}", file=log_file)

        if random_ordering:
            random.shuffle(component_nodes)
            for direction in DIRECTIONS:
                print(f"\tComputing random_{direction}", file=log_file)
                component_fas_builder.add_ordering(
                    f"random_{direction}",
                    compute_fas(
                        component,
                        component_nodes,
                        direction,
                        use_smartAE=use_smartAE,
                    ),
                )
                print(f"\tFinished random_{direction}", file=log_file)

        if greedy_orderings:
            scores1, scores2 = compute_scores(component, component_nodes)

            for direction in DIRECTIONS:
                print(f"\tComputing greedy1_{direction}", file=log_file)
                component_fas_builder.add_ordering(
                    f"greedy1_{direction}",
                    compute_fas(
                        component,
                        scores1,
                        direction,
                        use_smartAE=use_smartAE,
                    ),
                )
                print(f"\tFinished greedy1_{direction}", file=log_file)

            for direction in DIRECTIONS:
                print(f"\tComputing greedy2_{direction}", file=log_file)
                component_fas_builder.add_ordering(
                    f"greedy2_{direction}",
                    compute_fas(
                        component,
                        scores1,
                        direction,
                        use_smartAE=use_smartAE,
                    ),
                )
                print(f"\tFinished greedy2_{direction}", file=log_file)

        fas_builder.merge(component_fas_builder)

    if len(fas_builder.orderings) > 0:
        instances = {
            name: fas_builder.build_fas(name)
            for name, ordering in fas_builder.orderings.items()
        }
    else:
        # all components have single edges
        instances = {"edges": fas_builder.fas_edges}

    return instances


def compute_scores(
    graph: FASGraph, nodes: list[int]
) -> tuple[list[int], list[int]]:
    score1: dict[Node, float] = {}
    score2: dict[Node, float] = {}
    for node in nodes:
        in_degree = graph.get_in_degree(node)
        out_degree = graph.get_out_degree(node)
        score1[node] = abs(in_degree - out_degree)
        if out_degree == 0 and in_degree == 0:
            score2[node] = 0
            continue
        if out_degree > 0:
            ratio = in_degree / out_degree
        else:
            ratio = float("inf")
        if in_degree > 0:
            inv_ratio = out_degree / in_degree
        else:
            inv_ratio = float("inf")
        score2[node] = max(ratio, inv_ratio)

    scores1 = [
        node
        for node, _score in sorted(
            score1.items(), key=lambda x: x[1], reverse=False
        )
    ]
    scores2 = [
        node
        for node, _score in sorted(
            score2.items(), key=lambda item: item[1], reverse=False
        )
    ]
    return scores1, scores2


def compute_fas(
    graph: FASGraph,
    ordering: list[int],
    direction: str,
    use_smartAE: bool = True,
) -> OrderingFASBuilder:
    """
    Computes a minimal FAS for the given graph and node ordering.
    """
    builder = OrderingFASBuilder()

    assert direction in ["forward", "backward"]
    if direction == "forward":
        edges, reduced_graph = get_forward_edges(graph, ordering)
    elif direction == "backward":
        edges, reduced_graph = get_backward_edges(graph, ordering)

    builder.add_removed_edges(edges)

    # reduce the size of the FAS with the smartAE heuristic
    if use_smartAE:
        builder.add_smartAE_restored(smart_ae(reduced_graph, edges))

    return builder


def get_forward_edges(
    graph: FASGraph, ordering: list[int]
) -> tuple[list[tuple[int, int]], FASGraph]:
    forward_edges = []
    forward_graph = copy(graph)
    for i in range(len(ordering)):
        edges = get_forward_edges_from(graph, ordering, i)
        forward_edges.extend(edges)
        forward_graph.remove_edges(edges)
        if forward_graph.is_acyclic():
            break

    return forward_edges, forward_graph


def get_forward_edges_from(
    graph: FASGraph, ordering: list[int], start_index: int
) -> list[tuple[int, int]]:
    forward_edges = []
    start_node = ordering[start_index]
    start_neighbors = SortedList(graph.iter_out_neighbors(start_node))
    for node_index in range(start_index + 1, len(ordering)):
        node = ordering[node_index]
        if node in start_neighbors:
            for _ in range(graph.get_edge_weight(start_node, node)):
                forward_edges.append((start_node, node))
    return forward_edges


def get_backward_edges(
    graph: FASGraph, ordering: list[int]
) -> tuple[list[tuple[int, int]], FASGraph]:
    backward_edges = []
    backward_graph = copy(graph)
    for i in range(len(ordering)):
        edges = get_backward_edges_from(graph, ordering, i)
        backward_edges.extend(edges)
        backward_graph.remove_edges(edges)
        if backward_graph.is_acyclic():
            break

    return backward_edges, backward_graph


def get_backward_edges_from(
    graph: FASGraph, ordering: list[int], start_index: int
) -> list[tuple[int, int]]:
    backward_edges = []
    start_node = ordering[start_index]
    start_neighbors = SortedList(graph.iter_in_neighbors(start_node))
    for node_index in range(start_index + 1, len(ordering)):
        node = ordering[node_index]
        if node in start_neighbors:
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

        for edge_index in processed_edge_indices:
            del fas[edge_index]

    return added_edges
