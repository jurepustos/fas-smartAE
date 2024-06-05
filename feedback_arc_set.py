import random
from collections import defaultdict
from copy import copy

from sortedcontainers import SortedList
from fas_graph import FASGraph


def feedback_arc_set(
    graph: FASGraph,
    use_smartAE: bool = True,
    reduce: bool = True,
    random_ordering: bool = False,
    greedy_orderings: bool = False,
) -> dict[str, tuple[str, str]]:
    """
    Searches for a minimal Feedback Arc Set of the input graph
    and returns an approximate answer as a list of edges.
    """
    fas_instances = defaultdict(list)
    if reduce:
        # graph.remove_sinks()
        # graph.remove_sources()
        graph.remove_runs()
        graph.remove_2cycles()

    components = graph.iter_strongly_connected_components()
    for component in components:
        component_fas_instances = {}
        # TODO: run in 8 parallel threads (2 per ordering)
        if component.get_num_nodes() >= 2:
            component_nodes = component.get_nodes()
            component_nodes.sort(key=component.get_out_degree)
            out_asc = compute_fas(
                component,
                component_nodes,
                "out_asc",
                component_fas_instances,
                use_smartAE=use_smartAE,
            )
            component_nodes.reverse()
            out_desc = compute_fas(
                component,
                component_nodes,
                "out_desc",
                component_fas_instances,
                use_smartAE=use_smartAE,
            )
            component_nodes.sort(key=component.get_in_degree)
            in_asc = compute_fas(
                component,
                component_nodes,
                "in_asc",
                component_fas_instances,
                use_smartAE=use_smartAE,
            )
            component_nodes.reverse()
            in_desc = compute_fas(
                component,
                component_nodes,
                "in_desc",
                component_fas_instances,
                use_smartAE=use_smartAE,
            )

            orderings = [out_asc, out_desc, in_asc, in_desc]

            if random_ordering:
                random.shuffle(component_nodes)
                random_order = compute_fas(
                    component,
                    component_nodes,
                    "random_order",
                    component_fas_instances,
                    use_smartAE=use_smartAE,
                )
                orderings.append(random_order)

            if greedy_orderings:
                scores1, scores2 = compute_scores(component, component_nodes)
                greedy1 = compute_fas(
                    component,
                    scores1,
                    "greedy1",
                    component_fas_instances,
                    use_smartAE=use_smartAE,
                )
                greedy2 = compute_fas(
                    component,
                    scores2,
                    "greedy1",
                    component_fas_instances,
                    use_smartAE=use_smartAE,
                )
                orderings.append(greedy1)
                orderings.append(greedy2)

            for name, instance in component_fas_instances.items():
                for source, target in instance:
                    source_label = component.get_node_labels()[source]
                    target_label = component.get_node_labels()[target]
                    fas_instances[name].append((source_label, target_label))

    return fas_instances


def compute_scores(graph: FASGraph, nodes: list[int]) -> tuple[list[int], list[int]]:
    score1 = {}
    score2 = {}
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

    scores1 = sorted(score1.items(), key=lambda x: x[1], reverse=True)
    scores1 = [t[0] for t in scores1]
    scores2 = sorted(score2.items(), key=lambda item: item[1], reverse=True)
    scores2 = [t[0] for t in scores2]
    return scores1, scores2


def compute_fas(
    graph: FASGraph,
    ordering: list[int],
    ordering_name: str,
    fas_instances: dict[str, list[tuple[int, int]]],
    use_smartAE: bool = True,
) -> list[tuple[int, int]]:
    """
    Computes a minimal FAS for the given graph and node ordering.
    """
    # TODO: run in 2 separate threads
    forward_edges, forward_graph = get_forward_edges(graph, ordering)
    backward_edges, backward_graph = get_backward_edges(graph, ordering)

    fas_instances[ordering_name + "_forward"] = copy(forward_edges)
    fas_instances[ordering_name + "_backward"] = copy(backward_edges)

    # reduce the size of the FAS with the smartAE heuristic
    if use_smartAE:
        forward_edges = smart_ae(forward_graph, forward_edges)
        backward_edges = smart_ae(backward_graph, backward_edges)

    fas_instances[ordering_name + "_forward_smartAE"] = copy(forward_edges)
    fas_instances[ordering_name + "_backward_smartAE"] = copy(backward_edges)
    return min(forward_edges, backward_edges, key=len)


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
            backward_edges.append((node, start_node))
    return backward_edges


def smart_ae(graph: FASGraph, fas: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    For details of this algorithm, consult the paper.
    """
    added_edges = []
    eliminated_edges = []
    while len(fas):
        added_count = 0
        processed_edges = []
        n = len(fas)
        i = 0
        while i + added_count < n:
            edge = fas[i + added_count]
            processed_edges.append(edge)

            if graph.edge_preserves_acyclicity(edge):
                graph.add_edge(edge)
                added_edges.append(edge)
                added_count += 1
            else:
                eliminated_edges.append(edge)
            i += 1

        for edge in processed_edges:
            fas.remove(edge)

    return eliminated_edges
