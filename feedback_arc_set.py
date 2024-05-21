from copy import copy
import random
from sortedcontainers import SortedList

from fas_graph import FASGraph


def feedback_arc_set(
    graph: FASGraph, use_smartAE: bool = True, reduce: bool = True
) -> list[int]:
    """
    Searches for a minimal Feedback Arc Set of the input graph
    and returns an approximate answer as a list of edges.
    """
    total_fas = []
    if reduce:
        graph.remove_sinks()
        graph.remove_sources()
        graph.remove_runs()
        total_fas.extend(graph.remove_2cycles())

    
    components = graph.iter_strongly_connected_components()
    for component in components:
        # TODO: run in 8 parallel threads (2 per ordering)
        if component.get_num_nodes() >= 2:
            component_nodes = component.get_nodes()
            component_nodes.sort(key=graph.get_out_degree)
            out_asc = compute_fas(
                component,
                component_nodes,
                use_smartAE=use_smartAE,
            )
            component_nodes.reverse()
            out_desc = compute_fas(
                component,
                component_nodes,
                use_smartAE=use_smartAE,
            )
            component_nodes.sort(key=graph.get_in_degree)
            in_asc = compute_fas(
                component,
                component_nodes,
                use_smartAE=use_smartAE,
            )
            component_nodes.reverse()
            in_desc = compute_fas(
                component,
                component_nodes,
                use_smartAE=use_smartAE,
            )

            random.shuffle(component_nodes)
            random_order = compute_fas(component, component_nodes, use_smartAE=use_smartAE)

            scores1, scores2 = compute_scores(component, component_nodes)
            greedy1 = compute_fas(component, scores1, use_smartAE=use_smartAE)
            greddy2 = compute_fas(component, scores2, use_smartAE=use_smartAE)

            total_fas.extend(min(out_asc, out_desc, in_asc, in_desc, key=len))

    return total_fas


def compute_scores(graph:  FASGraph, nodes: list[int]) -> tuple[list[int], list[int]]:
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
            ratio = float('inf')
        if in_degree > 0:
            inv_ratio = out_degree / in_degree
        else:
            inv_ratio = float('inf')
        score2[node] = max(ratio, inv_ratio)

    scores1 = sorted(score1.items(), key=lambda x: x[1], reverse=True)
    scores1 = [t[0] for t in scores1]
    scores2 = sorted(score2.items(), key=lambda item: item[1], reverse=True)
    scores2 = [t[0] for t in scores2]
    """print("score1", scores1)
    print("score2", scores2)"""
    return scores1, scores2



def compute_fas(
    graph: FASGraph,
    ordering: list[int],
    use_smartAE: bool = True,
) -> list[tuple[int, int]]:
    """
    Computes a minimal FAS for the given graph and node ordering.
    """
    # TODO: run in 2 separate threads
    forward_edges, forward_graph = get_forward_edges(graph, ordering)
    backward_edges, backward_graph = get_backward_edges(graph, ordering)

    # reduce the size of the FAS with the smartAE heuristic
    if use_smartAE:
        if len(forward_edges) < len(backward_edges):
            return smart_ae(forward_graph, forward_edges)
        else:
            return smart_ae(backward_graph, backward_edges)
    else:
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

            graph.add_edge(edge)
            if graph.is_acyclic():
                added_edges.append(edge)
                added_count += 1
            else:
                graph.remove_edge(edge)
                eliminated_edges.append(edge)
            i += 1

        for edge in processed_edges:
            fas.remove(edge)

    return eliminated_edges