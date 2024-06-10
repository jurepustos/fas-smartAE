import random
from copy import copy

from sortedcontainers import SortedList

from fas_builder import FASBuilder, OrderingFASBuilder
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
    fas_builder = FASBuilder(graph.get_node_labels())
    reduction_merged_edges = {}
    reduction_fas_edges = [(n, n) for n in graph.get_self_loops()]
    if reduce:
        reduction_merged_edges.update(graph.remove_runs())
        acyclicFlag, fas = graph.remove_2cycles()
        reduction_fas_edges.extend(fas)
        if acyclicFlag:
            fas_builder.add_fas_edges(reduction_fas_edges)
            fas_builder.add_merged_edges(reduction_merged_edges)
            fas_builder.ordering("just2cycle"),
            instances = {
                f"{name}": ordering.forward.build_fas()
                for name, ordering in fas_builder.orderings.items()
            }
            return instances

    fas_builder.add_fas_edges(reduction_fas_edges)
    fas_builder.add_merged_edges(reduction_merged_edges)
    for component in graph.iter_strongly_connected_components():
        component_fas_builder = FASBuilder(component.get_node_labels())

        if component.get_num_nodes() < 2:
            fas_builder.merge(component_fas_builder)
            continue

        component_nodes = component.get_nodes()
        component_nodes.sort(key=component.get_out_degree)
        compute_fas(
            component,
            component_nodes,
            component_fas_builder.ordering("out_asc"),
            use_smartAE=use_smartAE,
        )

        component_nodes.reverse()
        compute_fas(
            component,
            component_nodes,
            component_fas_builder.ordering("out_desc"),
            use_smartAE=use_smartAE,
        )

        component_nodes.sort(key=component.get_in_degree)
        compute_fas(
            component,
            component_nodes,
            component_fas_builder.ordering("in_asc"),
            use_smartAE=use_smartAE,
        )

        component_nodes.reverse()
        compute_fas(
            component,
            component_nodes,
            component_fas_builder.ordering("in_desc"),
            use_smartAE=use_smartAE,
        )

        if random_ordering:
            random.shuffle(component_nodes)
            compute_fas(
                component,
                component_nodes,
                component_fas_builder.ordering("random"),
                use_smartAE=use_smartAE,
            )

        if greedy_orderings:
            scores1, scores2 = compute_scores(component, component_nodes)
            compute_fas(
                component,
                scores1,
                component_fas_builder.ordering("greedy1"),
                use_smartAE=use_smartAE,
            )
            compute_fas(
                component,
                scores2,
                component_fas_builder.ordering("greedy2"),
                use_smartAE=use_smartAE,
            )

        fas_builder.merge(component_fas_builder)

    instances = {
        f"{name}_forward": ordering.forward.build_fas()
        for name, ordering in fas_builder.orderings.items()
    }
    instances.update(
        {
            f"{name}_backward": ordering.backward.build_fas()
            for name, ordering in fas_builder.orderings.items()
        }
    )
    return instances


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

    scores1 = sorted(score1.items(), key=lambda x: x[1], reverse=False)
    scores1 = [t[0] for t in scores1]
    scores2 = sorted(score2.items(), key=lambda item: item[1], reverse=False)
    scores2 = [t[0] for t in scores2]
    return scores1, scores2


def compute_fas(
    graph: FASGraph,
    ordering: list[int],
    builder: OrderingFASBuilder,
    use_smartAE: bool = True,
) -> OrderingFASBuilder:
    """
    Computes a minimal FAS for the given graph and node ordering.
    """
    forward_edges, forward_graph = get_forward_edges(graph, ordering)
    backward_edges, backward_graph = get_backward_edges(graph, ordering)

    builder.forward.add_removed_edges(forward_edges)
    builder.backward.add_removed_edges(backward_edges)

    # reduce the size of the FAS with the smartAE heuristic
    if use_smartAE:
        builder.forward.add_smartAE_restored(smart_ae(forward_graph, forward_edges))
        builder.backward.add_smartAE_restored(smart_ae(backward_graph, backward_edges))

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


def smart_ae(graph: FASGraph, fas: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    For details of this algorithm, consult the paper.
    """
    added_edges = []
    while len(fas):
        added_count = 0
        processed_edges = []
        n = len(fas)
        i = 0
        while i + added_count < n:
            edge = fas[i + added_count]
            processed_edges.append(edge)

            graph.add_edge(*edge)
            if graph.is_acyclic():
                added_edges.append(edge)
                added_count += 1
            else:
                graph.remove_edge(*edge)
            i += 1

        for edge in processed_edges:
            fas.remove(edge)

    return added_edges
