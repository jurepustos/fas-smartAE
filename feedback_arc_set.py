from typing import TypeVar, Tuple, Dict
from copy import copy

from sortedcontainers import SortedList

from .fas_graph import FASGraph

Edge = TypeVar('Edge')
Node = TypeVar('Node')
Graph = TypeVar('Graph', FASGraph[Node, Edge])       #type: ignore
#TODO: check TypeVar error: TypeVar constraint type cannot be genericPylance


def feedback_arc_set(graph: Graph, use_smartAE: bool = True, reduce: bool = True,
                     stopping_condition: str = 'acyclic') -> list[Edge]:
    """
    Searches for a minimal Feedback Arc Set of the input graph
    and returns an approximate answer as a list of edges.
    """
    if reduce:
        graph.remove_sinks()
        graph.remove_sources()

    total_fas = []
    components = graph.iter_strongly_connected_components()
    for component in components:
        # TODO: run in 8 parallel threads (2 per ordering)
        if len(component) >= 2:
            component_nodes = component  #TODO: tle je vrjetn misln d dobimo nodes od component ne grafa

            component_nodes.sort(key=lambda node: graph.get_out_degree(node))  # changed lambda to key
            out_asc = compute_fas(component, component_nodes,
                                  use_smartAE=use_smartAE,
                                  stopping_condition=stopping_condition)
            component_nodes.reverse()
            out_desc = compute_fas(component, component_nodes,
                                   use_smartAE=use_smartAE,
                                   stopping_condition=stopping_condition)
            component_nodes.sort(key=lambda node: graph.get_in_degree(node))  # changed lambda to key
            in_asc = compute_fas(component, component_nodes,
                                 use_smartAE=use_smartAE,
                                 stopping_condition=stopping_condition)
            component_nodes.reverse()
            in_desc = compute_fas(component, component_nodes,
                                  use_smartAE=use_smartAE,
                                  stopping_condition=stopping_condition)
            
            score1, score2 = compute_scores(graph, component_nodes)
            greedy1 = compute_fas(component, score1, use_smartAE=use_smartAE, stopping_condition=stopping_condition)
            greedy2 = compute_fas(component, score2, use_smartAE=use_smartAE, stopping_condition=stopping_condition)

            total_fas.append(min(out_asc, out_desc, in_asc, in_desc, greedy1, greedy2, key=len))

    return total_fas


def compute_scores(graph: Graph, component_nodes) -> Tuple[Dict[int, int], Dict[int, float]]:
    score1 = {}
    score2 = {}
    for node in component_nodes:
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
    return scores1, scores2


def compute_fas(graph: Graph, ordering: list[Node], use_smartAE: bool = True,
                stopping_condition: str = 'acyclic') -> list[Edge]:
    """
    Computes a minimal FAS for the given graph and node ordering.
    """
    # TODO: run in 2 separate threads

    forward_edges, forward_graph = get_forward_edges(graph, ordering,
                                                     stopping_condition=stopping_condition)
    backward_edges, backward_graph = get_backward_edges(graph, ordering,
                                                        stopping_condition=stopping_condition)

    # reduce the size of the FAS with the smartAE heuristic
    if use_smartAE:
        if len(forward_edges) < len(backward_edges):
            return smart_ae(forward_graph, forward_edges)
        else:
            return smart_ae(backward_graph, backward_edges)
    else:
        return min(forward_edges, backward_edges, key=len)


def get_forward_edges(graph: Graph, ordering: list[int],
                      stopping_condition: str = 'acyclic') \
        -> tuple[list[Edge], Graph]:
    forward_edges = []
    forward_graph = copy(graph)
    for i in range(len(ordering)):
        edges = get_forward_edges_from(forward_graph, ordering, i)  # Corrected call
        forward_edges.extend(edges)
        forward_graph.remove_edges(forward_edges)
        # TODO: instead of checking acyclicity, we could use SCCs instead
        if forward_graph.is_acyclic(ordering):
            break

    return forward_edges, forward_graph


def get_forward_edges_from(graph: Graph, ordering: list[Node],
                           start_index: int) -> list[Edge]:
    forward_edges = []
    start_node = ordering[start_index]
    start_neighbors = SortedList(graph.iter_out_neighbors(start_node))
    for node_index in range(start_index+1, len(ordering)):
        node = ordering[node_index]
        if node in start_neighbors:
            forward_edges.append(graph.get_edge(start_node, node))
    return forward_edges


def get_backward_edges(graph: Graph, ordering: list[int],
                       stopping_condition: str = 'acyclic') \
        -> tuple[list[Edge], Graph]:
    backward_edges = []
    backward_graph = copy(graph)
    for i in range(len(ordering)):
        edges = get_backward_edges_from(backward_graph, ordering, i)  # Corrected call
        backward_edges.extend(edges)
        backward_graph.remove_edges(backward_edges)
        # TODO: instead of checking acyclicity, we could use SCCs instead
        if backward_graph.is_acyclic(ordering):
            break

    return backward_edges, backward_graph


def get_backward_edges_from(graph: Graph, ordering: list[Node],
                            start_index: int) -> list[Edge]:
    backward_edges = []
    start_node = ordering[start_index]
    start_neighbors = SortedList(graph.iter_in_neighbors(start_node))
    for node_index in range(start_index+1, len(ordering)):
        node = ordering[node_index]
        if node in start_neighbors:
            backward_edges.append(graph.get_edge(node, start_node))
    return backward_edges


def smart_ae(graph: Graph, fas: list[Edge]) -> list[Edge]:
    """
    For details of this algorithm, consult the paper.
    """
    added_edges = []
    eliminated_edges = []
    fas_count = len(fas)
    while fas_count > 0:
        added_count = 0
        for i in range(1, len(fas) + 1):
            edge = fas[i+added_count]
            # we don't actually remove the edge from `fas`,
            # since it's not really needed
            fas_count -= 1

            graph.add_edge(edge)
            if graph.is_acyclic():
                added_edges.append(edge)
                added_count += 1
            else:
                graph.remove_edge(edge)
                eliminated_edges.append(edge)
    return eliminated_edges
