from typing import TypeVar
from .fas_graph import FASGraph

Edge = TypeVar('Edge')
Node = TypeVar('Node')
Graph = TypeVar('Graph', FASGraph[Node, Edge])


def feedback_arc_set(graph: Graph) -> list[Edge]:
    """
    Searches for a minimal Feedback Arc Set of the input graph
    and returns it as a list of edges.
    """
    components = graph.iter_strongly_connected_components()
    total_fas = []
    for component in components:
        # TODO: run in 8 parallel threads (2 per ordering)
        if len(component) >= 2:
            component.remove_sinks()
            component.remove_sources()
            component_nodes = graph.get_nodes()
            component_nodes.sort(lambda node: graph.get_out_degree(node))
            out_asc = compute_fas(component, component_nodes)
            component_nodes.reverse()
            out_desc = compute_fas(graph, component)
            component_nodes.sort(lambda node: graph.get_in_degree(node))
            in_asc = compute_fas(graph, component)
            component_nodes.reverse()
            in_desc = compute_fas(graph, component)
            total_fas.append(min(out_asc, out_desc, in_asc, in_desc, key=len))


def compute_fas(graph: Graph, ordering: list[Node]) -> list[Edge]:
    """
    Computes a minimal FAS for the given graph and node ordering.
    """
    # TODO: run in 2 separate threads

    forward_edges, forward_graph = graph.get_forward_edges(ordering)
    backward_edges, backward_graph = graph.get_backward_edges(ordering)

    # reduce the size of the FAS with the smartAE heuristic
    if len(forward_edges) < len(backward_edges):
        return smart_ae(forward_graph, forward_edges)
    else:
        return smart_ae(backward_graph, backward_edges)


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

            graph.addEdge(edge)
            if graph.is_acyclic():
                added_edges.append(edge)
                added_count += 1
            else:
                graph.removeEdge()
                eliminated_edges.append(edge)
    return eliminated_edges
