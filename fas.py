from networkit.graph import Graph
from networkit.graphio import EdgeListReader
from networkit.components import StronglyConnectedComponents
from networkit.graphtools import GraphTools
from networkit.traversal import Traversal
from typing import Iterator
from copy import copy
import sys


def graph_fas(graph: Graph) -> list[int]:
    remove_unneeded_vertices(graph)
    components = get_strongly_connected_components(graph)
    total_fas = []
    for component in components:
        if len(component) >= 2:
            component_nodes = list(graph.iterNodes())
            component_nodes.sort(lambda node: graph.degreeOut(node))
            out_asc = compute_fas(component, component_nodes)
            component_nodes.reverse()
            out_desc = compute_fas(graph, component)
            component_nodes.sort(lambda node: graph.degreeIn(node))
            in_asc = compute_fas(graph, component)
            component_nodes.reverse()
            in_desc = compute_fas(graph, component)
            total_fas.append(min(out_asc, out_desc, in_asc, in_desc, key=len))


def load_graph(filename: str) -> Graph:
    """
    Load the graph from an edge-list representation.
    The resulting graph does not have isolated vertices.
    """
    reader = EdgeListReader(directed=True, separator='\t')
    return reader.read(filename)


def remove_unneeded_vertices(graph: Graph):
    """
    Removes vertices, which can be removed as they are guaranteed
    to not appear in a minimum FAS.
    Currently removes sinks and sources. """
    sinks = [node for node in graph.iterNodes() if graph.degreeOut(node) == 0]
    graph.removeEdges(sinks)
    sources = [node for node in graph.iterNodes() if graph.degreeIn(node) == 0]
    graph.removeEdges(sources)


def get_strongly_connected_components(graph: Graph) -> Iterator[Graph]:
    """
    Returns a list of strongly connected components,
    represented as lists of nodes.
    """
    cc = StronglyConnectedComponents(graph)
    cc.run()
    for component_nodes in cc.getComponents():
        yield GraphTools.subgraphFromNodes(graph, component_nodes)


def compute_fas(graph: Graph, ordering: list[int]) -> list[int]:
    """
    Computes a minimal FAS for the given nodes.
    """
    forward_edges = []
    forward_graph = copy(graph)

    # try removing forward edges
    for i in range(len(ordering)):
        edges = get_forward_edges_from(forward_graph, ordering, i)
        forward_edges.extend(edges)
        forward_graph.removeEdges(forward_edges)
        if is_acyclic(forward_graph, ordering):
            break

    backward_edges = []
    backward_graph = copy(graph)

    # try removing backward edges
    for i in range(len(ordering)):
        edges = get_backward_edges_from(backward_graph, ordering, i)
        backward_edges.extend(edges)
        backward_graph.removeEdges(backward_edges)
        if is_acyclic(backward_graph, ordering):
            break
    graph.addEdges(backward_edges)

    # reduce the size of the FAS with the smartAE heuristic
    if len(forward_edges) < len(backward_edges):
        return smart_ae(forward_graph, forward_edges)
    else:
        return smart_ae(backward_graph, backward_edges)


def get_forward_edges_from(graph: Graph, ordering: list[int],
                           start: int) -> list[int]:
    forward_edges = []
    start_neighbors = set(graph.iterOutNeighbors(ordering[start]))
    for node in range(start+1, len(ordering)):
        if ordering[node] in start_neighbors:
            forward_edges.append(node)
    return forward_edges


def get_backward_edges_from(graph: Graph, ordering: list[int],
                            start: int) -> list[int]:
    backward_edges = []
    start_neighbors = set(graph.iterInNeighbors(ordering[start]))
    for node in range(start+1, len(ordering)):
        if ordering[node] in start_neighbors:
            backward_edges.append(node)
    return backward_edges


class CycleDetectedError(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "A cycle was detected"
        super().__init__(message)


def is_acyclic(graph: Graph) -> bool:
    try:
        Traversal.DFSfrom(graph, GraphTools.randomNode(graph),
                          acyclic_dfs_callback(graph))
        return True
    except CycleDetectedError:
        return False


def acyclic_dfs_callback(graph: Graph):
    active_path = {node: False for node in graph.iterNodes()}
    prev_node = None

    def callback_inner(node):
        nonlocal prev_node
        # remove the previously visited node from the active path
        if prev_node is not None and active_path[prev_node]:
            active_path[prev_node] = False

        # check if a neighbor is in the active path
        for neighbor in graph.iterOutNeighbors():
            if active_path[neighbor]:
                raise CycleDetectedError

        # add current node to the active path
        active_path[node] = True
        # set the previous node to the current node, so that it is removed
        # from the active path when backtracking
        prev_node = node

    return callback_inner


def smart_ae(graph: Graph, fas: list[int]) -> list[int]:
    """
    For details of this algorithm, consult the paper.
    """
    added_edges = []
    eliminated_edges = []
    fas_count = len(fas)
    while fas_count > 0:
        count = 0
        for i in range(1, len(fas) + 1):
            edge = fas[i+count]
            # we don't actually remove the edge from `fas`,
            # since it's not really needed
            fas_count -= 1
            graph.addEdge(edge)
            if is_acyclic(graph):
                added_edges.append(edge)
                count += 1
            else:
                graph.removeEdge(edge)
                eliminated_edges.append(edge)
    return eliminated_edges


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {sys.argv[0]} [filename]")
    graph = load_graph(sys.argv[0])
