import os
import sys
from typing import Iterator

from fas_graph import FASGraph
from networkit_fas import NetworkitGraph
from feedback_arc_set import reduce_graph


def expand_files(files: list[str]) -> Iterator[str]:
    file_stack: list[str] = list(reversed(files))
    while len(file_stack) > 0:
        filename = file_stack.pop()
        if os.path.isfile(filename):
            yield filename
        elif os.path.isdir(filename):
            files = [
                os.path.join(filename, file) for file in os.listdir(filename)
            ]
            file_stack.extend(files)


def print_scc_stats(graph: FASGraph):
    components = [
        comp
        for comp in graph.iter_components()
        if comp.get_num_nodes() > 2
    ]
    print(f"SCC: {len(components)}")
    num_nodes_sorter = graph.__class__.get_num_nodes
    max_graph = max(components, key=num_nodes_sorter)
    max_nodes = max_graph.get_num_nodes()
    max_edges = max_graph.get_num_edges()
    print(f"maxSCC: {max_nodes} nodes, {max_edges} edges")


if __name__ == "__main__":
    for filename in expand_files(sys.argv[1:]):
        print(f"Reading input file {filename}")
        graph, _labels = NetworkitGraph.load_from_adjacency_list(filename)
        print(f"{graph.get_num_nodes()} nodes, {graph.get_num_edges()} edges")
        print_scc_stats(graph)
        reduce_graph(graph)
        print("After reduction:")
        print_scc_stats(graph)
