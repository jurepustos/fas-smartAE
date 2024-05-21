import sys

from feedback_arc_set import feedback_arc_set

from networkit_fas import NetworkitGraph

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {sys.argv[0]} [filename]")
    graph = NetworkitGraph.load_from_adjacency_list(sys.argv[1])
    print('node degrees:')
    for node in graph.get_nodes():
        print(node, graph.get_in_degree(node), graph.get_out_degree(node))
    arcset = feedback_arc_set(graph, use_smartAE=True, reduce=True)

    graph.remove_edges(arcset)
    for node in graph.get_nodes():
        if graph.get_out_degree(node) == 0:
            print('isolated node:', node)
        else:
            for neighbor in graph.iter_out_neighbors(node):
                print(node, neighbor)
    test = NetworkitGraph.is_acyclic_topologically(graph)
    print('result is acyclic:', test)

    print('result arc set:')
    print(arcset)