import sys

from feedback_arc_set import feedback_arc_set

from networkit_fas import NetworkitGraph

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {sys.argv[0]} [filename]")
    graph = NetworkitGraph.load_from_edge_list(sys.argv[1])
    print('node degrees:')
    for node in graph.get_nodes():
        print(node, graph.get_in_degree(node), graph.get_out_degree(node))
    arcset = feedback_arc_set(graph, use_smartAE=True, reduce=False)
    print('graph acyclic', graph.is_acyclic())
    print('result arc set:')
    print(arcset)
