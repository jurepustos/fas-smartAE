import sys

from feedback_arc_set import feedback_arc_set

from networkit_fas import NetworkitGraph

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {sys.argv[0]} [filename]")
    graph = NetworkitGraph.load_from_adjacency_list(sys.argv[1])
    arcset = feedback_arc_set(graph, use_smartAE=True, reduce=True)

    print('result arc set:')
    print(arcset)
