from .networkit_fas import NetworkitGraph
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {sys.argv[0]} [filename]")
    graph = NetworkitGraph.load_from_edge_list(sys.argv[1])
