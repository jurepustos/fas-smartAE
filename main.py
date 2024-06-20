import argparse
import sys
import time

from feedback_arc_set import feedback_arc_set
from networkit_fas import NetworkitGraph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A runner of a heuristic algorithm for calculating a minimum Feedback Arc Set of a directed graph"
    )
    parser.add_argument(
        "filename", help="path to the file containing the input graph"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-a",
        "--adjacency-list",
        action="store_const",
        dest="format",
        const="adjacency-list",
        default="adjacency-list",
        help="read the input file as an adjacency list",
    )
    input_group.add_argument(
        "-e",
        "--edge-list",
        action="store_const",
        dest="format",
        const="edge-list",
        help="read the input file as an edge list (enabled by default)",
    )
    parser.add_argument(
        "-r",
        "--reduce",
        action="store_true",
        dest="reduce",
        default=False,
        help="apply graph reductions",
    )
    parser.add_argument(
        "-s",
        "--smartAE",
        action="store_true",
        dest="smartAE",
        default=False,
        help="apply the smartAE heuristic",
    )
    parser.add_argument(
        "-ro",
        "--random-ordering",
        action="store_true",
        dest="random_ordering",
        default=False,
        help="use a random ordering",
    )
    parser.add_argument(
        "-go",
        "--greedy-orderings",
        action="store_true",
        dest="greedy_orderings",
        default=False,
        help="use gredy orderings",
    )

    args = parser.parse_args()

    print(f"Reading input file {args.filename}", file=sys.stderr)
    if args.format == "adjacency-list":
        graph, node_id_mapping = NetworkitGraph.load_from_adjacency_list(
            args.filename
        )
    else:
        graph, node_id_mapping = NetworkitGraph.load_from_edge_list(
            args.filename
        )

    print("Starting calculation of minFAS")
    start_time = time.time()
    fas_instances = feedback_arc_set(
        graph,
        use_smartAE=args.smartAE,
        reduce=args.reduce,
        random_ordering=args.random_ordering,
        greedy_orderings=args.greedy_orderings,
    )
    end_time = time.time()

    print(f"V = {graph.get_num_nodes()}, E = {graph.get_num_edges()}")
    for method, fas in fas_instances.items():
        # print(method, len(fas), fas)
        print(method, len(fas))
    print(f"Execution time: {end_time - start_time} s")
