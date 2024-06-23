import argparse
import itertools
import os
import sys
import time
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Iterator, TextIO

from feedback_arc_set import feedback_arc_set
from networkit_fas import NetworkitGraph


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


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A runner of a heuristic algorithm for calculating a minimum Feedback Arc Set of a directed graph"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="paths to files containing input graphs",
    )
    parser.add_argument(
        "-f",
        "--format",
        nargs=1,
        action="store",
        choices=["adjacency-list", "edge-list"],
        default="adjacency-list",
        help="select the input file format (default adjacency-list)",
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
        help="use greedy orderings",
    )
    parser.add_argument(
        "-t",
        "--threads",
        nargs=1,
        action="store",
        default=1,
        help="set the number of concurrent threads to run on (default 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        default=None,
        help="set the output directory (outputs to stdout by default)",
    )
    parser.add_argument(
        "-l",
        "--log",
        action="store",
        default=None,
        help="set the log directory (outputs to stderr by default)",
    )

    return parser


@contextmanager
def open_textIO(
    dir: str | None, filename: str | None, fallback: TextIO
) -> Iterator[TextIO]:
    if dir is not None and filename is not None:
        os.makedirs(os.path.join(dir, os.path.dirname(filename)), exist_ok=True)
        with open(filename, "w") as output:
            yield output
    else:
        yield fallback


def run_algorithm(
    filename: str,
    output_dir: str | None,
    log_dir: str | None,
    use_smartAE: bool,
    reduce: bool,
    random_ordering: bool,
    greedy_orderings: bool,
):
    with (
        open_textIO(
            output_dir, f"{filename}.out", sys.stdout
        ) as output,
        open_textIO(log_dir, f"{filename}.log", sys.stderr) as log,
    ):
        print(f"Reading input file {filename}", file=log)
        print(filename, file=output)
        if args.format == "adjacency-list":
            graph, node_id_mapping = NetworkitGraph.load_from_adjacency_list(
                filename
            )
        else:
            graph, node_id_mapping = NetworkitGraph.load_from_edge_list(filename)

        print("Starting calculation of minFAS", file=log)
        start_time = time.time()
        fas_instances = feedback_arc_set(
            graph,
            use_smartAE=use_smartAE,
            reduce=reduce,
            random_ordering=random_ordering,
            greedy_orderings=greedy_orderings,
            log_file=log
        )
        end_time = time.time()

        print(
            f"V = {graph.get_num_nodes()}, E = {graph.get_num_edges()}", file=output
        )
        for method, fas in fas_instances.items():
            # print(method, len(fas), fas)
            print(method, len(fas), file=output)
        print(f"Execution time: {end_time - start_time} s", file=output)


if __name__ == "__main__":
    args = argument_parser().parse_args()
    if args.threads == 1:
        for filename in expand_files(args.files):
            run_algorithm(
                filename,
                args.output,
                args.log,
                args.smartAE,
                args.reduce,
                args.random_ordering,
                args.greedy_orderings,
            )
    else:
        with ProcessPoolExecutor(args.threads) as executor:
            executor.map(
                run_algorithm,
                expand_files(args.files),
                itertools.repeat(args.output),
                itertools.repeat(args.log),
                itertools.repeat(args.smartAE),
                itertools.repeat(args.reduce),
                itertools.repeat(args.random_ordering),
                itertools.repeat(args.greedy_orderings),
            )
