import argparse
import itertools
import os
import sys
import time
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Iterator, TextIO

from feedback_arc_set import Mode, feedback_arc_set
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
        "-m",
        "--mode",
        action="store",
        choices=list(Mode),
        type=Mode.from_str,
        default="normal",
        help="chooses the algorithm running mode (default normal)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        action="store_true",
        default=False,
        help="makes the algorithm slower, but produces better results",
    )
    parser.add_argument(
        "-c",
        "--concurrent-threads",
        action="store",
        dest="concurrent_threads",
        type=int,
        default=1,
        help="set the number of concurrent threads to run separate instances (default 1)",
    )
    parser.add_argument(
        "-p",
        "--parallel-threads",
        action="store",
        dest="parallel_threads",
        type=int,
        default=1,
        help="set the number of threads to use for each instance (only with -m parallel, default 1)",
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
        dirpath = os.path.join(dir, os.path.dirname(filename))
        filepath = os.path.join(dir, filename)
        os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w") as output:
            yield output
    else:
        yield fallback


def run_algorithm(
    filename: str,
    output_dir: str | None,
    log_dir: str | None,
    use_smartAE: bool,
    reduce: bool,
    mode: Mode,
    quality: bool,
    threads: int,
):
    with (
        open_textIO(output_dir, f"{filename}.out", sys.stdout) as out_file,
        open_textIO(log_dir, f"{filename}.log", sys.stderr) as log_file,
    ):
        print(f"Reading input file {filename}", file=log_file)
        if args.format == "adjacency-list":
            graph, node_id_mapping = NetworkitGraph.load_from_adjacency_list(
                filename
            )
        else:
            graph, node_id_mapping = NetworkitGraph.load_from_edge_list(
                filename
            )
        
        num_nodes = graph.get_num_nodes()
        num_edges = graph.get_num_edges()

        print("Starting calculation of minFAS", file=log_file)
        start_time = time.time()
        fas_instances = feedback_arc_set(
            graph,
            use_smartAE=use_smartAE,
            reduce=reduce,
            mode=mode,
            quality=quality,
            log_file=log_file,
            threads=threads,
        )
        end_time = time.time()

        print(
            f"V = {num_nodes}, E = {num_edges}",
            file=out_file,
        )
        for method, fas in fas_instances.items():
            print(method, len(fas), file=out_file)
        print(f"Execution time: {end_time - start_time} s", file=out_file)


if __name__ == "__main__":
    args = argument_parser().parse_args()
    if args.concurrent_threads == 1:
        for filename in expand_files(args.files):
            run_algorithm(
                filename,
                args.output,
                args.log,
                args.smartAE,
                args.reduce,
                args.mode,
                args.quality,
                args.parallel_threads,
            )
    else:
        with ProcessPoolExecutor(args.concurrent_threads) as executor:
            executor.map(
                run_algorithm,
                expand_files(args.files),
                itertools.repeat(args.output),
                itertools.repeat(args.log),
                itertools.repeat(args.smartAE),
                itertools.repeat(args.reduce),
                itertools.repeat(args.mode),
                itertools.repeat(args.quality),
                itertools.repeat(args.parallel_threads),
            )
