import sys
import os
from feedback_arc_set import feedback_arc_set
import time

from networkit_fas import NetworkitGraph

if __name__ == "__main__":

    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <folder_path>")
    #     sys.exit(1)
    #
    # folder_path = sys.argv[1]
    #
    # if not os.path.isdir(folder_path):
    #     print("Error: Invalid folder path.")
    #     sys.exit(1)
    #
    # for filename in os.listdir(folder_path):
    #     if filename.endswith(".al"):
    #         file_path = os.path.join(folder_path, filename)
    #         graph = NetworkitGraph.load_from_adjacency_list(file_path)
    #         start_time = time.time()
    #         arcset = feedback_arc_set(graph, use_smartAE=True, reduce=True)
    #         end_time = time.time()
    #         total_time = end_time - start_time
    #         
    #         print(f"{filename:<40}{len(arcset):<15}{total_time:.4f}")
    #
    # # TODO: primer k crkne
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {sys.argv[0]} [filename]")
    filename = sys.argv[1]
    graph = NetworkitGraph.load_from_adjacency_list(filename)
    start_time = time.time()
    arcset = feedback_arc_set(graph, use_smartAE=True, reduce=True)
    end_time = time.time()
    total_time = end_time - start_time

    print(f"{filename:<40} {len(arcset):<15}{total_time:.4f}")
