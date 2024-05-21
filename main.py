import sys
import os
from feedback_arc_set import feedback_arc_set

from networkit_fas import NetworkitGraph

if __name__ == "__main__":

    """if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print("Error: Invalid folder path.")
        sys.exit(1)

    for filename in os.listdir(folder_path):
        if filename.endswith(".al"):
            file_path = os.path.join(folder_path, filename)
            graph = NetworkitGraph.load_from_adjacency_list(file_path)
            arcset = feedback_arc_set(graph, use_smartAE=True, reduce=True)
            print(filename, ": ", len(arcset))"""

    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {sys.argv[0]} [filename]")
    file_path = os.path.join("random-large", "randomlarge-0003-25-30.al")
    graph = NetworkitGraph.load_from_adjacency_list(file_path)
    arcset = feedback_arc_set(graph, use_smartAE=True, reduce=True)

    print('result arc set:')
    print(arcset)
