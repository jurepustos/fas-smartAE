import sys
import os
from feedback_arc_set import feedback_arc_set
import time

from networkit_fas import NetworkitGraph

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if folder_path == "graf_adjacency.txt":
        graph = NetworkitGraph.load_from_adjacency_list(sys.argv[1])
        arcset = feedback_arc_set(graph, use_smartAE=True, reduce=True)
        print(arcset)
        
    
    else:
        print(f"{'Filename':<40}{'Arcset Length':<15}{'Time (s)':<10}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".al"):
                file_path = os.path.join(folder_path, filename)
                graph = NetworkitGraph.load_from_adjacency_list(file_path)
                start_time = time.time()
                arcset = feedback_arc_set(graph, use_smartAE=True, reduce=True)
                end_time = time.time()
                total_time = end_time - start_time
                
                print(f"{filename:<40}{len(arcset):<15}{total_time:.4f}")
