import os
from fas_builder import FASBuilder
from networkit_fas import NetworkitGraph

def compute_sccs_and_max_scc(graph: NetworkitGraph, reduce=True):
    # Initial graph properties
    initial_num_nodes = graph.get_num_nodes()
    initial_num_edges = graph.get_num_edges()

    # Compute initial SCCs
    initial_sccs = list(graph.iter_components())
    initial_max_scc_size = max(scc.get_num_nodes() for scc in initial_sccs) if initial_sccs else 0
    initial_num_sccs = len(initial_sccs)

    # Reduction and re-computation if enabled
    if reduce:
        fas_builder = FASBuilder(graph.get_node_labels())
        reduction_merged_edges = {}
        reduction_fas_edges = [(n, n) for n in graph.get_self_loops()]
        reduction_merged_edges.update(graph.remove_runs())
        reduction_fas_edges.extend(graph.remove_2cycles())

        fas_builder.add_fas_edges(reduction_fas_edges)
        fas_builder.add_merged_edges(reduction_merged_edges)

    # Compute SCCs after reduction
    reduced_sccs = list(graph.iter_components())
    reduced_max_scc_size = max(scc.get_num_nodes() for scc in reduced_sccs) if reduced_sccs else 0
    reduced_num_sccs = len(reduced_sccs)

    # Post-reduction graph properties
    reduced_num_nodes = graph.get_num_nodes()
    reduced_num_edges = graph.get_num_edges()

    return (initial_num_sccs, initial_max_scc_size, reduced_num_sccs, reduced_max_scc_size,
            initial_num_nodes, initial_num_edges, reduced_num_nodes, reduced_num_edges)

def process_graphs_in_directory(directory_path):
    # Predefined order of filenames
    order = [
        's27.dot', 's208.d', 's420.dot', 'mm4a.d', 's382.d',
        's344.d', 's349.d', 's400.d', 's526n.d', 'mult16a.d',
        's444.d', 's526.d', 'mult16b.d', 's641.d', 's713.d',
        'mult32a.d', 'mm9a.d', 's838.d', 's953.d', 'mm9b.d',
        's1423.d', 'sbc.d', 'ecc.d', 'phase_decoder.d', 'daio_receiver.d',
        'mm30a.d', 'parker1986.d', 's5378.d', 's9234.d', 'bigkey.d',
        'dsip.d', 's38584.d', 's38417.d'
    ]

    files = os.listdir(directory_path)
    files.sort(key=lambda x: order.index(x) if x in order else 9999)

    for filename in files:
        if filename in order:
            full_path = os.path.join(directory_path, filename)
            try:
                if full_path.endswith(".dot"):
                    graph, labels = NetworkitGraph.load_from_dot(full_path)
                elif full_path.endswith(".d"):
                    graph, labels = NetworkitGraph.load_from_edge_list(full_path)
                else:
                    print(f"Unsupported file type for {filename}")
                    continue

                results = compute_sccs_and_max_scc(graph)
                (initial_num_sccs, initial_max_scc_size, reduced_num_sccs, reduced_max_scc_size,
                 initial_num_nodes, initial_num_edges, reduced_num_nodes, reduced_num_edges) = results

                print(f"File: {filename}")
                print(
                    f"Initial: {initial_num_sccs} SCCs, Max SCC Size: {initial_max_scc_size}, V-E: {initial_num_nodes}-{initial_num_edges}")
                print(
                    f"Reduced: {reduced_num_sccs} SCCs, Max SCC Size: {reduced_max_scc_size}, V-E: {reduced_num_nodes}-{reduced_num_edges}")
                print("----------")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

iscas_directory = 'iscas'

if __name__ == "__main__":
    process_graphs_in_directory(iscas_directory)
