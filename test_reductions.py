import unittest

import networkit as nk

from networkit_fas import NetworkitGraph


class TestRemoveRuns(unittest.TestCase):
    def test_path(self):
        # before:
        # 0 -> ... -> n - 1
        # after:
        # 0 -> n - 1

        for n in range(3, 10):
            with self.subTest(n):
                nkgraph = nk.Graph(n, weighted=True, directed=True)
                for i in range(0, n - 1):
                    nkgraph.addEdge(i, i + 1)
                labels = list(range(0, n))
                graph = NetworkitGraph(nkgraph, node_labels=labels)
                graph.remove_runs()
                self.assertEqual([(0, n - 1)], list(graph.graph.iterEdges()))

    def test_bypass(self):
        # before:
        # 0 -> 1 -> ................... -> k - 1
        # |                                  |
        # \ -> n - 1 -> n - 2 -> ... -> k -> /
        # after:
        # 0 -2-> k - 1

        for n in range(3, 10):
            for k in range(1, n):
                with self.subTest((n, k)):
                    nkgraph = nk.Graph(n, weighted=True, directed=True)
                    for i in range(0, k):
                        nkgraph.addEdge(i, i + 1)
                    nkgraph.addEdge(0, n - 1)
                    for i in range(n - 2, k - 1, -1):
                        nkgraph.addEdge(i + 1, i)
                    labels = list(range(0, n))
                    graph = NetworkitGraph(nkgraph, node_labels=labels)
                    graph.remove_runs()
                    self.assertEqual([(0, k)], list(graph.graph.iterEdges()))
                    self.assertEqual(2, graph.graph.weight(0, k))

    def test_cycle(self):
        # before:
        # 0 -> ... -> n - 1
        # |             |
        # \ <---------- /
        # after:
        # 0 <-> n - 1

        for n in range(3, 10):
            with self.subTest(n):
                nkgraph = nk.Graph(n, weighted=True, directed=True)
                for i in range(0, n - 1):
                    nkgraph.addEdge(i, i + 1)
                nkgraph.addEdge(n - 1, 0)
                labels = list(range(0, n))
                graph = NetworkitGraph(nkgraph, node_labels=labels)
                graph.remove_runs()
                self.assertSetEqual(
                    {(0, n - 1), (n - 1, 0)}, set(graph.graph.iterEdges())
                )
