import itertools
import math
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
                graph = NetworkitGraph(nkgraph)
                merged_edges = graph.remove_runs()
                self.assertEqual([(0, n - 1)], list(graph.graph.iterEdges()))

    def test_bypass(self):
        # before:
        # 0 -> 1 -> ................... -> k - 1
        # |                                  |
        # \ -> n - 1 -> n - 2 -> ... -> k -> /
        # after:
        # 0 -2-> k - 1

        for n in range(3, 10):
            for k, w1, w2, w3 in itertools.product(
                range(1, n), range(1, 6), range(1, 6), range(1, 6)
            ):
                with self.subTest((n, k, w1, w2, w3)):
                    nkgraph = nk.Graph(n, weighted=True, directed=True)
                    for i in range(0, k):
                        nkgraph.addEdge(i, i + 1, w1)
                    nkgraph.addEdge(0, n - 1, w2)
                    for i in range(n - 2, k - 1, -1):
                        nkgraph.addEdge(i + 1, i, w3)
                    graph = NetworkitGraph(nkgraph)
                    graph.remove_runs()
                    self.assertEqual([(0, k)], list(graph.graph.iterEdges()))
                    if k < n - 1:
                        self.assertEqual(
                            w1 + min(w2, w3), graph.graph.weight(0, k)
                        )
                    else:
                        self.assertEqual(w1 + w2, graph.graph.weight(0, k))

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


class TestRemoveTwoCycles(unittest.TestCase):
    def test_source_node(self):
        # before:
        # G1 <-> n <-w1/w2-> n + 1 <- G2, w1 <= w2
        #
        # after:
        # G1 <-> n G2; delete n + 1, w1 * [(n, n + 1)] into FAS

        for n in range(3, 7):
            for w1, w2 in itertools.product(range(1, 6), range(1, 6)):
                with self.subTest((n, w1, w2)):
                    nkgraph = nk.Graph(2 * n + 2, weighted=True, directed=True)
                    # subgraph to the left of the 2-cycle
                    for i in range(n - 1):
                        nkgraph.addEdge(i, i + 1)
                    nkgraph.addEdge(n - 1, 0)

                    # connect to the 2-cycle
                    for i in range(math.ceil(n / 2)):
                        nkgraph.addEdge(i, n)
                    for i in range(math.ceil(n / 2), n):
                        nkgraph.addEdge(n, i)

                    # the 2-cycle
                    nkgraph.addEdge(n, n + 1, w1)
                    nkgraph.addEdge(n + 1, n, w2)

                    # subgraph to the right of the 2-cycle
                    nkgraph.addEdge(2 * n + 1, n + 2)
                    for i in range(n + 2, 2 * n + 1):
                        nkgraph.addEdge(i, i + 1)

                    # connect to the 2-cycle
                    for i in range(n + 2, n + 2 + math.ceil(n / 2)):
                        nkgraph.addEdge(n + 1, i)

                    graph = NetworkitGraph(nkgraph)
                    edges = graph.remove_2cycles()
                    if w1 <= w2:
                        self.assertFalse(graph.graph.hasNode(n + 1))
                        self.assertEqual(w1 * [(n, n + 1)], edges)
                    else:
                        self.assertTrue(graph.graph.hasNode(n + 1))
                        self.assertEqual([], edges)

    def test_sink_node(self):
        # before:
        # G1 <-> n <-w1/w2-> n + 1 <- G2, w2 <= w1
        #
        # after:
        # G1 <-> n G2; delete n + 1, w1 * [(n + 1, n)] into FAS

        for n in range(3, 7):
            for w1, w2 in itertools.product(range(1, 6), range(1, 6)):
                with self.subTest((n, w1, w2)):
                    nkgraph = nk.Graph(2 * n + 2, weighted=True, directed=True)
                    # subgraph to the left of the 2-cycle
                    for i in range(n - 1):
                        nkgraph.addEdge(i, i + 1)
                    nkgraph.addEdge(n - 1, 0)

                    # connect to the 2-cycle
                    for i in range(math.ceil(n / 2)):
                        nkgraph.addEdge(i, n)
                    for i in range(math.ceil(n / 2), n):
                        nkgraph.addEdge(n, i)

                    # the 2-cycle
                    nkgraph.addEdge(n, n + 1, w1)
                    nkgraph.addEdge(n + 1, n, w2)

                    # subgraph to the right of the 2-cycle
                    nkgraph.addEdge(2 * n + 1, n + 2)
                    for i in range(n + 2, 2 * n + 1):
                        nkgraph.addEdge(i, i + 1)

                    # connect to the 2-cycle
                    for i in range(n + 2, n + 2 + math.ceil(n / 2)):
                        nkgraph.addEdge(i, n + 1)

                    graph = NetworkitGraph(nkgraph)
                    edges = graph.remove_2cycles()
                    if w1 >= w2:
                        self.assertFalse(graph.graph.hasNode(n + 1))
                        self.assertEqual(w2 * [(n + 1, n)], edges)
                    else:
                        self.assertTrue(graph.graph.hasNode(n + 1))
                        self.assertEqual([], edges)
