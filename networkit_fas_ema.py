import graphlib
import networkit as nk
from networkit.graph import Graph
from typing import Iterator, Self
from networkit.components import StronglyConnectedComponents
from networkit.graphtools import GraphTools


G = Graph(10, False, True)

G.addEdge(1, 2)
G.addEdge(2, 4)
G.addEdge(2, 5)
G.addEdge(2, 7)
G.addEdge(3, 1)
G.addEdge(4, 3)
G.addEdge(4, 5)
G.addEdge(5, 6)
G.addEdge(5, 7)
G.addEdge(5, 8)	
G.addEdge(6, 3)
G.addEdge(6, 4)
G.addEdge(7, 6)
G.addEdge(7, 9)
G.addEdge(8, 9)
G.addEdge(8, 7)
G.addEdge(9, 6)


def is_acyclic_topologically(self) -> bool:
        sorter = graphlib.TopologicalSorter()

        for node in get_nodes(self):
            sorter.add(node)
        for source, target in self.iterEdges():
            sorter.add(source, target)     

        try:
            sorted_order = [*sorter.static_order()]
            print(sorted_order)
            return True
        except graphlib.CycleError:
            print("Error: Graph has a cycle")
            return False



def get_nodes(self) -> list[int]:
    return list(self.iterNodes())

def get_edge(self, source: int, target: int) -> tuple[int, int]:
    return (source, target)


def get_edges(self) -> list[tuple[int, int]]:
    return [get_edge(self, source, target) for source, target in self.iterEdges()]


def get_out_degree(self, node: int) -> int:
    return self.degreeOut(node)

def get_in_degree(self, node: int) -> int:
    return self.degreeIn(node)

def iter_out_neighbors(self, node: int) -> Iterator[int]:
    return self.iterNeighbors(node)        #TODO: interOutNeighbors doesnt exist!!

def iter_in_neighbors(self, node: int) -> Iterator[int]:
    return self.iterInNeighbors(node)

def remove_sinks(self):
    sinks = [node for node in self.iterNodes()
              if self.degreeOut(node) == 0]
    for sink in sinks:
        self.removeNode(sink)

def remove_sources(self):
    sources = [node for node in self.iterNodes()
                if self.degreeIn(node) == 0]
    for source in sources:
        self.removeNode(source)

def iter_strongly_connected_components(self) -> Iterator[Self]:
    """
    Returns a list of strongly connected components
    """
    cc = StronglyConnectedComponents(self)
    cc.run()
    for component_nodes in cc.getComponents():
        yield GraphTools.subgraphFromNodes(self, component_nodes)


def find2cycles(self):
    twoCycles = []
    for u in get_nodes(self):
        for v in self.iterNeighbors(u):
            for w in self.iterNeighbors(v):
                if u == w and u < v:
                    twoCycles.append((u, v))
    return twoCycles

    
def find_2_3_cycles(self):
    threeCycles = []
    sorted3cycles = []
    threeBypass = []
    twoCycles = []
    for u in get_nodes(self):
        for v in self.iterNeighbors(u):
            for w in self.iterNeighbors(v):
                if u == w and u < v:
                    twoCycles.append((u, v))
                triple = [u, v, w]
                
                for x in self.iterInNeighbors(w):
                    sorted3 = sorted(triple)
                    if u == x and sorted3 not in threeBypass:
                        if get_in_degree(self, triple[1]) == 1 and get_out_degree(self, triple[1]) == 1:
                          threeBypass.append(triple)
                        
                
                for x in self.iterNeighbors(w):
                  sorted3 = sorted(triple)
                  if u == x and sorted3 not in sorted3cycles:
                      if get_in_degree(self, triple[1]) == 1 and get_out_degree(self, triple[1]) == 1:
                        threeCycles.append(triple)
                        sorted3cycles.append(sorted3)
    return twoCycles, threeCycles, threeBypass


def find3cycles(self):
    threeCycles = []
    sorted3cycles = []
    threeBypass = []
    for u in get_nodes(self):
        for v in self.iterNeighbors(u):
            for w in self.iterNeighbors(v):
                triple = [u, v, w]
                
                for x in self.iterInNeighbors(w):
                    sorted3 = sorted(triple)
                    if u == x and sorted3 not in threeBypass:
                        if get_in_degree(self, triple[1]) == 1 and get_out_degree(self, triple[1]) == 1:
                          threeBypass.append(triple)
                        
                for x in self.iterNeighbors(w):
                  sorted3 = sorted(triple)
                  if u == x and sorted3 not in sorted3cycles:
                      if get_in_degree(self, triple[1]) == 1 and get_out_degree(self, triple[1]) == 1:
                        threeCycles.append(triple)
                        sorted3cycles.append(sorted3)
    return threeCycles, threeBypass


def removeRuns(self):   #remove runs larger than 2
    
    for u in get_nodes(self):
        for v in self.iterNeighbors(u):
            while get_in_degree(self, v) == 1 and get_out_degree(self, v) == 1:
                w = next(iter_out_neighbors(self, v))
                if u < w:
                    self.removeNode(v)
                    self.addEdge(u, w)
                v = w
    return


def removeRuns2(self):    #remove runs of size 2
    for u in get_nodes(self):
        for v in self.iterNeighbors(u):
            if get_in_degree(self, v) == 1 and get_out_degree(self, v) == 1:
                w = next(iter_out_neighbors(self, v))
                if u != w:
                    self.removeNode(v)
                    self.addEdge(u, w)
    return 


def simplification(self):
    
    removeRuns(self)
    
    cy3, by3 = find3cycles(self)
    FAS = []
    """print("two cycles", cy2)
    print("three cycles", cy3)
    print("three bypass", by3)"""

    for triple in cy3:
        a, b, c = triple
        if get_out_degree(self, b) != 1 and get_in_degree(self, b) != 1:
            print("error")
        else:
            self.addEdge(a, c)
            self.removeNode(b)

    for bypass in by3:
        a, b, c = bypass
        if get_out_degree(self, b) != 1 and get_in_degree(self, b) != 1:
            print("error")
        else:
          self.addEdge(a, c)
          self.removeNode(b)

    cy2 = find2cycles(self)

    for pair in cy2:
        a = pair[0]
        b = pair[1]
        print("pair", a, b)
        if get_out_degree(G,b) == 1:
              edge = get_edge(G, b, a)
              FAS.append(edge)
              G.removeNode(b)
        elif get_in_degree(G,b) == 1 and get_out_degree(G, b) <= 3:  # TODO: a rabs tm get_out_degree <= 3?
            edge = get_edge(G, a, b)
            FAS.append(edge)
            G.removeNode(b)     

    return FAS


def compute_scores(self, nodes):
    score1 = {}
    score2 = {}
    for node in nodes:
        in_degree = get_in_degree(self, node)
        out_degree = get_out_degree(self, node)
        score1[node] = abs(in_degree - out_degree)
        if out_degree == 0 and in_degree == 0:
            score2[node] = 0
            continue
        if out_degree > 0:
            ratio = in_degree / out_degree
        else:
            ratio = float('inf')
        if in_degree > 0:
            inv_ratio = out_degree / in_degree
        else:
            inv_ratio = float('inf')
        score2[node] = max(ratio, inv_ratio)

    scores1 = sorted(score1.items(), key=lambda x: x[1], reverse=True)
    scores1 = [t[0] for t in scores1]
    scores2 = sorted(score2.items(), key=lambda item: item[1], reverse=True)
    scores2 = [t[0] for t in scores2]
    print("score1", scores1)
    print("score2", scores2)
    return scores1, scores2


def ureditve(self):
    component_nodes = get_nodes(self)

    out_asc = sorted(component_nodes, key=lambda node: get_out_degree(self, node))
    #out_asc = component_nodes.sort(lambda node: self.get_out_degree(node))
    
    out_desc = sorted(component_nodes, key=lambda node: get_out_degree(self, node), reverse=True)

    in_asc = sorted(component_nodes, key=lambda node: get_in_degree(self, node))
    
    in_desc = sorted(component_nodes, key=lambda node: get_in_degree(self, node), reverse=True)
    
    print("out_asc", out_asc)
    print("out_desc", out_desc)
    print("in_asc", in_asc)
    print("in_desc", in_desc)

    return 


def ureditve2(self):
    component_nodes = get_nodes(self)
    
    component_nodes.sort(key=lambda node: get_out_degree(G, node))  # changed lambda to key
            
    print("out_asc", component_nodes)
    """component_nodes.reverse()
    out_desc = compute_fas(component, component_nodes,
                            use_smartAE=use_smartAE,
                            stopping_condition=stopping_condition)
    component_nodes.sort(lambda node: graph.get_in_degree(node))
    in_asc = compute_fas(component, component_nodes,
                            use_smartAE=use_smartAE,
                            stopping_condition=stopping_condition)
    component_nodes.reverse()
    in_desc = compute_fas(component, component_nodes,
                            use_smartAE=use_smartAE,
                            stopping_condition=stopping_condition)"""

    return 


compute_scores(G, get_nodes(G))

ureditve2(G)

"""#is_acyclic_topologically(G)
print("nodes", get_nodes(G))
print("edges:", get_edges(G))

FAS = simplification(G)

print("nodes", get_nodes(G))
print("edges:", get_edges(G))"""
