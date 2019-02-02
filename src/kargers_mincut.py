from random import sample
from math import log, ceil
from collections import deque

class Graph(object):
    '''
        Wrapper for a graph equipped with randomized min-cut algorithms (Karger, Karger-stein) and deterministic Edmond-Karp algorithm
        for finding the *size* of the minimum cut in the graph.
    '''

    def __init__(self, adj_mat):
        '''
            Takes adjacency matrix (symmetric) as an input; of an UNDIRECTED weighted graph.
            Non-existent edges have weight zero. All weights are nonnegative.    // O(|V|^2)
        '''
        self.V = len(adj_mat) # number of vertices
        # adjacency matrix copy
        self.g = [[adj_mat[i][j] for j in range(self.V)] for i in range(self.V)]
        # stores whether nodes have been removed in edge contractions
        self.nodes = [True for _ in range(self.V)]

        # helps to uniquely identify edges in temporary multigraphs
        self.edge_id = 0
        # stores the ids of every edge between two vertices
        self.edge_logger = [[[] for _ in range(self.V)] for _ in range(self.V)]
        # set of all of the non-zero edges in the multigraph
        # (at the start, no multiedges are assumed)
        self.edges = set()
        for i in range(self.V):
            for j in range(i + 1, self.V):
                if self.g[i][j]:
                    self.add_edge(i, j)

        # backup of self.g, self.V for future restoration of the original graph
        self.gg = [[self.g[i][j] for j in range(self.V)] for i in range(self.V)]
        self.VV = self.V

    def contract_random_edge(self):
        ''' Picks an edge in graph uniformly at random, contracts it.    // O(|V|)'''
        if len(self.edges) > 0:
            # sample edge uniformly
            fr, to, e_id = sample(self.edges, 1)[0]
            # we remove vertex to and contract it into fr.
            for i in range(self.VV):
                if i != fr:
                    self.g[fr][i] += self.g[to][i]
                    self.g[i][fr] += self.g[i][to]
                    # redirect any edges starting/ending at to, into fr
                    if self.g[to][i] and fr < i:
                        self.add_edge(fr, i)
                    if self.g[i][to] and i < fr:
                        self.add_edge(i, fr)
                self.g[to][i] = 0
                self.g[i][to] = 0

                # remove any edges going into to in our list
                if to < i:
                    for rem_id in self.edge_logger[to][i]:
                        self.edges.remove((to, i, rem_id))
                    self.edge_logger[to][i] = []
                elif i < to:
                    for rem_id in self.edge_logger[i][to]:
                        self.edges.remove((i, to, rem_id))
                    self.edge_logger[i][to] = []

            # mark to as deleted
            self.nodes[to] = False
        # one vertex was removed, decrease vertex counter
        self.V -= 1

    def add_edge(self, fr, to):
        ''' Adds a new (multi)edge between fr and to into the graph. '''
        self.edges.add((fr, to, self.edge_id))
        self.edge_logger[fr][to].append(self.edge_id)
        self.edge_id += 1

    def contract_until(self, t=2):
        ''' Contracts edges uniformly at random until the number of vertices is t.   // O(|V|^2)'''
        while self.V > t:
            self.contract_random_edge()

    def karger_trial(self):
        '''
            One trial in the Karger algorithm. Contract until only two vertices remain,
            return the sum of all of the weights of edges between these two.
        '''
        self.contract_until(2)
        return max(self.g[i][j] for i in range(self.VV) for j in range(self.VV))

    def restore_original(self):
        ''' Restore the original graph. '''
        self.V = self.VV
        self.g = [[self.gg[i][j] for j in range(self.VV)] for i in range(self.VV)]
        self.nodes = [True for _ in range(self.VV)]
        self.edge_id = 0
        self.edge_logger = [[[] for _ in range(self.V)] for _ in range(self.V)]
        self.edges = set()
        for i in range(self.V):
            for j in range(i + 1, self.V):
                if self.g[i][j]:
                    self.add_edge(i, j)

    def karger(self):
        '''
            Kargers contraction algorithm. Repeat karger trial |V|^2 log|V| times.
            Returns the *size* of the (1/n probable) minimum cut.  // O(|V|^4 log|V|)
        '''
        mincut = float('inf')
        for _ in range(1 + int(self.VV * self.VV * log(1 + self.VV))):
            mincut = min(mincut, self.karger_trial())
            self.restore_original()
        return mincut

    def karger_stein(self):
        '''
            Karger-Stein algorithm. Should be rougly an order of magnitude faster than Karger.
            Returns the *size* of the (1/ log n probable) minimum cut.
        '''
        # adjustable constant
        if self.V <= 30:
            return Graph(self.g).mincut(method='edmonds_karp')
        t = ceil(1 + self.V / 2**0.5)

        self.contract_until(t)
        G1 = Graph([[self.g[i][j] for j in range(self.VV) if self.nodes[j]] for i in range(self.VV) if self.nodes[i]])
        # important: add *all* of the remaining edges with their multiplicities into G1
        skippedi = 0
        for i in range(self.VV):
            if self.nodes[i]:
                skippedj = 0
                for j in range(self.VV):
                    if self.nodes[j]:
                        for _ in range(len(self.edge_logger[i][j]) - 1):
                            G1.add_edge(i - skippedi, j - skippedj)
                    else:
                        skippedj += 1
            else:
                skippedi += 1
        self.restore_original()

        self.contract_until(t)
        G2 = Graph([[self.g[i][j] for j in range(self.VV) if self.nodes[j]] for i in range(self.VV) if self.nodes[i]])
        skippedi = 0
        for i in range(self.VV):
            if self.nodes[i]:
                skippedj = 0
                for j in range(self.VV):
                    if self.nodes[j]:
                        for _ in range(len(self.edge_logger[i][j]) - 1):
                            G2.add_edge(i - skippedi, j - skippedj)
                    else:
                        skippedj += 1
            else:
                skippedi += 1

        return min(G1.karger_stein(), G2.karger_stein())


    def edmonds_karp_bfs(self,s, t, parent):
        '''Returns true if there is a path from source 's' to sink 't' in
        residual graph. Also fills parent[] to store the path '''
        visited = [False] * (self.VV)
        queue = deque()
        queue.append(s)
        visited[s] = True
        while queue:
            u = queue.popleft()
            for ind, val in enumerate(self.g[u]):
                if visited[ind] == False and val > 0 :
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
        return visited[t]

    def edmonds_karp(self, source, sink):
        ''' Returns the max flow / min cut from source to sink. '''
        parent = [-1] * (self.VV)
        max_flow = 0
        while self.edmonds_karp_bfs(source, sink, parent):
            path_flow = float("Inf")
            s = sink
            while s != source:
                path_flow = min(path_flow, self.g[parent[s]][s])
                s = parent[s]
            max_flow += path_flow
            v = sink
            while v !=  source:
                u = parent[v]
                self.g[u][v] -= path_flow
                self.g[v][u] += path_flow
                v = parent[v]
        return max_flow #, parent

    def mincut(self, method='karger_stein'):
        ''' Randomized mincut. Returns O(1/n) probable minimum cut. '''
        for i in range(self.VV):
            if all(self.g[j][i] == self.g[i][j] == 0 for j in range(self.VV)):
                return 0

        if method == 'karger':
            return self.karger()

        elif method == 'karger_stein':
            result = float('inf')
            for _ in range(1 + int(log(self.VV)**1.25)):
                result = min(result, self.karger_stein())
                self.restore_original()
            return result

        elif method == 'edmonds_karp':
            result = float('inf')
            for j in range(1, self.VV):
                result = min(result, self.edmonds_karp(0, j))
                self.restore_original()
            return result
        return None

    def __str__(self):
        return '\n'.join(' '.join(map(str, row)) for row in self.g)

    def __repr__(self):
        return self.__str__()
