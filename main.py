from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse

class WorkerMachineAssignment:
    def __init__(self, n_workers, n_machines):
        self.n_workers = n_workers
        self.n_machines = n_machines
        self.compatibility_matrix = np.zeros((n_workers, n_machines), dtype=int)
        self.flow_network = None
        self.assignment = {}

    def set_compatibility(self, worker_id, machine_id, compatible=True):
        if 0 <= worker_id < self.n_workers and 0 <= machine_id < self.n_machines:
            self.compatibility_matrix[worker_id][machine_id] = 1 if compatible else 0
        else:
            raise ValueError("Nieprawidłowe ID pracownika lub maszyny")

    def set_compatibility_matrix(self, matrix):
        if matrix.shape != (self.n_workers, self.n_machines):
            raise ValueError(f"Macierz musi mieć wymiary {self.n_workers}x{self.n_machines}")
        self.compatibility_matrix = matrix.astype(int)

    def build_flow_network(self):
        total_vertices = 2 + self.n_workers + self.n_machines
        self.flow_network = defaultdict(lambda: defaultdict(int))

        source = 0
        sink = total_vertices - 1

        for w in range(1, self.n_workers + 1):
            self.flow_network[source][w] = 1

        for w in range(self.n_workers):
            for m in range(self.n_machines):
                if self.compatibility_matrix[w][m] == 1:
                    worker_vertex = w + 1
                    machine_vertex = self.n_workers + 1 + m
                    self.flow_network[worker_vertex][machine_vertex] = 1

        for m in range(self.n_machines):
            machine_vertex = self.n_workers + 1 + m
            self.flow_network[machine_vertex][sink] = 1

        return source, sink, total_vertices

    def bfs_find_path(self, source, sink, parent):
        visited = set([source])
        queue = deque([source])

        while queue:
            u = queue.popleft()

            for v in self.flow_network[u]:
                if v not in visited and self.flow_network[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
        return False

    def ford_fulkerson(self):
        source, sink, total_vertices = self.build_flow_network()
        parent = [-1] * total_vertices
        max_flow_value = 0

        residual_graph = defaultdict(lambda: defaultdict(int))
        for u in self.flow_network:
            for v in self.flow_network[u]:
                residual_graph[u][v] = self.flow_network[u][v]

        while self.bfs_find_path_residual(source, sink, parent, residual_graph):
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, residual_graph[parent[s]][s])
                s = parent[s]

            v = sink
            while v != source:
                u = parent[v]
                residual_graph[u][v] -= path_flow
                residual_graph[v][u] += path_flow
                v = parent[v]

            max_flow_value += path_flow

        return max_flow_value, residual_graph

    def bfs_find_path_residual(self, source, sink, parent, residual_graph):
        visited = set([source])
        queue = deque([source])

        while queue:
            u = queue.popleft()

            for v in residual_graph[u]:
                if v not in visited and residual_graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
        return False

    def extract_assignment(self, residual_graph):
        self.assignment = {}

        for w in range(1, self.n_workers + 1):
            for m in range(self.n_workers + 1, self.n_workers + 1 + self.n_machines):
                if residual_graph[m][w] > 0:
                    worker_id = w - 1
                    machine_id = m - self.n_workers - 1
                    self.assignment[worker_id] = machine_id

    def solve(self):
        if not np.any(self.compatibility_matrix):
            return 0, {}

        max_flow, residual_graph = self.ford_fulkerson()
        self.extract_assignment(residual_graph)

        return max_flow, self.assignment

    def print_solution(self):
        max_assignments, assignment = self.solve()

        print(f"Maksymalna liczba przypisań: {max_assignments}")
        print(f"Całkowita liczba pracowników: {self.n_workers}")
        print(f"Całkowita liczba maszyn: {self.n_machines}")
        print()

        if assignment:
            print("Optymalne przypisania:")
            for worker_id, machine_id in assignment.items():
                print(f"Pracownik {worker_id} -> Maszyna {machine_id}")
        else:
            print("Brak możliwych przypisań!")

        print("\nMacierz zgodności:")
        print("   ", end="")
        for m in range(self.n_machines):
            print(f"M{m:2d}", end=" ")
        print()

        for w in range(self.n_workers):
            print(f"W{w:2d}", end=" ")
            for m in range(self.n_machines):
                symbol = " ✓ " if self.compatibility_matrix[w][m] else " ✗ "
                print(symbol, end=" ")
            print()

    def visualize_bipartite_graph(self):
        G = nx.Graph()
        workers = [f'W{i}' for i in range(self.n_workers)]
        machines = [f'M{i}' for i in range(self.n_machines)]

        G.add_nodes_from(workers, bipartite=0)
        G.add_nodes_from(machines, bipartite=1)

        for w in range(self.n_workers):
            for m in range(self.n_machines):
                if self.compatibility_matrix[w][m] == 1:
                    G.add_edge(f'W{w}', f'M{m}')

        pos = {}
        for i, worker in enumerate(workers):
            pos[worker] = (0, i)
        for i, machine in enumerate(machines):
            pos[machine] = (2, i)

        plt.figure(figsize=(12, 8))

        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray')

        if hasattr(self, 'assignment') and self.assignment:
            assigned_edges = []
            for worker_id, machine_id in self.assignment.items():
                assigned_edges.append((f'W{worker_id}', f'M{machine_id}'))

            nx.draw_networkx_edges(G, pos, edgelist=assigned_edges,
                                   width=3, edge_color='red', alpha=0.8)

        nx.draw_networkx_nodes(G, pos, nodelist=workers,
                               node_color='lightblue', node_size=1000, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=machines,
                               node_color='lightgreen', node_size=1000, alpha=0.8)

        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        plt.title('Graf dwudzielny: Przypisanie pracowników do maszyn\n' +
                  'Czerwone krawędzie = optymalne przypisania', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def interactive_input(n_workers, n_machines):
    matrix = np.zeros((n_workers, n_machines), dtype=int)
    print("\nPodaj kompatybilność (1 - kompatybilny, 0 - niekompatybilny):")
    for w in range(n_workers):
        for m in range(n_machines):
            while True:
                try:
                    val = int(input(f"Pracownik {w} - Maszyna {m}: "))
                    if val in (0, 1):
                        matrix[w][m] = val
                        break
                    print("Wprowadz liczbę 0 lub 1")
                except ValueError:
                    print("Wprowadz liczbę 0 lub 1")
    return matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Przypisywanie pracowników do maszyn')
    parser.add_argument('--pracownikow', type=int)
    parser.add_argument('--maszyn', type=int)
    return parser.parse_args()

def main():
    args = parse_args()

    matrix = interactive_input(args.pracownikow, args.maszyn)
    problem = WorkerMachineAssignment(args.pracownikow, args.maszyn)

    problem.set_compatibility_matrix(matrix)
    problem.print_solution()
    problem.visualize_bipartite_graph()

if __name__ == "__main__":
    main()