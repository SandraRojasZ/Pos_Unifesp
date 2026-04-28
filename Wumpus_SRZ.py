import time
import heapq
from collections import deque

# --- CONFIGURAÇÕES DO AMBIENTE ---
# Grade 4x4. Início em (1,1).
GRID_SIZE = 4
START = (1, 1)
GOLD = (4, 4) 
# Simulando perigos (Poços/Wumpus) conhecidos para forçar o agente a desviar
HAZARDS = {(2, 2), (3, 2), (3, 3)} 

# --- FUNÇÕES AUXILIARES ---
def get_neighbors(pos):
    """Retorna vizinhos válidos."""
    x, y = pos
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Cima, Baixo, Direita, Esquerda
    neighbors = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE and (nx, ny) not in HAZARDS:
            neighbors.append((nx, ny))
    return neighbors

def manhattan_distance(pos1, pos2):
    """Heurística admissível para a grade."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# --- ALGORITMOS DE BUSCA ---

def bfs(start, goal):
    """Busca em Largura (Cega)."""
    queue = deque([(start, [start])])
    visited = set([start])
    nodes_expanded = 0

    while queue:
        current, path = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            return path, nodes_expanded

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None, nodes_expanded

def dfs(start, goal):
    """Busca em Profundidade (Cega)."""
    stack = [(start, [start])]
    visited = set()
    nodes_expanded = 0

    while stack:
        current, path = stack.pop()
        
        if current in visited:
            continue
            
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            return path, nodes_expanded

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None, nodes_expanded

def ucs(start, goal):
    """Busca de Custo Uniforme (Cega)."""
    pq = [(0, start, [start])] # (custo, posição, caminho)
    visited = set()
    nodes_expanded = 0

    while pq:
        cost, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            return path, nodes_expanded

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                # O custo de cada passo no mapa seguro é 1
                heapq.heappush(pq, (cost + 1, neighbor, path + [neighbor]))
    return None, nodes_expanded

def greedy_bfs(start, goal):
    """Busca Gulosa (Informada)"""
    pq = [(manhattan_distance(start, goal), start, [start])] # h(n)
    visited = set()
    nodes_expanded = 0

    while pq:
        _, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            return path, nodes_expanded

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                h = manhattan_distance(neighbor, goal)
                heapq.heappush(pq, (h, neighbor, path + [neighbor]))
    return None, nodes_expanded

def a_star(start, goal):
    """Algoritmo A* (Informada)"""
    pq = [(0 + manhattan_distance(start, goal), 0, start, [start])] # f(n) = g(n) + h(n)
    visited = set()
    nodes_expanded = 0

    while pq:
        f, g, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            return path, nodes_expanded

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                new_g = g + 1
                new_f = new_g + manhattan_distance(neighbor, goal)
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
    return None, nodes_expanded

# --- AVALIAÇÃO E COMPARAÇÃO ---
def evaluate_algorithms():
    algorithms = {
        "BFS (Largura)": bfs,
        "DFS (Profundidade)": dfs,
        "UCS (Custo Uniforme)": ucs,
        "Greedy (Gulosa)": greedy_bfs,
        "A* (A-Estrela)": a_star
    }

    print(f"{'Algoritmo':<22} | {'Custo (Passos)':<15} | {'Nós Expandidos':<15} | {'Tempo (ms)':<10}")
    print("-" * 70)

    for name, func in algorithms.items():
        start_time = time.perf_counter()
        path, nodes = func(START, GOLD)
        end_time = time.perf_counter()
        
        exec_time_ms = (end_time - start_time) * 1000
        cost = len(path) - 1 if path else float('inf') # -1 para descontar a casa inicial
        
        print(f"{name:<22} | {cost:<15} | {nodes:<15} | {exec_time_ms:.4f}")

if __name__ == "__main__":
    evaluate_algorithms()