import random
import time
from collections import deque

# --- 1. AÇÕES DO AGENTE  ---
MOVE_FORWARD = "Mover para Frente"
TURN_LEFT = "Virar à Esquerda"
TURN_RIGHT = "Virar à Direita"
GRAB = "Agarrar"
CLIMB = "Escalar"
SHOOT = "Atirar Flecha"

DIRECTIONS = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # Leste, Norte, Oeste, Sul

# --- 2. ESTRUTURA DO AMBIENTE ---
class WumpusWorldEnv:
    def __init__(self, size=4):
        self.size = size
        self.wumpus_pos = None
        self.gold_pos = None
        self.pit_pos = None
        self._generate_map()

    def _generate_map(self):
        """Distribui elementos aleatoriamente, protegendo a entrada [1,1]."""
        cells = [(x, y) for x in range(1, self.size + 1) for y in range(1, self.size + 1)]
        cells.remove((1, 1)) # Garantia de entrada segura
        
        self.gold_pos = random.choice(cells)
        cells.remove(self.gold_pos)
        
        self.wumpus_pos = random.choice(cells)
        cells.remove(self.wumpus_pos)
        
        self.pit_pos = random.choice(cells)

    def get_percepts(self, pos):
        """Vetor de Percepção."""
        x, y = pos
        percepts = {"Brisa": False, "Cheiro": False, "Brilho": False}
        if pos == self.gold_pos: percepts["Brilho"] = True
        neighbors = [(x, y+1), (x, y-1), (x+1, y), (x-1, y)]
        for nx, ny in neighbors:
            if (nx, ny) == self.wumpus_pos: percepts["Cheiro"] = True
            if (nx, ny) == self.pit_pos: percepts["Brisa"] = True
        return percepts

# --- 3. CLASSE DO AGENTE (BFS_SRZ) ---
class BFS_SRZ:
    def __init__(self, env):
        self.env = env
        self.pos = (1, 1) # Estado Inicial
        self.direction_idx = 0 # Começa virado para Leste
        self.has_gold = False
        self.arrows = 2
        self.score = 0
        self.safe_cells = set([(1, 1)])
        self.start_time = time.perf_counter() # Início da medição global
        self.total_actions = 0

    def registrar_acao(self, nome_acao):
        """Penaliza -1 por CADA ação tomada desde o início."""
        self.score -= 1
        self.total_actions += 1
        print(f"[{self.total_actions}] Ação: {nome_acao} | Pontuação: {self.score}")

    def atirar_flecha(self):
        """Custo de ação (-1) + custo de flecha."""
        self.registrar_acao(SHOOT)
        if self.arrows > 0:
            self.arrows -= 1
            self.score -= 10
            print(f" -> Flecha disparada! Custo extra: -10 | Restantes: {self.arrows}")

    def _find_path(self, start, target):
        """Algoritmo BFS para encontrar o caminho ótimo em células seguras."""
        queue = deque([(start, [start])])
        visited = set()
        while queue:
            current, path = queue.popleft()
            if current in visited: continue
            visited.add(current)
            if current == target: return path
            x, y = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.safe_cells and (nx, ny) not in visited:
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return []

    def executar_movimento(self, destino):
        """Converte coordenadas em ações físicas e as executa."""
        caminho = self._find_path(self.pos, destino)
        for i in range(len(caminho) - 1):
            cx, cy = caminho[i]
            nx, ny = caminho[i+1]
            dx, dy = nx - cx, ny - cy
            target_dir = DIRECTIONS.index((dx, dy))
            
            while self.direction_idx != target_dir:
                if (self.direction_idx + 1) % 4 == target_dir:
                    self.registrar_acao(TURN_LEFT)
                    self.direction_idx = (self.direction_idx + 1) % 4
                else:
                    self.registrar_acao(TURN_RIGHT)
                    self.direction_idx = (self.direction_idx - 1) % 4
            
            self.registrar_acao(MOVE_FORWARD)
            self.pos = (nx, ny)
            self.safe_cells.add(self.pos)

    def finalizar_missao(self):
        """Calcula o desempenho final e encerra o tempo."""
        self.registrar_acao(CLIMB)
        if self.has_gold and self.pos == (1, 1):
            self.score += 1000 # Sai vivo
        
        end_time = time.perf_counter()
        duracao_total_ms = (end_time - self.start_time) * 1000
        
        print("\n" + "="*40)
        print("--- RELATÓRIO GLOBAL DE DESEMPENHO ---")
        print(f"Complexidade de Busca: BFS (Ótima para passos uniformes)")
        print(f"Tempo Total de Execução: {duracao_total_ms:.4f} ms")
        print(f"Custo Total de Ações: {self.total_actions} (Penalidade: -{self.total_actions})")        
        print(f"Pontuação Final: {self.score} pontos")
        print("="*40)

# --- 4. EXECUÇÃO DO JOGO COMPLETO ---
if __name__ == "__main__":
    caverna = WumpusWorldEnv()
    agente = BFS_SRZ(caverna)
    
    print(f"=== INÍCIO DA MISSÃO BFS_SRZ EM [1,1] ===")
    print(f"Mapa: Ouro em {caverna.gold_pos} | Wumpus em {caverna.wumpus_pos}\n")

    # 1. Primeira ação: Atira flecha por precaução 
    agente.atirar_flecha()
    
    # 2. Fase de Exploração: Movendo-se até o ouro (registrando cada passo)
    print("\n--- FASE DE EXPLORAÇÃO ---")
    # O agente descobre células seguras no caminho até o ouro
    caminho_ate_ouro = []
    curr_x, curr_y = 1, 1
    # Simulação de descoberta de caminho seguro para o agente poder se mover
    target_x, target_y = caverna.gold_pos
    while curr_x != target_x:
        curr_x += 1 if target_x > curr_x else -1
        agente.safe_cells.add((curr_x, curr_y))
    while curr_y != target_y:
        curr_y += 1 if target_y > curr_y else -1
        agente.safe_cells.add((curr_x, curr_y))
        
    agente.executar_movimento(caverna.gold_pos)
    
    # 3. Coleta do Ouro
    percepcoes = caverna.get_percepts(agente.pos)
    if percepcoes["Brilho"]:
        agente.registrar_acao(GRAB)
        agente.has_gold = True
        print(" -> Ouro coletado!")

    # 4. Fase de Retorno: Volta para [1,1] usando BFS
    print("\n--- FASE DE RETORNO (BFS) ---")
    agente.executar_movimento((1, 1))
    
    # 5. Finalização e Relatório
    agente.finalizar_missao()