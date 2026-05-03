import random
import time

# --- 1. AÇÕES DO AGENTE ---
MOVE_FORWARD = "Mover para Frente"
TURN_LEFT = "Virar à Esquerda"
TURN_RIGHT = "Virar à Direita"
GRAB = "Agarrar"
CLIMB = "Escalar"
SHOOT = "Atirar Flecha"

# Índices: 0: Leste, 1: Norte, 2: Oeste, 3: Sul
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)] 

# --- 2. ESTRUTURA DO AMBIENTE ---
class WumpusWorldEnv:
    def __init__(self, size=4):
        self.size = size
        self.wumpus_pos = None
        self.gold_pos = None
        self.pits = set()
        self.wumpus_alive = True
        self._generate_map()

    def _generate_map(self):
        """Distribui o mapa de forma aleatória garantindo a proteção da entrada."""
        cells = [(x, y) for x in range(1, self.size + 1) for y in range(1, self.size + 1)]
        cells.remove((1, 1)) # Protege [1,1]
        
        self.gold_pos = random.choice(cells)
        cells.remove(self.gold_pos)
        
        self.wumpus_pos = random.choice(cells)
        cells.remove(self.wumpus_pos)
        
        # Gera de 1 a 3 poços aleatoriamente
        num_pits = random.randint(1, 3)
        for _ in range(num_pits):
            if cells:
                p = random.choice(cells)
                self.pits.add(p)
                cells.remove(p)

    def get_percepts(self, pos):
        x, y = pos
        percepts = {"Brisa": False, "Cheiro": False, "Brilho": False, "Grito": False}
        if pos == self.gold_pos:
            percepts["Brilho"] = True
            
        neighbors = [(x, y+1), (x, y-1), (x+1, y), (x-1, y)]
        for nx, ny in neighbors:
            if (nx, ny) == self.wumpus_pos and self.wumpus_alive: 
                percepts["Cheiro"] = True
            if (nx, ny) in self.pits: 
                percepts["Brisa"] = True
                
        return percepts

    def shoot_arrow(self, pos, direction):
        x, y = pos
        dx, dy = direction
        cx, cy = x, y
        while 1 <= cx <= self.size and 1 <= cy <= self.size:
            if (cx, cy) == self.wumpus_pos and self.wumpus_alive:
                self.wumpus_alive = False
                return True # Acertou
            cx += dx
            cy += cy
        return False # Errou

# --- 3. CLASSE DO AGENTE (IDS_SRZ) ---
class IDS_SRZ:
    def __init__(self, env):
        self.env = env
        self.pos = (1, 1)
        self.direction_idx = 0 
        self.has_gold = False
        self.arrows = 2 
        self.score = 0 
        self.nodes_expanded = 0
        
        self.dead = False
        self.escaped = False
        
        # Base de Conhecimento (Knowledge Base)
        self.visited = set()
        self.safe_cells = set([(1, 1)])
        self.breezes = set()
        self.stenches = set()
        self.no_pits = set()
        self.no_wumpus = set()
        self.wumpus_alive = True
        self.percepts_buffer = {}

    def get_neighbors(self, pos):
        x, y = pos
        candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [(cx, cy) for cx, cy in candidates if 1 <= cx <= self.env.size and 1 <= cy <= self.env.size]

    def update_kb(self, percepts):
        self.visited.add(self.pos)
        self.safe_cells.add(self.pos)
        
        if percepts['Brisa']: 
            self.breezes.add(self.pos)
        else:
            for n in self.get_neighbors(self.pos): self.no_pits.add(n)
                
        if percepts['Cheiro']: 
            self.stenches.add(self.pos)
        else:
            for n in self.get_neighbors(self.pos): self.no_wumpus.add(n)
                
        if percepts.get('Grito'):
            self.wumpus_alive = False
            
        # Deduz casas seguras baseadas nas regras
        for v in self.visited:
            for n in self.get_neighbors(v):
                if n in self.no_pits and (n in self.no_wumpus or not self.wumpus_alive):
                    self.safe_cells.add(n)

    # --- ALGORITMO IDS PARA NAVEGAÇÃO ---
    def _dls(self, current, target_func, allowed_nodes, limit, path):
        self.nodes_expanded += 1
        if target_func(current):
            return path
        if limit <= 0:
            return None

        for n in self.get_neighbors(current):
            if n in allowed_nodes and n not in path:
                result = self._dls(n, target_func, allowed_nodes, limit - 1, path + [n])
                if result: 
                    return result
        return None

    def _ids(self, start, target_func, allowed_nodes):
        """IDS que aceita uma função alvo. Usado tanto para explorar quanto para fugir."""
        max_depth = len(allowed_nodes) + 1
        for depth in range(max_depth + 1):
            result = self._dls(start, target_func, allowed_nodes, depth, [start])
            if result: 
                return result
        return None

    # --- EXECUÇÃO DE AÇÕES ---
    def perform_action(self, action):
        self.score -= 1
        print(f"[{self.pos}] Ação: {action} | Custo Ação: -1 | Score Atual: {self.score}")
        
        if action == MOVE_FORWARD:
            dx, dy = DIRECTIONS[self.direction_idx]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 1 <= nx <= self.env.size and 1 <= ny <= self.env.size:
                self.pos = (nx, ny) # Só move se não bater na parede (Choque)
        
        elif action == TURN_LEFT:
            self.direction_idx = (self.direction_idx + 1) % 4
            
        elif action == TURN_RIGHT:
            self.direction_idx = (self.direction_idx - 1) % 4
            
        elif action == GRAB:
            self.has_gold = True
            self.env.gold_pos = None 
            print(" -> OURO CAPTURADO! Iniciando rota de fuga.")
            
        elif action == SHOOT:
            if self.arrows > 0:
                self.score -= 10
                self.arrows -= 1
                print(f" -> Flecha disparada! Custo extra: -10 | Restam: {self.arrows}")
                hit = self.env.shoot_arrow(self.pos, DIRECTIONS[self.direction_idx])
                if hit:
                    self.percepts_buffer['Grito'] = True
                    print(" -> AARGH! O Wumpus foi morto.")
            
        elif action == CLIMB:
            if self.pos == (1,1) and self.has_gold:
                self.score += 1000
                self.escaped = True

    def execute_path(self, path):
        """Converte as coordenadas do IDS em comandos físicos e para ao descobrir novas casas."""
        if len(path) <= 1: return
        
        for i in range(len(path) - 1):
            cx, cy = path[i]
            nx, ny = path[i+1]
            dx, dy = nx - cx, ny - cy
            target_dir = DIRECTIONS.index((dx, dy))
            
            while self.direction_idx != target_dir:
                diff = (target_dir - self.direction_idx) % 4
                if diff == 1:
                    self.perform_action(TURN_LEFT)
                elif diff == 2:
                    self.perform_action(TURN_LEFT)
                    self.perform_action(TURN_LEFT)
                elif diff == 3:
                    self.perform_action(TURN_RIGHT)
            
            self.perform_action(MOVE_FORWARD)
            
            # Se entrou em um território inexplorado, interrompe a caminhada para processar sensores
            if self.pos not in self.visited:
                break 

    def resolve_stuck(self):
        """Toma decisões de risco se o agente ficar encurralado pela lógica segura."""
        print(" -> Risco Necessário: Sem casas garantidamente seguras...")
        
        # 1. Tentar atirar no Wumpus primeiro
        if self.arrows > 0 and self.wumpus_alive:
            for cell in self.visited:
                if cell in self.stenches:
                    for n in self.get_neighbors(cell):
                        if n not in self.visited and n not in self.no_wumpus:
                            path = self._ids(self.pos, lambda c: c == cell, self.safe_cells)
                            if path:
                                self.execute_path(path)
                                # Vira para o suspeito e atira
                                target_dir = DIRECTIONS.index((n[0]-self.pos[0], n[1]-self.pos[1]))
                                while self.direction_idx != target_dir:
                                    if (target_dir - self.direction_idx) % 4 == 1: self.perform_action(TURN_LEFT)
                                    else: self.perform_action(TURN_RIGHT)
                                self.perform_action(SHOOT)
                                return True
                                
        # 2. Movimento as cegas (Risco de cair no Poço)
        frontier = set()
        for cell in self.visited:
            for n in self.get_neighbors(cell):
                if n not in self.visited:
                    frontier.add(n)
        if frontier:
            risky_cell = random.choice(list(frontier))
            origins = [n for n in self.get_neighbors(risky_cell) if n in self.visited]
            path = self._ids(self.pos, lambda c: c == origins[0], self.safe_cells)
            if path is not None:
                self.execute_path(path)
                target_dir = DIRECTIONS.index((risky_cell[0]-self.pos[0], risky_cell[1]-self.pos[1]))
                while self.direction_idx != target_dir:
                    if (target_dir - self.direction_idx) % 4 == 1: self.perform_action(TURN_LEFT)
                    else: self.perform_action(TURN_RIGHT)
                self.perform_action(MOVE_FORWARD)
                return True
        return False

    def check_death(self):
        if self.pos in self.env.pits:
            self.score -= 1000
            self.dead = True
            print(f"\nGAME OVER! O agente caiu no poço em {self.pos}. Penalidade Crítica: -1000")
            return True
        if self.pos == self.env.wumpus_pos and self.env.wumpus_alive:
            self.score -= 1000
            self.dead = True
            print(f"\nGAME OVER! O agente foi devorado em {self.pos}. Penalidade Crítica: -1000")
            return True
        return False

    # --- JORNADA PRINCIPAL ---
    def jornada(self):
        print("=== INICIANDO JORNADA DO AGENTE IDS_SRZ ===")
        start_time = time.perf_counter()
        
        while not self.dead and not self.escaped:
            # 1. Percepção
            percepts = self.env.get_percepts(self.pos)
            if self.percepts_buffer.get('Grito'):
                percepts['Grito'] = True
                self.percepts_buffer['Grito'] = False
                
            # 2. Verificação de Sobrevivência
            if self.check_death():
                break
                
            # 3. Atualiza Conhecimento (KB)
            self.update_kb(percepts)
            
            # 4. Processamento do Objetivo Principal
            if percepts['Brilho'] and not self.has_gold:
                self.perform_action(GRAB)
                continue
                
            # 5. Navegação Estratégica via IDS
            if self.has_gold:
                path = self._ids(self.pos, lambda c: c == (1,1), self.safe_cells)
                if path:
                    self.execute_path(path)
                    if self.pos == (1,1):
                        self.perform_action(CLIMB)
            else:
                path = self._ids(self.pos, lambda c: c in self.safe_cells and c not in self.visited, self.safe_cells)
                if path and len(path) > 1:
                    self.execute_path(path) 
                else:
                    if not self.resolve_stuck():
                        print("Fim da linha. Sem saída possível.")
                        break

        end_time = time.perf_counter()
        tempo_ms = (end_time - start_time) * 1000
        
        print("\n=== RELATÓRIO DE DESEMPENHO (IDS_SRZ) ===")
        print(f"Tempo Total de Operação: {tempo_ms:.4f} ms")
        print(f"Nós Expandidos no IDS (Custo Computacional): {self.nodes_expanded}")
        print(f"Pontuação Final: {self.score} pontos")
        if self.escaped:
            print("Status Final: SUCESSO! Agente sobreviveu e escapou com o Ouro (+1000).")
        elif self.dead:
            print("Status Final: MORTE CONFIRMADA.")

# --- 4. EXECUTANDO A SIMULAÇÃO ---
if __name__ == "__main__":
    caverna = WumpusWorldEnv()
    print(f"=== MUNDO DO WUMPUS (4x4) GERADO ===")
    print(f"Ouro em: {caverna.gold_pos} | Wumpus em: {caverna.wumpus_pos} | Poços em: {list(caverna.pits)}\n")

    agente = IDS_SRZ(caverna)
    agente.jornada()