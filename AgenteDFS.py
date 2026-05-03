import random
import time

# --- 1. AÇÕES E PERCEPÇÕES ---
MOVE_FORWARD = "Mover para Frente"
TURN_LEFT = "Virar à Esquerda"
TURN_RIGHT = "Virar à Direita"
GRAB = "Agarrar"
CLIMB = "Escalar"
SHOOT = "Atirar Flecha"

# Sensores baseados no manual: Cheiro, Brisa, Brilho, Choque, Grito
DIRECTIONS = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # Leste, Norte, Oeste, Sul

# --- 2. AMBIENTE ALEATÓRIO ---
class WumpusWorldEnv:
    def __init__(self, size=4):
        self.size = size
        self.wumpus_pos = None
        self.gold_pos = None
        self.pit_pos = None
        self.wumpus_alive = True
        self._generate_map()

    def _generate_map(self):
        """Gera o mapa aleatoriamente garantindo [1,1] seguro."""
        cells = [(x, y) for x in range(1, self.size + 1) for y in range(1, self.size + 1)]
        cells.remove((1, 1))
        
        self.gold_pos = random.choice(cells)
        cells.remove(self.gold_pos)
        
        self.wumpus_pos = random.choice(cells)
        cells.remove(self.wumpus_pos)
        
        self.pit_pos = random.choice(cells)

    def get_percepts(self, pos, bumped=False, screamed=False):
        """Retorna o vetor de percepção do agente."""
        x, y = pos
        percepts = {
            "Cheiro": False, "Brisa": False, "Brilho": False, 
            "Choque": bumped, "Grito": screamed
        }
        
        if pos == self.gold_pos: percepts["Brilho"] = True
        
        neighbors = [(x, y+1), (x, y-1), (x+1, y), (x-1, y)]
        for nx, ny in neighbors:
            if (nx, ny) == self.wumpus_pos and self.wumpus_alive: 
                percepts["Cheiro"] = True
            if (nx, ny) == self.pit_pos: 
                percepts["Brisa"] = True
        return percepts

# --- 3. AGENTE DFS_SRZ ---
class DFS_SRZ:
    def __init__(self, env):
        self.env = env
        self.pos = (1, 1)
        self.direction_idx = 0 
        self.has_gold = False
        self.arrows = 2
        self.score = 0
        self.is_alive = True
        self.start_time = time.perf_counter() # Monitoramento inicia no estado inicial
        self.total_actions = 0
        
        self.safe_cells = set([(1, 1)])
        self.visited = set()
        self.unvisited_safe = set([(1, 1)])

    def registrar_acao(self, nome_acao):
        """Deduz -1 por ação e monitora o custo."""
        self.score -= 1
        self.total_actions += 1
        print(f"[{self.total_actions:02d}] Ação: {nome_acao:18} | Posição: {self.pos} | Score: {self.score}")

    def atirar_flecha(self):
        """Custo: -1 da ação e -10 da flecha."""
        if self.arrows > 0:
            self.registrar_acao(SHOOT)
            self.arrows -= 1
            self.score -= 10
            # Simulação simplificada do grito se o Wumpus estiver na linha de tiro
            print(f" -> Flecha disparada! Custo extra: -10 | Restantes: {self.arrows}")
            return True
        return False

    def _find_path_dfs(self, start, targets):
        """Algoritmo DFS usando Pilha para busca e retorno."""
        stack = [(start, [start])]
        visited_dfs = set()
        
        while stack:
            current, path = stack.pop() # LIFO
            if current in visited_dfs: continue
            visited_dfs.add(current)
            
            if current in targets: return path
            
            x, y = current
            for dx, dy in reversed(DIRECTIONS):
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.safe_cells and (nx, ny) not in visited_dfs:
                    stack.append(((nx, ny), path + [(nx, ny)]))
        return []

    def executar_movimento(self, caminho):
        """Executa fisicamente as ações no ambiente."""
        if not caminho or len(caminho) == 1: return

        for i in range(len(caminho) - 1):
            if not self.is_alive: break
            cx, cy = caminho[i]
            nx, ny = caminho[i+1]
            dx, dy = nx - cx, ny - cy
            target_dir = DIRECTIONS.index((dx, dy))
            
            # Ajuste de direção
            while self.direction_idx != target_dir:
                self.registrar_acao(TURN_LEFT)
                self.direction_idx = (self.direction_idx + 1) % 4
            
            self.registrar_acao(MOVE_FORWARD)
            self.pos = (nx, ny)
            
            # Verificação de morte (Wumpus ou Poço)
            if self.pos == self.env.wumpus_pos and self.env.wumpus_alive:
                self.is_alive = False
            elif self.pos == self.env.pit_pos:
                self.is_alive = False

    def finalizar(self):
        """Análise final de desempenho, custo e tempo."""
        if not self.is_alive:
            self.score -= 1000
            print(f"\n GAME OVER: Agente morreu em {self.pos}. Penalidade: -1000")
        else:
            self.registrar_acao(CLIMB)
            if self.has_gold and self.pos == (1, 1):
                self.score += 1000
                print("\nVITÓRIA: Ouro coletado e saída realizada! Pontos: +1000")

        tempo_total = (time.perf_counter() - self.start_time) * 1000
        print("\n" + "="*45)
        print("--- ANÁLISE DE DESEMPENHO (DFS_SRZ) ---")
        print(f"Tempo Total: {tempo_total:.4f} ms")
        print(f"Custo Total de Ações: {self.total_actions}")
        print(f"Pontuação Final: {self.score} pontos")
        print("="*45)

# --- 4. CICLO DE VIDA DA MISSÃO ---
if __name__ == "__main__":
    caverna = WumpusWorldEnv()
    agente = DFS_SRZ(caverna)
    
    print(f"=== MISSÃO DFS_SRZ INICIADA EM [1,1] ===")
    print(f"Mapa Aleatório: Ouro{caverna.gold_pos}, Wumpus{caverna.wumpus_pos}, Poço{caverna.pit_pos}\n")

    # Exploração até encontrar o ouro ou morrer
    while agente.is_alive and not agente.has_gold:
        caminho = agente._find_path_dfs(agente.pos, agente.unvisited_safe)
        if not caminho: break 
            
        agente.executar_movimento(caminho)
        if not agente.is_alive: break

        agente.unvisited_safe.discard(agente.pos)
        agente.visited.add(agente.pos)
        
        percepcoes = caverna.get_percepts(agente.pos)
        
        if percepcoes["Brilho"]:
            agente.registrar_acao(GRAB)
            agente.has_gold = True
            print(" -> Brilho detectado! Ouro coletado.")
        
        # Se sentir perigo, pode usar flecha (opcional na lógica DFS)
        if percepcoes["Cheiro"] and agente.arrows > 0:
            agente.atirar_flecha()

        # Atualiza células seguras baseadas em sensores
        if not percepcoes["Brisa"] and not percepcoes["Cheiro"]:
            x, y = agente.pos
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 1 <= nx <= caverna.size and 1 <= ny <= caverna.size:
                    if (nx, ny) not in agente.visited:
                        agente.safe_cells.add((nx, ny))
                        agente.unvisited_safe.add((nx, ny))

    # Retorno à posição inicial [1,1]
    if agente.is_alive and agente.pos != (1, 1):
        print("\n--- INICIANDO RETORNO ---")
        rota_volta = agente._find_path_dfs(agente.pos, {(1, 1)})
        agente.executar_movimento(rota_volta)

    agente.finalizar()