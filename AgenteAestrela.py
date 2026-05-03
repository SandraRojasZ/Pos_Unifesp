import random
import time
import heapq

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

# --- 3. AGENTE AE_SRZ (A-Estrela) ---
class AE_SRZ:
    def __init__(self, env):
        self.env = env
        self.pos = (1, 1)
        self.direction_idx = 0 
        self.has_gold = False
        self.arrows = 2
        self.score = 0
        self.is_alive = True
        
        # Monitoramento inicia no estado inicial [1,1]
        self.start_time = time.perf_counter() 
        self.total_actions = 0
        
        # Memória do Agente
        self.safe_cells = set([(1, 1)])
        self.visited = set()
        self.unvisited_safe = set([(1, 1)])

    def registrar_acao(self, nome_acao):
        """Deduz -1 por ação e monitora o custo."""
        self.score -= 1
        self.total_actions += 1
        print(f"[{self.total_actions:02d}] Ação: {nome_acao:18} | Posição: {self.pos} | Score: {self.score}")

    def atirar_flecha(self):
        """Custo: -1 da ação de atirar e -10 do uso da flecha."""
        if self.arrows > 0:
            self.registrar_acao(SHOOT)
            self.arrows -= 1
            self.score -= 10
            print(f" -> Flecha disparada! Custo extra: -10 | Restantes: {self.arrows} | Score Atual: {self.score}")
            return True
        return False

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def min_manhattan(self, pos, targets):
        """Heurística: Menor distância de Manhattan até o(s) alvo(s)."""
        if not targets: return 0
        return min(self.manhattan_distance(pos, t) for t in targets)

    def _find_path_a_star(self, start, targets):
        """Algoritmo A* usado tanto para buscar o ouro quanto para voltar."""
        if start in targets:
            return [start]

        # Fila de prioridade: (f_cost, g_cost, atual, caminho)
        pq = [(self.min_manhattan(start, targets), 0, start, [start])]
        visited_astar = set()
        
        while pq:
            _, g, current, path = heapq.heappop(pq)
            
            if current in visited_astar: continue
            visited_astar.add(current)
            
            if current in targets: 
                return path
            
            x, y = current
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                # O agente navega apenas pela sua memória de casas seguras
                if (nx, ny) in self.safe_cells and (nx, ny) not in visited_astar:
                    new_g = g + 1
                    h = self.min_manhattan((nx, ny), targets)
                    heapq.heappush(pq, (new_g + h, new_g, (nx, ny), path + [(nx, ny)]))
        return []

    def executar_movimento(self, caminho):
        """Executa fisicamente as ações no ambiente traduzindo coordenadas."""
        if not caminho or len(caminho) == 1: return

        for i in range(len(caminho) - 1):
            if not self.is_alive: break
            cx, cy = caminho[i]
            nx, ny = caminho[i+1]
            dx, dy = nx - cx, ny - cy
            target_dir = DIRECTIONS.index((dx, dy))
            
            # Ajuste de direção rotacionando o agente
            while self.direction_idx != target_dir:
                if (self.direction_idx + 1) % 4 == target_dir:
                    self.registrar_acao(TURN_LEFT)
                    self.direction_idx = (self.direction_idx + 1) % 4
                else:
                    self.registrar_acao(TURN_RIGHT)
                    self.direction_idx = (self.direction_idx - 1) % 4
            
            self.registrar_acao(MOVE_FORWARD)
            self.pos = (nx, ny)
            
            # Verificação de morte (Wumpus ou Poço)
            if self.pos == self.env.wumpus_pos and self.env.wumpus_alive:
                self.is_alive = False
            elif self.pos == self.env.pit_pos:
                self.is_alive = False

    def finalizar(self):
        """Análise final de desempenho, custo e tempo abrangendo toda a jornada."""
        if not self.is_alive:
            self.score -= 1000
            print(f"\n☠️ GAME OVER: Agente morreu na casa {self.pos}. Penalidade: -1000")
        else:
            self.registrar_acao(CLIMB)
            if self.has_gold and self.pos == (1, 1):
                self.score += 1000
                print("\n🏆 VITÓRIA: Ouro coletado e saída realizada! Recompensa: +1000")

        tempo_total = (time.perf_counter() - self.start_time) * 1000
        print("\n" + "="*45)
        print("--- ANÁLISE DE DESEMPENHO (Agente AE_SRZ) ---")
        print(f"Tempo Total (A* em toda jornada): {tempo_total:.4f} ms")
        print(f"Custo Total de Ações Físicas: {self.total_actions}")
        print(f"Pontuação Final: {self.score} pontos")
        print("="*45)

# --- 4. CICLO DE VIDA DA MISSÃO ---
if __name__ == "__main__":
    caverna = WumpusWorldEnv()
    agente = AE_SRZ(caverna)
    
    print(f"=== MISSÃO AE_SRZ INICIADA EM [1,1] ===")
    print(f"Mapa Aleatório gerado: Ouro {caverna.gold_pos}, Wumpus {caverna.wumpus_pos}, Poço {caverna.pit_pos}\n")

    # Fase 1: Exploração usando A* até encontrar o ouro ou morrer
    while agente.is_alive and not agente.has_gold:
        
        # Lógica de não-aborto: Se acabarem as casas seguras, ele assume o risco!
        if not agente.unvisited_safe:
            risky_cells = set()
            for (vx, vy) in agente.visited:
                for dx, dy in DIRECTIONS:
                    nx, ny = vx + dx, vy + dy
                    if 1 <= nx <= caverna.size and 1 <= ny <= caverna.size:
                        if (nx, ny) not in agente.visited and (nx, ny) not in agente.safe_cells:
                            risky_cells.add((nx, ny))
            if risky_cells:
                # Escolhe uma casa adjacente cega e assume o risco
                risky_choice = random.choice(list(risky_cells))
                agente.safe_cells.add(risky_choice)
                agente.unvisited_safe.add(risky_choice)
                print(f"\n⚠️ Sem caminhos 100% seguros! Agente AE_SRZ assume o risco de ir para {risky_choice}!")
            else:
                break # Sem mais casas no mapa (raríssimo em 4x4)

        # Roda o algoritmo A* procurando a casa segura/não-visitada mais próxima
        caminho = agente._find_path_a_star(agente.pos, agente.unvisited_safe)
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
        
        # Se sentir o cheiro do Wumpus, atira flecha para tentar matar (heurística de defesa)
        if percepcoes["Cheiro"] and agente.arrows > 0:
            agente.atirar_flecha()

        # Atualiza o mapa de células seguras apenas se não houver Brisa ou Cheiro
        if not percepcoes["Brisa"] and not percepcoes["Cheiro"]:
            x, y = agente.pos
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 1 <= nx <= caverna.size and 1 <= ny <= caverna.size:
                    if (nx, ny) not in agente.visited:
                        agente.safe_cells.add((nx, ny))
                        agente.unvisited_safe.add((nx, ny))

    # Fase 2: Retorno à posição inicial [1,1] usando o MESMO algoritmo A*
    if agente.is_alive and agente.pos != (1, 1):
        print("\n--- INICIANDO RETORNO ---")
        rota_volta = agente._find_path_a_star(agente.pos, {(1, 1)})
        agente.executar_movimento(rota_volta)

    # Encerra o cronômetro, calcula pontos e mostra o desempenho
    agente.finalizar()