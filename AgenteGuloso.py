import random
import time
import heapq
import itertools

# --- 1. AÇÕES E PERCEPÇÕES ---
# O agente pode Mover, Virar, Agarrar, Escalar e Atirar
MOVE_FORWARD = "Mover para Frente"
TURN_LEFT = "Virar à Esquerda"
TURN_RIGHT = "Virar à Direita"
GRAB = "Agarrar"
CLIMB = "Escalar"
SHOOT = "Atirar Flecha"

# 0=Leste, 1=Norte, 2=Oeste, 3=Sul
DIRECTIONS = [(0, 1), (-1, 0), (0, -1), (1, 0)]  

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
        """Retorna o vetor de percepção do agente (Cheiro, Brisa, Brilho, Choque, Grito)."""
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

# --- 3. AGENTE GULOSO_SRZ ---
class Guloso_SRZ:
    def __init__(self, env):
        self.env = env
        self.pos = (1, 1)
        self.direction_idx = 0 
        self.has_gold = False
        self.arrows = 2 # Agente possui 2 flechas
        self.score = 0
        self.is_alive = True
        self.start_time = time.perf_counter() # Monitoramento inicia no estado inicial
        self.total_actions = 0
        
        self.safe_cells = set([(1, 1)])
        self.visited = set()
        self.unvisited_safe = set([(1, 1)])
        self.contador_tiebreak = itertools.count() # Desempate para a Fila de Prioridade

    def registrar_acao(self, nome_acao):
        """Sistema penaliza ações desnecessárias deduzindo -1 por ação."""
        self.score -= 1
        self.total_actions += 1
        print(f"[{self.total_actions:02d}] Ação: {nome_acao:18} | Posição: {self.pos} | Score: {self.score}")

    def atirar_flecha(self):
        """Custo: -1 da ação e -10 por usar a flecha."""
        if self.arrows > 0:
            self.registrar_acao(SHOOT)
            self.arrows -= 1
            self.score -= 10
            # Regra simples: Se o wumpus está vivo, o agente atira por precaução.
            # Aqui não simulamos trajetória complexa, assumimos que atirou na direção que estava olhando.
            print(f" -> Flecha disparada! Custo extra: -10 | Restantes: {self.arrows}")
            return True
        return False

    def heuristica(self, pos, targets):
        """
        Calcula a Distância de Manhattan até o alvo mais próximo.
        Fórmula: MD = |x1 - x2| + |y1 - y2|.
        """
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)

    def _find_path_greedy(self, start, targets, arriscar_perigo=False):
        """Busca Gulosa usando Fila de Prioridade focada na Heurística h(n)."""
        if not targets:
            return []
            
        # priority_queue armazena tuplas: (h(n), tiebreaker, posicao, caminho)
        pq = [(self.heuristica(start, targets), next(self.contador_tiebreak), start, [start])]
        visited_greedy = set()
        
        while pq:
            h, _, current, path = heapq.heappop(pq)
            
            if current in visited_greedy: continue
            visited_greedy.add(current)
            
            if current in targets:
                return path
                
            x, y = current
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 1 <= nx <= self.env.size and 1 <= ny <= self.env.size:
                    # Explora se é seguro, ou se o agente foi forçado a arriscar
                    if (nx, ny) in self.safe_cells or (arriscar_perigo and (nx, ny) not in self.visited):
                        if (nx, ny) not in visited_greedy:
                            novo_caminho = path + [(nx, ny)]
                            prioridade = self.heuristica((nx, ny), targets)
                            heapq.heappush(pq, (prioridade, next(self.contador_tiebreak), (nx, ny), novo_caminho))
        return []

    def executar_movimento(self, caminho):
        """Executa os comandos físicos traduzindo coordenadas em ações."""
        if not caminho or len(caminho) == 1: return

        for i in range(len(caminho) - 1):
            if not self.is_alive: break
            cx, cy = caminho[i]
            nx, ny = caminho[i+1]
            dx, dy = nx - cx, ny - cy
            target_dir = DIRECTIONS.index((dx, dy))
            
            # Cálculo eficiente de rotação
            diff = (target_dir - self.direction_idx) % 4
            if diff == 1:
                self.registrar_acao(TURN_LEFT)
            elif diff == 3:
                self.registrar_acao(TURN_RIGHT)
            elif diff == 2:
                self.registrar_acao(TURN_RIGHT)
                self.registrar_acao(TURN_RIGHT)
            
            self.direction_idx = target_dir
            
            self.registrar_acao(MOVE_FORWARD)
            self.pos = (nx, ny)
            
            # Condições de Morte: Abismo ou Wumpus
            if self.pos == self.env.wumpus_pos and self.env.wumpus_alive:
                self.is_alive = False
            elif self.pos == self.env.pit_pos:
                self.is_alive = False

    def finalizar(self):
        """Encerra a missão calculando a pontuação de recompensa ou penalidade."""
        if not self.is_alive:
            self.score -= 1000
            print(f"\nGAME OVER: Agente morreu em {self.pos}. Penalidade: -1000")
        else:
            self.registrar_acao(CLIMB)
            if self.has_gold and self.pos == (1, 1):
                self.score += 1000
                print("\nVITÓRIA: Ouro coletado e fuga bem-sucedida! Recompensa: +1000")

        tempo_total = (time.perf_counter() - self.start_time) * 1000
        print("\n" + "="*45)
        print("--- RELATÓRIO DE DESEMPENHO: GULOSO_SRZ ---")
        print(f"Tempo Total de Missão : {tempo_total:.4f} ms")
        print(f"Custo Operacional     : {self.total_actions} ações")
        print(f"Pontuação Final       : {self.score} pontos")
        print("="*45)

# --- 4. CICLO DE VIDA DA MISSÃO ---
if __name__ == "__main__":
    caverna = WumpusWorldEnv()
    agente = Guloso_SRZ(caverna)
    
    print(f"=== MISSÃO GULOSO_SRZ INICIADA EM [1,1] ===")
    print(f"Mapa Aleatório: Ouro{caverna.gold_pos}, Wumpus{caverna.wumpus_pos}, Poço{caverna.pit_pos}\n")

    # FASE 1: Exploração sistemática (Busca do Ouro)
    while agente.is_alive and not agente.has_gold:
        # Usa Busca Gulosa para achar o caminho até a célula segura não visitada mais próxima
        caminho = agente._find_path_greedy(agente.pos, agente.unvisited_safe)
        
        # Lógica Kamikaze: Se acabou a segurança e não achou o ouro, não aborta, mas Arrisca!
        if not caminho:
            fronteira = set()
            for vx, vy in agente.visited:
                for dx, dy in DIRECTIONS:
                    nx, ny = vx + dx, vy + dy
                    if 1 <= nx <= caverna.size and 1 <= ny <= caverna.size:
                        if (nx, ny) not in agente.visited:
                            fronteira.add((nx, ny))
            
            if fronteira:
                print("\n[!] Sem caminho seguro. Arriscando ir para o desconhecido...")
                caminho = agente._find_path_greedy(agente.pos, fronteira, arriscar_perigo=True)
            else:
                break # Preso ou mapa 100% explorado
            
        agente.executar_movimento(caminho)
        if not agente.is_alive: break

        agente.unvisited_safe.discard(agente.pos)
        agente.visited.add(agente.pos)
        
        percepcoes = caverna.get_percepts(agente.pos)
        
        if percepcoes["Brilho"]:
            agente.registrar_acao(GRAB)
            agente.has_gold = True
            print(" -> Brilho detectado! Ouro coletado.")
        
        if percepcoes["Cheiro"] and agente.arrows > 0:
            agente.atirar_flecha()

        if not percepcoes["Brisa"] and not percepcoes["Cheiro"]:
            x, y = agente.pos
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 1 <= nx <= caverna.size and 1 <= ny <= caverna.size:
                    if (nx, ny) not in agente.visited:
                        agente.safe_cells.add((nx, ny))
                        agente.unvisited_safe.add((nx, ny))

    # FASE 2: Fuga com o Ouro usando o mesmo Algoritmo Guloso
    if agente.is_alive and agente.pos != (1, 1):
        print("\n--- INICIANDO RETORNO ---")
        # Busca o caminho de volta para [1,1] focado na Distância de Manhattan
        rota_volta = agente._find_path_greedy(agente.pos, {(1, 1)})
        agente.executar_movimento(rota_volta)

    # Monitoramento Final
    agente.finalizar()