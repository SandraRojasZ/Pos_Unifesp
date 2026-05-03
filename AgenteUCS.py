import random
import time
import heapq

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
        cells.remove((1, 1))
        
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

# --- 3. CLASSE DO AGENTE (UCS_SRZ) ---
class UCS_SRZ:
    def __init__(self, env):
        self.env = env
        self.pos = (1, 1)
        self.direction_idx = 0 
        self.has_gold = False
        self.arrows = 2
        self.score = 0
        self.start_time = time.perf_counter()
        self.total_actions = 0
        
        # Estruturas para a IA Autônoma baseada em UCS
        self.safe_cells = set([(1, 1)])       # Células conhecidas 100% seguras
        self.visited = set()                  # Células já pisadas
        self.unvisited_safe = set([(1, 1)])   # Fronteira: Seguras, mas não visitadas

    def registrar_acao(self, nome_acao):
        """Penaliza -1 por CADA ação tomada desde o início."""
        self.score -= 1
        self.total_actions += 1
        print(f"[{self.total_actions:02d}] Ação: {nome_acao:18} | Posição: {self.pos} | Pontuação: {self.score}")

    def atirar_flecha(self):
        """Custo de ação (-1) + custo de flecha (-10)."""
        self.registrar_acao(SHOOT)
        if self.arrows > 0:
            self.arrows -= 1
            self.score -= 10
            print(f" -> Flecha disparada! Custo extra: -10 | Restantes: {self.arrows} | Nova Pontuação: {self.score}")

    def _find_path_ucs(self, start, targets):
        """
        Algoritmo UCS.
        Ele calcula o caminho de menor custo até QUALQUER uma das células 'targets'.
        """
        priority_queue = [(0, start, [start])]
        visited_ucs = {} 
        
        while priority_queue:
            cost, current, path = heapq.heappop(priority_queue)
            
            if current in visited_ucs and visited_ucs[current] <= cost:
                continue
            visited_ucs[current] = cost
            
            # Se chegamos a algum dos alvos desejados, retornamos o caminho
            if current in targets: 
                return path
            
            x, y = current
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                # O agente SÓ planeja rotas por cima de células que já validou como seguras
                if (nx, ny) in self.safe_cells:
                    new_cost = cost + 1
                    if (nx, ny) not in visited_ucs or new_cost < visited_ucs.get((nx, ny), float('inf')):
                        heapq.heappush(priority_queue, (new_cost, (nx, ny), path + [(nx, ny)]))
        return []

    def executar_caminho(self, caminho):
        """Navega fisicamente seguindo a rota gerada pelo UCS."""
        if not caminho or len(caminho) == 1:
            return

        for i in range(len(caminho) - 1):
            cx, cy = caminho[i]
            nx, ny = caminho[i+1]
            dx, dy = nx - cx, ny - cy
            target_dir = DIRECTIONS.index((dx, dy))
            
            while self.direction_idx != target_dir:
                diff = (target_dir - self.direction_idx) % 4
                if diff == 1 or diff == -3:
                    self.registrar_acao(TURN_LEFT)
                    self.direction_idx = (self.direction_idx + 1) % 4
                else:
                    self.registrar_acao(TURN_RIGHT)
                    self.direction_idx = (self.direction_idx - 1) % 4
            
            self.registrar_acao(MOVE_FORWARD)
            self.pos = (nx, ny)

    def finalizar_missao(self):
        """Relatório Final."""
        self.registrar_acao(CLIMB)
        if self.has_gold and self.pos == (1, 1):
            self.score += 1000
            print("\n -> SUCESSO: Agente escapou com o ouro! (+1000 pontos)")
        elif self.pos == (1, 1):
            print("\n -> RETIRADA: Agente abortou a missão e sobreviveu, mas sem o ouro.")
        else:
            self.score -= 1000
            print("\n -> GAME OVER: Agente morreu! (-1000 pontos)")
        
        duracao_total_ms = (time.perf_counter() - self.start_time) * 1000
        
        print("\n" + "="*45)
        print("--- RELATÓRIO GLOBAL DE DESEMPENHO ---")
        print(f"Agente                 : UCS_SRZ (100% Autônomo)")
        print(f"Algoritmo Principal    : UCS (Exploração e Fuga)")
        print(f"Tempo Total de Execução: {duracao_total_ms:.4f} ms")
        print(f"Custo de Movimentação  : -{self.total_actions} pontos (ações)")
        print(f"Pontuação Final        : {self.score} pontos")
        print("="*45)


# --- 4. EXECUÇÃO DO JOGO COMPLETO ---
if __name__ == "__main__":
    caverna = WumpusWorldEnv()
    agente = UCS_SRZ(caverna)
    
    print(f"=== INÍCIO DA MISSÃO UCS_SRZ EM [1,1] ===")
    print(f"Mapa Revelado (Backstage): Ouro em {caverna.gold_pos} | Wumpus em {caverna.wumpus_pos} | Poço em {caverna.pit_pos}")

    agente.atirar_flecha()
    
    # ------------------------------------------------------------------
    # FASE 1: EXPLORAÇÃO DIRIGIDA PELO UCS
    # ------------------------------------------------------------------
    print("\n--- FASE DE EXPLORAÇÃO (UCS ATIVO) ---")
    
    while agente.unvisited_safe and not agente.has_gold:
        # Usa o UCS para achar a rota até a célula inexplorada segura MAIS PRÓXIMA
        caminho = agente._find_path_ucs(agente.pos, agente.unvisited_safe)
        
        if not caminho:
            print(" -> Agente cercado por perigos ou sem rotas! Abortando.")
            break
            
        # O agente se move fisicamente até o alvo
        agente.executar_caminho(caminho)
        
        # Atualiza o status da célula atual
        agente.unvisited_safe.remove(agente.pos)
        agente.visited.add(agente.pos)
        
        # Sente o ambiente
        percepcoes = caverna.get_percepts(agente.pos)
        
        if percepcoes["Brilho"]:
            agente.registrar_acao(GRAB)
            agente.has_gold = True
            print(" -> OURO ENCONTRADO E COLETADO!")
            break
            
        # Se for seguro, descobre novas células vizinhas e adiciona à fronteira do UCS
        if not percepcoes["Brisa"] and not percepcoes["Cheiro"]:
            x, y = agente.pos
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 1 <= nx <= caverna.size and 1 <= ny <= caverna.size: # Limites do mapa
                    if (nx, ny) not in agente.visited and (nx, ny) not in agente.safe_cells:
                        agente.safe_cells.add((nx, ny))
                        agente.unvisited_safe.add((nx, ny))

    # ------------------------------------------------------------------
    # FASE 2: RETORNO DIRIGIDO PELO UCS
    # ------------------------------------------------------------------
    print("\n--- FASE DE RETORNO (UCS ATIVO) ---")
    if agente.pos != (1, 1):
        # Usa o UCS novamente: alvo é a saída
        caminho_fuga = agente._find_path_ucs(agente.pos, {(1, 1)})
        if caminho_fuga:
            agente.executar_caminho(caminho_fuga)
        else:
            print(" -> Erro crítico: Caminho de fuga inexistente.")

    agente.finalizar_missao()