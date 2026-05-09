import sys
import heapq
import random

MOVER = "MOVER"
VIRAR_E = "VIRAR (E)"
VIRAR_D = "VIRAR (D)"
ATIRAR = "ATIRAR"
AGARRAR = "AGARRAR"
ESCALAR = "ESCALAR"
ACOES_VALIDAS = {MOVER, VIRAR_E, VIRAR_D, ATIRAR, AGARRAR, ESCALAR}

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

class Agente:
    def __init__(self):
        self.nome = "AE_SRZ"
        self.estado = "EXPLORANDO"
        self.pos = (1, 1)
        self.direcao = 0 
        self.flechas = 2
        self.tem_ouro = False
        self.wumpus_vivo = True
        
        self.visitadas = set()
        self.celulas_seguras = {(1, 1)}
        self.seguras_nao_visitadas = set()
        
        # Mapeamento Probabilístico
        self.possivel_poco = {(x, y): True for x in range(1, 5) for y in range(1, 5)}
        self.possivel_wumpus = {(x, y): True for x in range(1, 5) for y in range(1, 5)}
        self.possivel_poco[(1, 1)] = False
        self.possivel_wumpus[(1, 1)] = False

        self.mapa_brisa = {}
        self.mapa_cheiro = {}
        self.ultimo_alvo_tiro = None

        self.fila_acoes = []

    def _vizinhos(self, p):
        """Retorna as casas adjacentes válidas dentro do mapa 4x4."""
        v = []
        for dx, dy in DIRECTIONS:
            nx, ny = p[0] + dx, p[1] + dy
            if 1 <= nx <= 4 and 1 <= ny <= 4:
                v.append((nx, ny))
        return v

    def _distancia_manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _a_estrela(self, inicio, destinos):
        """Busca A* garantindo navegação apenas por casas seguras."""
        if inicio in destinos: return [inicio]
        pq = [(0, 0, inicio, [inicio])]
        visitados_busca = set()
        
        while pq:
            _, g, atual, caminho = heapq.heappop(pq)
            if atual in visitados_busca: continue
            visitados_busca.add(atual)
            
            if atual in destinos: return caminho
            
            for prox in self._vizinhos(atual):
                if prox in self.celulas_seguras and prox not in visitados_busca:
                    novo_g = g + 1
                    h = min(self._distancia_manhattan(prox, d) for d in destinos)
                    heapq.heappush(pq, (novo_g + h, novo_g, prox, caminho + [prox]))
        return []

    def _converter_rota(self, rota, atirar_no_fim=False):
        """Converte a rota invisível da IA em movimentos físicos do robô."""
        comandos = []
        dir_temp = self.direcao
        for i in range(len(rota) - 1):
            dx, dy = rota[i+1][0] - rota[i][0], rota[i+1][1] - rota[i][1]
            alvo_dir = DIRECTIONS.index((dx, dy))
            
            while dir_temp != alvo_dir:
                if (dir_temp + 1) % 4 == alvo_dir:
                    comandos.append(VIRAR_E)
                    dir_temp = (dir_temp + 1) % 4
                else:
                    comandos.append(VIRAR_D)
                    dir_temp = (dir_temp - 1) % 4
            
            # Trava de Segurança: Se for para atirar, atira e PARA!           
            if atirar_no_fim and i == len(rota) - 2:
                comandos.append(ATIRAR)
                break
            comandos.append(MOVER)
        return comandos

    def tomar_decisao(self, sensores):
        # 1. Tratamento de Choque
        if "CHOQUE" in sensores:
            dx, dy = DIRECTIONS[self.direcao]
            self.pos = (self.pos[0] - dx, self.pos[1] - dy)
            self.fila_acoes.clear()

        # 2. Avaliação de Morte do Wumpus pelo Tiro do turno anterior
        if "GRITO" in sensores: 
            self.wumpus_vivo = False
            for x in range(1, 5):
                for y in range(1, 5):
                    self.possivel_wumpus[(x, y)] = False
        elif self.ultimo_alvo_tiro is not None:
            # Se atirou e não gritou, Wumpus não está na casa que suspeitávamos
            self.possivel_wumpus[self.ultimo_alvo_tiro] = False
        
        self.ultimo_alvo_tiro = None

        self.visitadas.add(self.pos)
        self.seguras_nao_visitadas.discard(self.pos)
        self.possivel_poco[self.pos] = False
        self.possivel_wumpus[self.pos] = False

        # 3. Coleta do Ouro
        if "BRILHO" in sensores and not self.tem_ouro:
            self.fila_acoes.insert(0, AGARRAR)
        if "OURO" in sensores:
            self.tem_ouro = True
            self.fila_acoes.clear()
            rota = self._a_estrela(self.pos, {(1, 1)})
            if rota: self.fila_acoes.extend(self._converter_rota(rota))
            self.fila_acoes.append(ESCALAR)

        # 4. Inferência Lógica Avançada (Dedução de Poço e Wumpus únicos)
        brisa = "BRISA" in sensores
        cheiro = "CHEIRO" in sensores
        
        self.mapa_brisa[self.pos] = brisa
        self.mapa_cheiro[self.pos] = cheiro

        for nx, ny in self._vizinhos(self.pos):
            if not brisa: self.possivel_poco[(nx, ny)] = False
            if not cheiro: self.possivel_wumpus[(nx, ny)] = False

        mudou = True
        while mudou:
            mudou = False
            for v in self.visitadas:
                if self.mapa_brisa.get(v, False):
                    viz = [n for n in self._vizinhos(v) if self.possivel_poco[n]]
                    if len(viz) == 1: # Descobriu onde está o único poço do jogo!
                        poco_conf = viz[0]
                        for x in range(1, 5):
                            for y in range(1, 5):
                                if (x, y) != poco_conf and self.possivel_poco[(x, y)]:
                                    self.possivel_poco[(x, y)] = False
                                    mudou = True
                
                if self.wumpus_vivo and self.mapa_cheiro.get(v, False):
                    viz = [n for n in self._vizinhos(v) if self.possivel_wumpus[n]]
                    if len(viz) == 1: # Descobriu onde está o único Wumpus!
                        wumpus_conf = viz[0]
                        for x in range(1, 5):
                            for y in range(1, 5):
                                if (x, y) != wumpus_conf and self.possivel_wumpus[(x, y)]:
                                    self.possivel_wumpus[(x, y)] = False
                                    mudou = True

        for x in range(1, 5):
            for y in range(1, 5):
                if not self.possivel_poco[(x, y)] and (not self.wumpus_vivo or not self.possivel_wumpus[(x, y)]):
                    if (x, y) not in self.visitadas and (x, y) not in self.celulas_seguras:
                        self.celulas_seguras.add((x, y))
                        self.seguras_nao_visitadas.add((x, y))

        # 5. Planejamento do A* e Escolha de Risco Inteligente
        if not self.fila_acoes and not self.tem_ouro:
            if self.seguras_nao_visitadas:
                rota = self._a_estrela(self.pos, self.seguras_nao_visitadas)
                if rota: self.fila_acoes.extend(self._converter_rota(rota))
            else:
                fronteira = set()
                for v in self.visitadas:
                    for n in self._vizinhos(v):
                        if n not in self.visitadas:
                            fronteira.add(n)
                
                if fronteira:
                    # Tenta arriscar apenas em casas que já deduziu não possuírem poço
                    alvos_sem_poco = [f for f in fronteira if not self.possivel_poco[f]]
                    alvo = None
                    atirar = False
                    
                    if alvos_sem_poco:
                        alvos_com_wumpus = [f for f in alvos_sem_poco if self.possivel_wumpus[f]]
                        if alvos_com_wumpus and self.flechas > 0:
                            alvo = random.choice(alvos_com_wumpus)
                            atirar = True
                        else:
                            alvo = random.choice(alvos_sem_poco)
                    else:
                        alvo = random.choice(list(fronteira))
                        
                    self.celulas_seguras.add(alvo)
                    rota = self._a_estrela(self.pos, {alvo})
                    if rota: self.fila_acoes.extend(self._converter_rota(rota, atirar_no_fim=atirar))

        # 6. Execução e Atualização de Estado
        if self.fila_acoes:
            acao = self.fila_acoes.pop(0)
            if acao == MOVER:
                self.pos = (self.pos[0] + DIRECTIONS[self.direcao][0], self.pos[1] + DIRECTIONS[self.direcao][1])
            elif acao == VIRAR_E: 
                self.direcao = (self.direcao + 1) % 4
            elif acao == VIRAR_D: 
                self.direcao = (self.direcao - 1) % 4
            elif acao == ATIRAR: 
                self.flechas -= 1
                dx, dy = DIRECTIONS[self.direcao]
                self.ultimo_alvo_tiro = (self.pos[0] + dx, self.pos[1] + dy)
            return acao
            
        return ESCALAR

# =====================================================================
# SISTEMA DE AVALIAÇÃO E CHAVEAMENTO DO JUIZ
# =====================================================================

def modo_master():
    agente = Agente()
    print(agente.nome, flush=True)
    while True:
        linha = sys.stdin.readline()
        if not linha: break
        linha = linha.strip()
        sensores = linha.split(',') if linha != "NADA" else []
        acao = agente.tomar_decisao(sensores)
        print(acao, flush=True)

def carregar_mundo_stdin():
    linhas = []
    for _ in range(4):
        linha = sys.stdin.readline()
        if not linha: break
        linhas.append(linha.strip().split())
    if len(linhas) < 4: return None
    mundo = {'wumpus': None, 'buraco': None, 'ouro': None, 'wumpus_vivo': True}
    for r in range(4):
        for c in range(4):
            val = linhas[r][c]
            x, y = c + 1, 4 - r
            if val == '#': mundo['wumpus'] = (x, y)
            elif val == '*': mundo['buraco'] = (x, y)
            elif val == '$': mundo['ouro'] = (x, y)
            elif val == '@': mundo['saida'] = (x, y)
    if mundo['ouro'] is None and mundo['wumpus'] is not None:
        mundo['ouro'] = mundo['wumpus']
    return mundo

def modo_judge_interno():
    mundo = carregar_mundo_stdin()
    if not mundo:
        print("FRACASSO")
        return

    agente = Agente()
    x, y = 1, 1
    direcao = 0
    dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]

    flechas = 2
    tem_ouro = False
    jogo_ativo = True
    resultado_final = "FRACASSO"

    bateu_parede = False
    gritou = False
    pegou_ouro = False
    erro_escada = False
    entrou_sala_ouro = False

    turnos = 0
    while jogo_ativo and turnos < 500:
        turnos += 1
        if (x, y) == mundo['wumpus'] and mundo['wumpus_vivo']: break
        if (x, y) == mundo['buraco']: break

        sensores = []
        adjacentes = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if mundo['buraco'] in adjacentes: sensores.append("BRISA")
        if mundo['wumpus_vivo'] and mundo['wumpus'] in adjacentes: sensores.append("CHEIRO")
        if entrou_sala_ouro:
            sensores.append("BRILHO")
            entrou_sala_ouro = False
        if bateu_parede:
            sensores.append("CHOQUE")
            bateu_parede = False
        if gritou:
            sensores.append("GRITO")
            gritou = False
        if pegou_ouro:
            sensores.append("OURO")
            pegou_ouro = False
        if erro_escada:
            sensores.append("NO ESCADA")
            erro_escada = False

        acao = agente.tomar_decisao(sensores)

        if acao == MOVER:
            nx, ny = x + dx[direcao], y + dy[direcao]
            if 1 <= nx <= 4 and 1 <= ny <= 4:
                x, y = nx, ny
                if (x, y) == mundo['ouro'] and not tem_ouro:
                    entrou_sala_ouro = True
            else:
                bateu_parede = True
        elif acao == VIRAR_E: direcao = (direcao + 1) % 4
        elif acao == VIRAR_D: direcao = (direcao - 1) % 4
        elif acao == ATIRAR:
            if flechas > 0:
                flechas -= 1
                tx, ty = x + dx[direcao], y + dy[direcao]
                if (tx, ty) == mundo['wumpus'] and mundo['wumpus_vivo']:
                    mundo['wumpus_vivo'] = False
                    gritou = True
        elif acao == AGARRAR:
            if (x, y) == mundo['ouro'] and not tem_ouro:
                tem_ouro = True
                pegou_ouro = True
        elif acao == ESCALAR:
            if (x, y) == (1, 1):
                if tem_ouro:
                    resultado_final = "SUCESSO"
                    jogo_ativo = False
                else:
                    erro_escada = True

    print(resultado_final)

if __name__ == "__main__":
    if "--master" in sys.argv:
        modo_master()
    else:
        modo_judge_interno()