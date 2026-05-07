"""
Template do Agente - Mundo do Wumpus
------------------------------------
INSTRUÇÕES PARA OS ALUNOS:
1. Este script se comunicará automaticamente com o Judge (Avaliador).
2. Não utilize funções como `input()` esperando digitar no teclado. O Judge 
   enviará as informações do ambiente diretamente pela entrada padrão do sistema.
3. Sempre que imprimir uma ação usando `print()`, mantenha o parâmetro `flush=True`.
4. As ações permitidas que você deve retornar (exatamente como escrito) são:
   - MOVER
   - VIRAR (E)
   - VIRAR (D)
   - ATIRAR
   - AGARRAR
   - ESCALAR
5. Os sensores que você receberá em uma string separada por vírgulas são:
   CHEIRO, BRISA, BRILHO, CHOQUE, GRITO, OURO, NO ESCADA, ou NADA
"""

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

# =====================================================================
# ÁREA DO AGENTE IMPLEMENTADO (AE_SRZ com Algoritmo A*)
# =====================================================================

class Agente:
    def __init__(self):
        self.nome = "AE_SRZ"
        self.estado = "EXPLORANDO"
        
        # Estado Físico e Mapeamento
        self.pos = (1, 1)
        self.direcao = 0  # 0: Leste, 1: Norte, 2: Oeste, 3: Sul
        self.flechas = 2
        self.tem_ouro = False
        
        # Vetores de direção correspondentes ao índice da direção (dx, dy)
        self.vetores_direcao = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Memória do Agente
        self.celulas_seguras = {(1, 1)}
        self.visitadas = set()
        self.seguras_nao_visitadas = set()
        
        # Fila de Ações (Planejamento do A*)
        self.fila_acoes = []

    def _distancia_manhattan(self, p1, p2):
        """Heurística para o A*."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _busca_a_estrela(self, inicio, destinos):
        """Algoritmo A* para encontrar o melhor caminho até um dos destinos."""
        if inicio in destinos:
            return [inicio]
        
        # Fila de prioridade: (f_cost, g_cost, nó_atual, caminho_percorrido)
        pq = [(0, 0, inicio, [inicio])]
        visitados_busca = set()
        
        while pq:
            _, g, atual, caminho = heapq.heappop(pq)
            
            if atual in visitados_busca:
                continue
            visitados_busca.add(atual)
            
            if atual in destinos:
                return caminho
            
            for dx, dy in self.vetores_direcao:
                nx, ny = atual[0] + dx, atual[1] + dy
                # A* navega apenas por células consideradas seguras
                if (nx, ny) in self.celulas_seguras and (nx, ny) not in visitados_busca:
                    novo_g = g + 1
                    h = min(self._distancia_manhattan((nx, ny), d) for d in destinos)
                    heapq.heappush(pq, (novo_g + h, novo_g, (nx, ny), caminho + [(nx, ny)]))
        return []

    def _converter_caminho_para_acoes(self, caminho):
        """Converte uma lista de coordenadas do A* em ações físicas reais."""
        acoes = []
        dir_atual = self.direcao
        
        for i in range(len(caminho) - 1):
            cx, cy = caminho[i]
            nx, ny = caminho[i+1]
            dx, dy = nx - cx, ny - cy
            dir_alvo = self.vetores_direcao.index((dx, dy))

            # Gira até ficar de frente para o alvo
            while dir_atual != dir_alvo:
                if (dir_atual + 1) % 4 == dir_alvo:
                    acoes.append(VIRAR_E)
                    dir_atual = (dir_atual + 1) % 4
                else:
                    acoes.append(VIRAR_D)
                    dir_atual = (dir_atual - 1) % 4
            acoes.append(MOVER)
            
        return acoes

    def tomar_decisao(self, sensores):	
        """
        sensores: lista de strings recebida do ambiente. Ex: ["BRISA"] ou []
        Retorne uma das ações permitidas.
        """
        
        # 1. Ajuste mental se bateu em uma parede
        if "CHOQUE" in sensores:
            dx, dy = self.vetores_direcao[self.direcao]
            # Desfaz o movimento mental da rodada passada
            self.pos = (self.pos[0] - dx, self.pos[1] - dy)
            self.fila_acoes.clear()

        # 2. Atualiza a casa atual como visitada
        self.visitadas.add(self.pos)
        self.seguras_nao_visitadas.discard(self.pos)

        # 3. Ações Imediatas (Prioridade Máxima)
        if "BRILHO" in sensores and not self.tem_ouro:
            self.fila_acoes.insert(0, AGARRAR)
            
        if "OURO" in sensores:
            self.tem_ouro = True
            self.fila_acoes.clear() # Cancela exploração
            # O próprio A* recalcula a rota de volta para a saída
            caminho_volta = self._busca_a_estrela(self.pos, {(1, 1)})
            if caminho_volta:
                self.fila_acoes.extend(self._converter_caminho_para_acoes(caminho_volta))
            self.fila_acoes.append(ESCALAR)

        # 4. Processa segurança do ambiente
        if "BRISA" not in sensores and "CHEIRO" not in sensores:
            for dx, dy in self.vetores_direcao:
                nx, ny = self.pos[0] + dx, self.pos[1] + dy
                if 1 <= nx <= 4 and 1 <= ny <= 4:
                    if (nx, ny) not in self.visitadas:
                        self.celulas_seguras.add((nx, ny))
                        self.seguras_nao_visitadas.add((nx, ny))

        # 5. Planejamento com Algoritmo A* (Se estiver sem ações na fila)
        if not self.fila_acoes and not self.tem_ouro:
            # Tenta ir para a casa 100% segura mais próxima
            if self.seguras_nao_visitadas:
                caminho = self._busca_a_estrela(self.pos, self.seguras_nao_visitadas)
                if caminho:
                    self.fila_acoes.extend(self._converter_caminho_para_acoes(caminho))
            
            # Se não há mais casas 100% seguras, assume o risco
            if not self.fila_acoes:
                celulas_risco = []
                for vx, vy in self.visitadas:
                    for dx, dy in self.vetores_direcao:
                        nx, ny = vx + dx, vy + dy
                        if 1 <= nx <= 4 and 1 <= ny <= 4 and (nx, ny) not in self.visitadas:
                            celulas_risco.append((nx, ny))
                
                if celulas_risco:
                    alvo_arriscado = random.choice(celulas_risco)
                    self.celulas_seguras.add(alvo_arriscado) # Finge que é segura para o A* traçar rota
                    caminho = self._busca_a_estrela(self.pos, {alvo_arriscado})
                    if caminho:
                        if self.flechas > 0:
                            self.fila_acoes.append(ATIRAR) # Atira por precaução antes de ir
                        self.fila_acoes.extend(self._converter_caminho_para_acoes(caminho))

        # 6. Executa a próxima ação do planejamento
        if self.fila_acoes:
            acao = self.fila_acoes.pop(0)
            
            # Atualiza a simulação interna da mente do agente
            if acao == MOVER:
                dx, dy = self.vetores_direcao[self.direcao]
                self.pos = (self.pos[0] + dx, self.pos[1] + dy)
            elif acao == VIRAR_E:
                self.direcao = (self.direcao + 1) % 4
            elif acao == VIRAR_D:
                self.direcao = (self.direcao - 1) % 4
            elif acao == ATIRAR:
                self.flechas -= 1
                
            return acao if acao in ACOES_VALIDAS else MOVER
            
        return ESCALAR # Failsafe se travar ou mapa for resolvido

# =====================================================================
# SISTEMA DE AVALIAÇÃO E CHAVEAMENTO (NÃO MODIFICAR ABAIXO)
# =====================================================================

def modo_master():
    agente = Agente()
    print(agente.nome, flush=True)
    while True:
        linha = sys.stdin.readline()
        if not linha:
            break
        linha = linha.strip()
        sensores = linha.split(',') if linha != "NADA" else []
        acao = agente.tomar_decisao(sensores)
        print(acao, flush=True)


def carregar_mundo_stdin():
    linhas = []
    for _ in range(4):
        linha = sys.stdin.readline()
        if not linha:
            break
        linhas.append(linha.strip().split())

    if len(linhas) < 4:
        return None

    mundo = {'wumpus': None, 'buraco': None, 'ouro': None, 'wumpus_vivo': True}
    for r in range(4):
        for c in range(4):
            val = linhas[r][c]
            x, y = c + 1, 4 - r
            if val == '#':
                mundo['wumpus'] = (x, y)
            elif val == '*':
                mundo['buraco'] = (x, y)
            elif val == '$':
                mundo['ouro'] = (x, y)
            elif val == '@':
                mundo['saida'] = (x, y)
    
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

        if (x, y) == mundo['wumpus'] and mundo['wumpus_vivo']:
            break
        if (x, y) == mundo['buraco']:
            break

        sensores = []
        adjacentes = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if mundo['buraco'] in adjacentes:
            sensores.append("BRISA")
        if mundo['wumpus_vivo'] and mundo['wumpus'] in adjacentes:
            sensores.append("CHEIRO")
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
        elif acao == VIRAR_E:
            direcao = (direcao + 1) % 4
        elif acao == VIRAR_D:
            direcao = (direcao - 1) % 4
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