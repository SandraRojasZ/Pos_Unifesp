import cv2
import numpy as np
import os
import glob
import sys

# Definimos um limite de recursão maior, embora nossa
# implementação de flood_fill seja iterativa (baseada em pilha) 
# e não sofra de estouro de recursão.
sys.setrecursionlimit(200000) 

# =============================================================================
# FUNÇÕES DE LEITURA/ESCRITA DE IMAGENS
# =============================================================================

def load_image(path: str) -> np.ndarray:
    """Carrega uma imagem em escala de cinza."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {path}")
    return image

def save_image(path: str, img: np.ndarray):
    """Salva uma imagem. Converte máscaras binárias (0/1) para (0/255)."""
    if img.dtype == bool or np.max(img) <= 1:
         # Converte máscara binária para imagem de 8 bits
         img = (img * 255).astype(np.uint8)
    else:
         img = img.astype(np.uint8)
    cv2.imwrite(path, img)

# =============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO
# =============================================================================

def my_resize_nearest(image: np.ndarray, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Redimensiona uma imagem usando o algoritmo "Vizinho Mais Próximo" (Nearest Neighbor).
    """
    h_orig, w_orig = image.shape
    h_targ, w_targ = target_size
    
    # Calcula as proporções
    h_ratio = h_orig / h_targ
    w_ratio = w_orig / w_targ
    
    resized_image = np.zeros(target_size, dtype=image.dtype)
    
    for r_targ in range(h_targ):
        for c_targ in range(w_targ):
            # Encontra o pixel correspondente na imagem original
            r_orig = int(r_targ * h_ratio)
            c_orig = int(c_targ * w_ratio)
            
            # Garante que os índices estão dentro dos limites
            r_orig = min(r_orig, h_orig - 1)
            c_orig = min(c_orig, w_orig - 1)
            
            resized_image[r_targ, c_targ] = image[r_orig, c_orig]
            
    return resized_image

def my_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Aplica a equalização de histograma em uma imagem 8-bits."""
    
    # 1. Calcula o histograma
    hist = np.zeros(256, dtype=int)
    for pixel_value in image.flat:
        hist[pixel_value] += 1
        
    # 2. Calcula a Função de Distribuição Cumulativa (CDF)
    cdf = np.zeros(256, dtype=int)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
        
    # 3. Encontra o valor mínimo da CDF (ignorando zeros)
    cdf_min = 0
    for val in cdf:
        if val > 0:
            cdf_min = val
            break
            
    # 4. Cria a tabela de mapeamento (lookup table)
    total_pixels = image.size
    
    # Evita divisão por zero se todos os pixels forem iguais
    if total_pixels == cdf_min:
        return image 

    mapped = np.zeros(256, dtype=np.uint8)
    den = total_pixels - cdf_min
    
    for v in range(256):
        num = (cdf[v] - cdf_min) * 255
        mapped[v] = int(num / den + 0.5) # +0.5 para arredondamento correto

    # 5. Aplica o mapeamento à imagem
    h, w = image.shape
    equalized_image = np.zeros_like(image)
    for r in range(h):
        for c in range(w):
            equalized_image[r, c] = mapped[image[r, c]]
            
    return equalized_image

# =============================================================================
# FUNÇÃO DE LIMIARIZAÇÃO
# =============================================================================

def my_otsu_threshold(image: np.ndarray) -> int:
    """
    Calcula o limiar ideal usando o método de Otsu.
    """
    hist = np.zeros(256, dtype=int)
    for pixel in image.flat:
        hist[pixel] += 1
    
    total_pixels = image.size
    
    # Soma total de intensidades
    sum_total = 0
    for i in range(256):
        sum_total += i * hist[i]
        
    sum_b = 0.0 # Soma das intensidades do background
    w_b = 0.0 # Peso (contagem de pixels) do background
    w_f = 0.0 # Peso (contagem de pixels) do foreground
    
    max_variance = 0.0
    best_threshold = 0
    
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
            
        w_f = total_pixels - w_b
        if w_f == 0:
            break
            
        sum_b += t * hist[t]
        
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f
        
        variance = w_b * w_f * ((mean_b - mean_f) ** 2)
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
            
    return best_threshold

def my_apply_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Cria máscara binária. 
    Nos raios-X, os pulmões são escuros (menores que o limiar).
    """
    h, w = image.shape
    # 1 = Pulmão/Fundo, 0 = Osso/Tecido Denso
    mask = np.zeros((h, w), dtype=np.uint8) 
    for r in range(h):
        for c in range(w):
            # Pulmões são escuros, então < limiar
            if image[r, c] < threshold:
                mask[r, c] = 1
    return mask

# =============================================================================
# FUNÇÃO DE PÓS-PROCESSAMENTO
# =============================================================================

def my_flood_fill(image: np.ndarray, seed_point: tuple) -> np.ndarray:
    """
    Encontra uma região conectada (de valor 1) a partir de um ponto semente.
    Usa uma abordagem iterativa (pilha) para evitar estouro de recursão.
    Não modifica a imagem original.
    """
    h, w = image.shape
    target_color = 1  # O valor que queremos preencher
    
    # Máscara de 'visitados' e máscara da 'área preenchida'
    visited = np.zeros((h, w), dtype=bool)
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    
    stack = [seed_point]
    
    while stack:
        r, c = stack.pop()
        
        # 1. Verifica limites
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
            
        # 2. Verifica se já foi visitado ou se não é a cor alvo
        if visited[r, c] or image[r, c] != target_color:
            continue
            
        # 3. Marca como visitado e parte da área preenchida
        visited[r, c] = True
        filled_mask[r, c] = 1
        
        # 4. Adiciona vizinhos à pilha (4-conectividade)
        stack.append((r+1, c))
        stack.append((r-1, c))
        stack.append((r, c+1))
        stack.append((r, c-1))
        
    return filled_mask


# =============================================================================
# PIPELINE PRINCIPAL DE EXECUÇÃO
# =============================================================================

def main():
    # pré-processamento
    input_folder = 'img' 
    output_folder = 'masks_segmented'
    debug_folder = 'masks_debug' # para ver etapas intermediárias
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_folder, '*.png'))
    image_paths.extend(glob.glob(os.path.join(input_folder, '*.jpg')))

    if not image_paths:
        print(f"Erro: Nenhuma imagem (.png ou .jpg) encontrada em '{input_folder}'.")
        return

    print(f"Encontradas {len(image_paths)} imagens. Iniciando segmentação 'from scratch'...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"--- Processando: {filename} ---")
        
        try:
            # --- ETAPA 1: Carrega e Pré-processa ---
            image_original = load_image(img_path)
            
            # 1a. Redimensiona
            image_resized = my_resize_nearest(image_original, target_size=(256, 256))
            
            # 1b. Equalização de Histograma (para realçar contraste)
            image_eq = my_histogram_equalization(image_resized)
            save_image(os.path.join(debug_folder, f"1_equalized_{filename}"), image_eq)

            # --- ETAPA 2: Limiarização (Otsu) ---
            threshold = my_otsu_threshold(image_eq)
            print(f"  Limiar de Otsu encontrado: {threshold}")
            
            # 2a. Aplica limiar (Pulmões/Fundo=1, Ossos=0)
            binary_mask = my_apply_threshold(image_eq, threshold)
            save_image(os.path.join(debug_folder, f"2_otsu_mask_{filename}"), binary_mask)

            # --- ETAPA 3: Pós-processamento (Remover Fundo) ---            
            # 3a. Encontra o fundo usando flood fill a partir dos cantos
            h, w = binary_mask.shape
            seeds = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
            background_mask = np.zeros((h, w), dtype=np.uint8)
            
            for seed in seeds:
                if binary_mask[seed[0], seed[1]] == 1: 
                    # para fundir as máscaras
                    background_mask = background_mask | my_flood_fill(binary_mask, seed)
            
            save_image(os.path.join(debug_folder, f"3_background_{filename}"), background_mask)

            # 3b. Remove o fundo da máscara binária (Subtração)            
            internal_mask = np.zeros((h, w), dtype=np.uint8)
            for r in range(h):
                for c in range(w):
                    if binary_mask[r, c] == 1 and background_mask[r, c] == 0:
                        internal_mask[r, c] = 1
                        
            save_image(os.path.join(debug_folder, f"4_internal_mask_{filename}"), internal_mask)
            
            # --- ETAPA 4: Preenche Buracos (Costelas)  ---            
            # 4a. Cria uma máscara invertida: Pulmões=0, Buracos/Fundo=1
            mask_to_fill = np.zeros((h, w), dtype=np.uint8)
            for r in range(h):
                for c in range(w):
                    if internal_mask[r, c] == 0:
                        mask_to_fill[r, c] = 1
            
            # 4b. Encontra a área externa (fundo) usando flood-fill
            # O fill começará em (0,0) (fundo, valor 1) e se espalhará.           
            external_area = my_flood_fill(mask_to_fill, (0, 0))
            save_image(os.path.join(debug_folder, f"5_external_area_{filename}"), external_area)
            
            # 4c. Inverte a área externa para obter a máscara final            
            filled_mask = np.zeros((h, w), dtype=np.uint8)
            for r in range(h):
                for c in range(w):
                    if external_area[r, c] == 0:
                        filled_mask[r, c] = 1

            save_image(os.path.join(debug_folder, f"6_filled_mask_{filename}"), filled_mask)
            
            # --- ETAPA 5: Salva Resultado Final ---
            
            # 5a. Redimensiona máscara de volta ao tamanho original
            final_mask = my_resize_nearest(filled_mask, (image_original.shape[0], image_original.shape[1]))
            
            # 5b. Salva na pasta de saída
            output_path = os.path.join(output_folder, f"mask_{filename}")
            save_image(output_path, final_mask)
            
        except Exception as e:
            print(f"  ERRO ao processar {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("--- Processamento concluído! ---")
    print(f"Máscaras salvas em: {output_folder}")
    print(f"Etapas intermediárias salvas em: {debug_folder}")

if __name__ == "__main__":
    main()