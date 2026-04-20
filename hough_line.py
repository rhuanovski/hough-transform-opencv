import cv2
import numpy as np
import os
import math

# pasta de entrada e saída
pasta_entrada = "img_line"
pasta_saida = "resultado_line"

# cria a pasta de saída, se não existir
os.makedirs(pasta_saida, exist_ok=True)

# extensões permitidas
extensoes_validas = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# parâmetros do algoritmo
CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 50
HOUGH_MAX_LINE_GAP = 10

# confiança mínima para aceitar a linha
CONFIANCA_MINIMA = 0.30


def limitar(valor, minimo=0.0, maximo=1.0):
    return max(minimo, min(maximo, valor))


def calcular_comprimento(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def amostrar_pixels_linha(edges, x1, y1, x2, y2):
    """
    Retorna os valores dos pixels ao longo da linha usando interpolação discreta.
    """
    comprimento = int(max(abs(x2 - x1), abs(y2 - y1))) + 1

    xs = np.linspace(x1, x2, comprimento).astype(int)
    ys = np.linspace(y1, y2, comprimento).astype(int)

    h, w = edges.shape[:2]
    valores = []

    for x, y in zip(xs, ys):
        if 0 <= x < w and 0 <= y < h:
            valores.append(edges[y, x])

    return np.array(valores, dtype=np.uint8)


def calcular_confianca_linha(edges, x1, y1, x2, y2, largura_img, altura_img):
    """
    Estima a confiança da linha com base em:
    1. comprimento relativo
    2. cobertura de borda ao longo da linha
    3. força média da borda
    """
    comprimento = calcular_comprimento(x1, y1, x2, y2)

    diagonal = math.hypot(largura_img, altura_img)
    comprimento_relativo = comprimento / diagonal

    valores = amostrar_pixels_linha(edges, x1, y1, x2, y2)

    if len(valores) == 0:
        return 0.0

    cobertura_borda = np.mean(valores > 0)

    if np.any(valores > 0):
        forca_borda = np.mean(valores[valores > 0]) / 255.0
    else:
        forca_borda = 0.0

    confianca = (
        0.40 * comprimento_relativo +
        0.40 * cobertura_borda +
        0.20 * forca_borda
    )

    return limitar(confianca)


def remover_linhas_duplicadas(linhas, tolerancia_dist=12, tolerancia_angulo=8):
    """
    Remove linhas muito parecidas, mantendo a de maior confiança.
    Cada item da lista é um dicionário:
    {
        "x1":..., "y1":..., "x2":..., "y2":...,
        "conf":..., "angulo":...
    }
    """
    if not linhas:
        return []

    linhas = sorted(linhas, key=lambda l: l["conf"], reverse=True)
    filtradas = []

    for linha in linhas:
        duplicada = False

        for f in filtradas:
            # compara centros
            cx1 = (linha["x1"] + linha["x2"]) / 2
            cy1 = (linha["y1"] + linha["y2"]) / 2
            cx2 = (f["x1"] + f["x2"]) / 2
            cy2 = (f["y1"] + f["y2"]) / 2

            dist = math.hypot(cx1 - cx2, cy1 - cy2)
            dif_angulo = abs(linha["angulo"] - f["angulo"])

            if dist <= tolerancia_dist and dif_angulo <= tolerancia_angulo:
                duplicada = True
                break

        if not duplicada:
            filtradas.append(linha)

    return filtradas


# lista os arquivos de imagem
arquivos_imagem = [
    arquivo for arquivo in os.listdir(pasta_entrada)
    if arquivo.lower().endswith(extensoes_validas)
]

if not arquivos_imagem:
    print(f"Nenhuma imagem encontrada na pasta '{pasta_entrada}'.")
    exit()

print(f"Foram encontradas {len(arquivos_imagem)} imagem(ns).")
print()

for nome_arquivo in arquivos_imagem:
    caminho_imagem = os.path.join(pasta_entrada, nome_arquivo)

    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        print(f"Não foi possível abrir '{caminho_imagem}'. Pulando arquivo.")
        continue

    nome_base = os.path.splitext(nome_arquivo)[0]
    altura, largura = imagem.shape[:2]

    # cópia para desenhar
    resultado = imagem.copy()

    # escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # suavização para reduzir ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # detector de bordas
    bordas = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # Transformada de Hough probabilística
    linhas = cv2.HoughLinesP(
        bordas,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    linhas_confiaveis = []

    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]

            angulo = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            confianca = calcular_confianca_linha(
                bordas, x1, y1, x2, y2, largura, altura
            )

            linhas_confiaveis.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": confianca,
                "angulo": angulo
            })

    # remove duplicadas
    linhas_confiaveis = remover_linhas_duplicadas(linhas_confiaveis)

    # filtra por confiança mínima
    linhas_finais = [
        linha for linha in linhas_confiaveis
        if linha["conf"] >= CONFIANCA_MINIMA
    ]

    # desenha as linhas encontradas
    for linha in linhas_finais:
        x1, y1, x2, y2 = linha["x1"], linha["y1"], linha["x2"], linha["y2"]
        conf = linha["conf"]

        cv2.line(resultado, (x1, y1), (x2, y2), (0, 255, 0), 2)

        texto = f"{conf:.2f}"
        tx = int((x1 + x2) / 2)
        ty = int((y1 + y2) / 2)

        cv2.putText(
            resultado,
            texto,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    # nomes dos arquivos de saída
    arquivo_gray = os.path.join(pasta_saida, f"{nome_base}_gray.jpg")
    arquivo_bordas = os.path.join(pasta_saida, f"{nome_base}_bordas.jpg")
    arquivo_resultado = os.path.join(pasta_saida, f"{nome_base}_resultado_hough.jpg")

    # salva resultados
    cv2.imwrite(arquivo_gray, gray)
    cv2.imwrite(arquivo_bordas, bordas)
    cv2.imwrite(arquivo_resultado, resultado)

    print(f"Imagem processada: {nome_arquivo}")
    print(f"  Linhas finais detectadas: {len(linhas_finais)}")
    print(f"  - {arquivo_gray}")
    print(f"  - {arquivo_bordas}")
    print(f"  - {arquivo_resultado}")
    print()

print("Processamento em lote concluído.")
