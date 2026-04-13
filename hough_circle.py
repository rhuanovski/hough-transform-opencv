import cv2
import numpy as np
import os
import math

# =========================
# PASTAS
# =========================
PASTA_ENTRADA = "img_circle"
PASTA_SAIDA = "resultado_circle"

os.makedirs(PASTA_SAIDA, exist_ok=True)

EXTENSOES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# =========================
# PARÂMETROS GERAIS
# =========================
BLUR_KERNEL = 5

# Canny para análise de borda usada na confiança
CANNY_LOW = 80
CANNY_HIGH = 160

# confiança mínima para aceitar círculo
CONFIANCA_MINIMA = 0.55

# tolerâncias para remover duplicados
TOL_CENTRO = 12
TOL_RAIO = 10

# =========================
# FUNÇÕES AUXILIARES
# =========================
def limitar(valor, minimo=0.0, maximo=1.0):
    return max(minimo, min(maximo, valor))


def calcular_confianca_circulo(edges, gray, x, y, r):
    """
    Confiança baseada em:
    1. cobertura de borda na circunferência
    2. força média da borda
    3. contraste entre dentro e fora do círculo
    4. validade geométrica da posição/raio
    """
    h, w = edges.shape[:2]

    if r <= 0:
        return 0.0

    # descarta círculos muito próximos das bordas da imagem
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return 0.0

    # amostragem da circunferência
    num_pontos = max(72, int(2 * math.pi * r))
    angulos = np.linspace(0, 2 * np.pi, num_pontos, endpoint=False)

    valores_borda = []

    for ang in angulos:
        px = int(round(x + r * math.cos(ang)))
        py = int(round(y + r * math.sin(ang)))

        if 0 <= px < w and 0 <= py < h:
            valores_borda.append(edges[py, px])

    if not valores_borda:
        return 0.0

    valores_borda = np.array(valores_borda, dtype=np.uint8)

    # 1. cobertura da borda
    cobertura_borda = np.mean(valores_borda > 0)

    # 2. força da borda
    if np.any(valores_borda > 0):
        forca_borda = np.mean(valores_borda[valores_borda > 0]) / 255.0
    else:
        forca_borda = 0.0

    # 3. contraste interno/externo
    mascara_interna = np.zeros_like(gray, dtype=np.uint8)
    mascara_anel = np.zeros_like(gray, dtype=np.uint8)

    raio_interno = max(1, int(r * 0.70))
    raio_externo = max(r + 2, int(r * 1.20))

    cv2.circle(mascara_interna, (x, y), raio_interno, 255, -1)
    cv2.circle(mascara_anel, (x, y), raio_externo, 255, -1)
    cv2.circle(mascara_anel, (x, y), max(1, int(r * 0.90)), 0, -1)

    media_interna = cv2.mean(gray, mask=mascara_interna)[0]
    media_anel = cv2.mean(gray, mask=mascara_anel)[0]

    contraste = abs(media_interna - media_anel) / 255.0

    # 4. consistência radial simples
    # mede se o centro parece diferente da borda/anel
    intensidade_centro = gray[y, x] / 255.0
    consistencia_centro = abs(intensidade_centro - (media_anel / 255.0))

    confianca = (
        0.40 * cobertura_borda +
        0.25 * forca_borda +
        0.25 * contraste +
        0.10 * consistencia_centro
    )

    return limitar(confianca)


def remover_circulos_duplicados(circulos, tolerancia_centro=TOL_CENTRO, tolerancia_raio=TOL_RAIO):
    """
    Mantém apenas círculos únicos, priorizando o de maior confiança.
    """
    if not circulos:
        return []

    circulos_ordenados = sorted(circulos, key=lambda c: c["conf"], reverse=True)
    filtrados = []

    for c in circulos_ordenados:
        duplicado = False

        for f in filtrados:
            dist = math.hypot(c["x"] - f["x"], c["y"] - f["y"])
            dr = abs(c["r"] - f["r"])

            if dist <= tolerancia_centro and dr <= tolerancia_raio:
                duplicado = True
                break

        if not duplicado:
            filtrados.append(c)

    return filtrados


def detectar_circulos_multiescala(img_para_hough, edges, gray):
    """
    Executa HoughCircles com múltiplas configurações para melhorar
    a detecção de círculos pequenos, médios e grandes.
    """
    candidatos = []

    configuracoes = [
        {"dp": 1.1, "minDist": 15, "param1": 110, "param2": 16, "minRadius": 4,  "maxRadius": 30},
        {"dp": 1.2, "minDist": 20, "param1": 120, "param2": 20, "minRadius": 8,  "maxRadius": 60},
        {"dp": 1.2, "minDist": 30, "param1": 130, "param2": 24, "minRadius": 20, "maxRadius": 120},
        {"dp": 1.3, "minDist": 40, "param1": 140, "param2": 28, "minRadius": 40, "maxRadius": 0},
    ]

    for cfg in configuracoes:
        circulos = cv2.HoughCircles(
            img_para_hough,
            cv2.HOUGH_GRADIENT,
            dp=cfg["dp"],
            minDist=cfg["minDist"],
            param1=cfg["param1"],
            param2=cfg["param2"],
            minRadius=cfg["minRadius"],
            maxRadius=cfg["maxRadius"]
        )

        if circulos is None:
            continue

        circulos = np.round(circulos[0, :]).astype(int)

        for (x, y, r) in circulos:
            if r <= 0:
                continue

            conf = calcular_confianca_circulo(edges, gray, x, y, r)

            candidatos.append({
                "x": int(x),
                "y": int(y),
                "r": int(r),
                "conf": float(conf)
            })

    return candidatos


# =========================
# PROCESSAMENTO EM LOTE
# =========================
arquivos_imagem = [
    arquivo for arquivo in os.listdir(PASTA_ENTRADA)
    if arquivo.lower().endswith(EXTENSOES_VALIDAS)
]

if not arquivos_imagem:
    print(f"Nenhuma imagem encontrada na pasta '{PASTA_ENTRADA}'.")
    raise SystemExit

print(f"Foram encontradas {len(arquivos_imagem)} imagem(ns).\n")

for nome_arquivo in arquivos_imagem:
    caminho_imagem = os.path.join(PASTA_ENTRADA, nome_arquivo)
    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        print(f"Não foi possível abrir '{caminho_imagem}'. Pulando arquivo.\n")
        continue

    nome_base = os.path.splitext(nome_arquivo)[0]
    resultado = imagem.copy()

    # 1. tons de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # 2. melhoria de contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # 3. suavização
    blur = cv2.GaussianBlur(gray_clahe, (BLUR_KERNEL, BLUR_KERNEL), 0)

    # 4. bordas para avaliação da confiança
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # 5. detecção multiescala com HoughCircles
    candidatos = detectar_circulos_multiescala(blur, edges, gray_clahe)

    # 6. remover duplicados
    circulos_unicos = remover_circulos_duplicados(candidatos)

    # 7. filtrar por confiança mínima
    circulos_finais = [c for c in circulos_unicos if c["conf"] >= CONFIANCA_MINIMA]

    # 8. desenhar resultado
    for c in circulos_finais:
        x, y, r = c["x"], c["y"], c["r"]
        conf = c["conf"]

        # contorno vermelho
        cv2.circle(resultado, (x, y), r, (0, 0, 255), 2)

        # centro vermelho
        cv2.circle(resultado, (x, y), 2, (0, 0, 255), 3)

        # texto
        texto = f"Conf: {conf:.2f}"
        tx = max(10, x - r)
        ty = max(20, y - r - 8)

        cv2.putText(
            resultado,
            texto,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # =========================
    # SALVAMENTO
    # =========================
    arquivo_gray = os.path.join(PASTA_SAIDA, f"{nome_base}_gray.jpg")
    arquivo_edges = os.path.join(PASTA_SAIDA, f"{nome_base}_edges.jpg")
    arquivo_resultado = os.path.join(PASTA_SAIDA, f"{nome_base}_resultado_circle.jpg")

    cv2.imwrite(arquivo_gray, gray_clahe)
    cv2.imwrite(arquivo_edges, edges)
    cv2.imwrite(arquivo_resultado, resultado)

    # log
    print(f"Imagem processada: {nome_arquivo}")
    print(f"  Círculos finais detectados: {len(circulos_finais)}")
    print(f"  - {arquivo_gray}")
    print(f"  - {arquivo_edges}")
    print(f"  - {arquivo_resultado}")

    if circulos_finais:
        print("  Círculos detectados:")
        for i, c in enumerate(circulos_finais, start=1):
            print(
                f"    {i}. centro=({c['x']}, {c['y']}), raio={c['r']} px, confiança={c['conf']:.2f}"
            )
    else:
        print("  Nenhum círculo detectado com confiança suficiente.")

    print()

print("Processamento em lote concluído.")
