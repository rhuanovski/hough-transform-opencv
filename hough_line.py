import cv2
import numpy as np
import os

# pasta de entrada e saída
pasta_entrada = "img"
pasta_saida = "resultado_line"

# cria a pasta de saída, se não existir
os.makedirs(pasta_saida, exist_ok=True)

# extensões permitidas
extensoes_validas = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

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

    # cópia para desenhar
    resultado = imagem.copy()

    # escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # suavização para reduzir ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # detector de bordas
    bordas = cv2.Canny(blur, 50, 150)

    # Transformada de Hough probabilística
    linhas = cv2.HoughLinesP(
        bordas,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10
    )

    # desenha as linhas encontradas
    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            cv2.line(resultado, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # nomes dos arquivos de saída
    arquivo_gray = os.path.join(pasta_saida, f"{nome_base}_gray.jpg")
    arquivo_bordas = os.path.join(pasta_saida, f"{nome_base}_bordas.jpg")
    arquivo_resultado = os.path.join(pasta_saida, f"{nome_base}_resultado_hough.jpg")

    # salva resultados
    cv2.imwrite(arquivo_gray, gray)
    cv2.imwrite(arquivo_bordas, bordas)
    cv2.imwrite(arquivo_resultado, resultado)

    print(f"Imagem processada: {nome_arquivo}")
    print(f"  - {arquivo_gray}")
    print(f"  - {arquivo_bordas}")
    print(f"  - {arquivo_resultado}")
    print()

print("Processamento em lote concluído.")
