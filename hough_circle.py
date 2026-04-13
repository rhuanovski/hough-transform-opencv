import cv2
import numpy as np
import math

# caminho da imagem
imagem = cv2.imread("imagem_teste.jpg")

if imagem is None:
    raise FileNotFoundError("Não foi possível abrir 'imagem_teste.jpg'.")

resultado = imagem.copy()

# escala de cinza
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# suavização
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# limiarização automática
_, thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# encontra contornos externos
contornos, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

def classificar_forma(contorno):
    area = cv2.contourArea(contorno)
    if area < 1000:
        return None, 0.0

    perimetro = cv2.arcLength(contorno, True)
    if perimetro == 0:
        return None, 0.0

    aproximacao = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
    vertices = len(aproximacao)

    x, y, w, h = cv2.boundingRect(aproximacao)
    aspecto = w / float(h) if h != 0 else 0

    area_caixa = w * h
    extent = area / float(area_caixa) if area_caixa != 0 else 0

    hull = cv2.convexHull(contorno)
    area_hull = cv2.contourArea(hull)
    solidez = area / float(area_hull) if area_hull != 0 else 0

    circularidade = (4 * math.pi * area) / (perimetro * perimetro)

    # círculo: alta circularidade e caixa quase quadrada
    if circularidade > 0.82 and 0.85 <= aspecto <= 1.15:
        confianca = min(1.0, (circularidade + solidez) / 2)
        if confianca > 0.85:
            return "Circulo", confianca

    # triângulo: 3 vértices e forma convexa/estável
    if vertices == 3:
        confianca = (solidez + extent) / 2
        if confianca > 0.75:
            return "Triangulo", confianca

    # quadrado ou retângulo: 4 vértices
    if vertices == 4:
        confianca = (solidez + extent) / 2
        if confianca > 0.80:
            if 0.90 <= aspecto <= 1.10:
                return "Quadrado", confianca
            else:
                return "Retangulo", confianca

    # pentágono
    if vertices == 5:
        confianca = (solidez + extent) / 2
        if confianca > 0.78:
            return "Pentagono", confianca

    # hexágono
    if vertices == 6:
        confianca = (solidez + extent) / 2
        if confianca > 0.78:
            return "Hexagono", confianca

    return None, 0.0


formas_detectadas = []

for contorno in contornos:
    nome, confianca = classificar_forma(contorno)

    if nome is None:
        continue

    formas_detectadas.append((contorno, nome, confianca))

# desenha apenas formas confiáveis
for contorno, nome, confianca in formas_detectadas:
    x, y, w, h = cv2.boundingRect(contorno)

    if nome == "Circulo":
        cor = (255, 0, 0)
    elif nome == "Triangulo":
        cor = (0, 0, 255)
    elif nome in ["Quadrado", "Retangulo"]:
        cor = (0, 255, 255)
    elif nome == "Pentagono":
        cor = (255, 0, 255)
    elif nome == "Hexagono":
        cor = (255, 255, 0)
    else:
        cor = (200, 200, 200)

    cv2.drawContours(resultado, [contorno], -1, cor, 2)
    cv2.putText(
        resultado,
        f"{nome} ({confianca:.2f})",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        cor,
        2
    )

# salva resultados
cv2.imwrite("1_gray.jpg", gray)
cv2.imwrite("2_thresh.jpg", thresh)
cv2.imwrite("3_resultado_formas.jpg", resultado)

print("Processamento concluído.")
print("Arquivos gerados:")
print("- 1_gray.jpg")
print("- 2_thresh.jpg")
print("- 3_resultado_formas.jpg")

if formas_detectadas:
    print("\nFormas detectadas com alta confiança:")
    for _, nome, confianca in formas_detectadas:
        print(f"- {nome}: {confianca:.2f}")
else:
    print("\nNenhuma forma foi detectada com confiança suficiente.")
