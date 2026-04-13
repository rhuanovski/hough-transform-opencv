import cv2
import numpy as np

# caminho da imagem
imagem = cv2.imread("circulo.jpg")

if imagem is None:
    raise FileNotFoundError("Não foi possível abrir 'circulo.jpg'.")

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

# salva resultados
cv2.imwrite("1_gray.jpg", gray)
cv2.imwrite("2_bordas.jpg", bordas)
cv2.imwrite("3_resultado_hough.jpg", resultado)

print("Processamento concluído.")
print("Arquivos gerados:")
print("- 1_gray.jpg")
print("- 2_bordas.jpg")
print("- 3_resultado_hough.jpg")
