"""
@author: Pedro Schuves Marodin, João Victor Castex, Rafael Merchiori

Descrição:
Este script utiliza um modelo CNN treinado para reconhecer letras da Língua Brasileira de Sinais (LIBRAS) 
a partir de capturas de câmera em tempo real. O modelo é carregado a partir de um arquivo `.h5` e realiza 
previsões em imagens processadas.

Dependências:
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

def nothing(x):
    pass

# Dimensões da imagem de entrada
image_x, image_y = 64, 64

# Carregar o modelo treinado
classifier = load_model('../../models/model_epoch_61_98%_leakyRelu.h5')

# Definir o número de classes e o mapeamento de classes para letras
classes = 21
letras = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F',
    '6': 'G', '7': 'H', '8': 'I', '9': 'L', '10': 'M', '11': 'N',
    '12': 'O', '13': 'P', '14': 'Q', '15': 'R', '16': 'S', '17': 'T',
    '18': 'U', '19': 'V', '20': 'W', '21': 'Y'
}

def predictor(test_image):
    """
    Realiza a previsão da imagem usando o modelo carregado.
    """
    # Converter a imagem para escala de cinza
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Redimensionar e normalizar a imagem
    test_image = cv2.resize(test_image, (image_x, image_y))
    test_image = np.array(test_image, dtype=np.float32) / 255.0  # Normalizar para [0, 1]
    test_image = np.expand_dims(test_image, axis=-1)  # Adicionar canal de cor (escala de cinza)
    test_image = np.expand_dims(test_image, axis=0)  # Adicionar dimensão do batch

    # Realizar a previsão
    result = classifier.predict(test_image)

    # Encontrar a classe com maior probabilidade
    maior, class_index = -1, -1
    for x in range(classes):
        if result[0][x] > maior:
            maior = result[0][x]
            class_index = x

    return [result, letras[str(class_index)]]

# Inicializar a câmera
cam = cv2.VideoCapture(0)

# Criar janela para os trackbars
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Criar janela para exibir a captura
cv2.namedWindow("test")

img_text = ['', '']

while True:
    ret, frame = cam.read()
    if not ret:
        print("Falha ao capturar a imagem da câmera.")
        break

    frame = cv2.flip(frame, 1)  # Espelhar a imagem

    # Obter valores dos trackbars
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    # Desenhar um retângulo na área de interesse
    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

    # Recortar a área de interesse
    imcrop = img[102:298, 427:623]

    # Converter para HSV e aplicar máscara
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Exibir texto com a previsão
    cv2.putText(frame, str(img_text[1]), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))

    # Exibir as janelas
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)

    # Redimensionar a máscara para o tamanho esperado pelo modelo
    img_resized = cv2.resize(mask, (image_x, image_y))

    # Fazer a previsão
    img_text = predictor(img_resized)
    print(f"Previsão: {img_text[0]} -> Classe: {img_text[1]}")

    # Encerrar o loop se a tecla ESC for pressionada
    if cv2.waitKey(1) == 27:  # 27 é o código ASCII para ESC
        break

# Liberar a câmera e fechar todas as janelas
cam.release()
cv2.destroyAllWindows()