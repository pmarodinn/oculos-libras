"""
@author: Pedro Schuves Marodin, João Victor Castex, Rafael Merchiori

Descrição:
Este script utiliza um modelo CNN treinado para reconhecer letras da Língua Brasileira de Sinais (LIBRAS) 
a partir de capturas de câmera em tempo real. Além disso, emite o som fonético da letra reconhecida.

Dependências:
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
- pyttsx3 (para geração de áudio)
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3  # Biblioteca para texto para fala

def nothing(x):
    pass

# Dimensões da imagem de entrada
image_x, image_y = 64, 64

# Carregar o modelo treinado
classifier = load_model('../models/cnn_model_LIBRAS_20190606_0106.h5')

# Definir o número de classes e o mapeamento de classes para letras
classes = 21
letras = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F',
    '6': 'G', '7': 'I', '8': 'L', '9': 'M', '10': 'N', '11': 'O',
    '12': 'P', '13': 'Q', '14': 'R', '15': 'S', '16': 'T', '17': 'U',
    '18': 'V', '19': 'W', '20': 'Y'
}

# Inicializar o motor de TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidade do áudio (palavras por minuto)

def predictor():
    """
    Realiza a previsão da imagem usando o modelo carregado.
    """
    # Carregar e pré-processar a imagem
    img_path = '../temp/img.png'
    test_image = cv2.imread(img_path)
    test_image = cv2.resize(test_image, (image_x, image_y))  # Redimensionar
    test_image = np.array(test_image, dtype=np.float32) / 255.0  # Normalizar para [0, 1]
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

# Função para emitir o som fonético
def speak(letter):
    """
    Emite o som fonético da letra fornecida.
    :param letter: Letra a ser falada.
    """
    engine.say(letter)
    engine.runAndWait()

# Inicializar a câmera
cam = cv2.VideoCapture(0)

img_text = ['', '']
last_spoken_letter = None  # Variável para evitar repetição de áudio

while True:
    ret, frame = cam.read()
    if not ret:
        print("Falha ao capturar a imagem da câmera.")
        break

    frame = cv2.flip(frame, 1)  # Espelhar a imagem

    # Desenhar um retângulo na área de interesse
    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

    # Recortar a área de interesse
    imcrop = img[102:298, 427:623]

    # Exibir texto com a previsão
    cv2.putText(frame, str(img_text[1]), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))

    # Exibir as janelas
    cv2.imshow("test", frame)
    cv2.imshow("mask", imcrop)

    # Salvar a imagem recortada temporariamente
    img_name = "../temp/img.png"
    save_img = cv2.resize(imcrop, (image_x, image_y))
    cv2.imwrite(img_name, save_img)

    # Fazer a previsão
    img_text = predictor()
    predicted_letter = img_text[1]

    # Emitir o som fonético apenas se a letra mudar
    if predicted_letter != last_spoken_letter:
        speak(predicted_letter)
        last_spoken_letter = predicted_letter  # Atualizar a última letra falada

    print(f"Previsão: {img_text[0]} -> Classe: {predicted_letter}")

    # Encerrar o loop se a tecla ESC for pressionada
    if cv2.waitKey(1) == 27:  # 27 é o código ASCII para ESC
        break

# Liberar a câmera e fechar todas as janelas
cam.release()
cv2.destroyAllWindows()