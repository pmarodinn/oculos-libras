"""
@author: Pedro Schuves Marodin, João Victor Castex, Rafael Merchiori

Descrição:
Este script captura imagens de uma câmera em tempo real para criar conjuntos de dados de treinamento e teste.
As imagens são salvas em pastas específicas para cada classe (letra) e redimensionadas para o tamanho desejado.

Dependências:
- OpenCV (cv2)
- NumPy
- Python 3.x
"""

import cv2
import time
import numpy as np
import os

# Dimensões da imagem de saída
image_x, image_y = 64, 64

# Teclas para controle
ESC = 27  # Código ASCII para ESC
CAPTURE = 32  # Código ASCII para espaço

# Diretórios para salvar as imagens
dir_img_training = './pre-processed/training/'
dir_img_test = './pre-processed/test/'

# Quantidade de imagens para treinamento e teste
QTD_TRAIN = 150
QTD_TEST = 50


def create_folder(folder_name):
    """
    Cria pastas para armazenar imagens de treinamento e teste, se ainda não existirem.
    """
    if not os.path.exists(os.path.join(dir_img_training, folder_name)):
        os.makedirs(os.path.join(dir_img_training, folder_name))
    if not os.path.exists(os.path.join(dir_img_test, folder_name)):
        os.makedirs(os.path.join(dir_img_test, folder_name))


def capture_images(letra, nome):
    """
    Captura imagens da câmera e as salva nos diretórios de treinamento e teste.

    :param letra: Nome da pasta correspondente à classe (letra).
    :param nome: Prefixo para os nomes das imagens.
    """
    create_folder(str(letra))

    cam = cv2.VideoCapture(0)

    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1
    folder = ''

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Falha ao capturar a imagem da câmera.")
            break

        frame = cv2.flip(frame, 1)

        # Desenhar um retângulo na área de interesse
        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

        # Recortar a área de interesse
        result = img[102:298, 427:623]

        # Exibir texto com o contador de imagens
        cv2.putText(frame, f"{folder}: {img_counter}", (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("frame", frame)
        cv2.imshow("result", result)

        key = cv2.waitKey(1)

        # Capturar imagem ao pressionar a tecla ESPAÇO
        if key == CAPTURE:
            if t_counter <= QTD_TRAIN:
                # Salvar imagem no conjunto de treinamento
                img_name = os.path.join(dir_img_training, str(letra), f"{nome}{training_set_image_name}.png")
                save_img = cv2.resize(result, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print(f"{img_name} written!")
                training_set_image_name += 1
                img_counter = training_set_image_name
                folder = "TRAIN"

            elif t_counter > QTD_TRAIN and t_counter <= (QTD_TRAIN + QTD_TEST):
                # Salvar imagem no conjunto de teste
                img_name = os.path.join(dir_img_test, str(letra), f"{nome}{test_set_image_name}.png")
                save_img = cv2.resize(result, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print(f"{img_name} written!")
                test_set_image_name += 1
                img_counter = test_set_image_name
                folder = "TEST"

            t_counter += 1

            # Encerrar o loop após capturar todas as imagens
            if t_counter > (QTD_TRAIN + QTD_TEST):
                print("[INFO] FIM")
                break

        # Encerrar o loop ao pressionar ESC
        if key == ESC:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    letra = input("LETRA: ").strip().upper()  # Nome da pasta (classe)
    nome = input("NOME: ").strip()  # Prefixo para os nomes das imagens
    capture_images(letra, nome)