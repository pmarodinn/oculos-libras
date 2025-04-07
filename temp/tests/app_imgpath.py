"""
@author: Pedro Schuves Marodin, João Victor Castex, Rafael Merchiori

Descrição:
Este script utiliza um modelo CNN treinado para reconhecer letras da Língua Brasileira de Sinais (LIBRAS) 
a partir de uma imagem fornecida como argumento. O modelo é carregado a partir de um arquivo `.h5` e realiza 
previsões em imagens processadas.

Dependências:
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
"""

import cv2
import numpy as np
import sys
from tensorflow.keras.models import load_model

# Dimensões da imagem de entrada
image_x, image_y = 64, 64

# Carregar o modelo treinado
classifier = load_model('../../models/cnn_model_LIBRAS_20190528_001136.h5')

# Definir o número de classes e o mapeamento de classes para letras
classes = 21
letras = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F',
    '6': 'G', '7': 'H', '8': 'I', '9': 'L', '10': 'M', '11': 'N',
    '12': 'O', '13': 'P', '14': 'Q', '15': 'R', '16': 'S', '17': 'T',
    '18': 'U', '19': 'V', '20': 'W', '21': 'Y'
}

def predictor(img):
    """
    Realiza a previsão da imagem usando o modelo carregado.
    """
    # Redimensionar e normalizar a imagem
    test_image = cv2.resize(img, (image_x, image_y))
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

def main():
    """
    Função principal para processar a imagem de entrada e realizar a previsão.
    """
    # Verificar se o caminho da imagem foi fornecido como argumento
    if len(sys.argv) < 2:
        print("Erro: Forneça o caminho da imagem como argumento.")
        sys.exit(1)

    # Carregar a imagem
    path_img = sys.argv[1]
    img = cv2.imread(path_img)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem '{path_img}'.")
        sys.exit(1)

    # Redimensionar a imagem
    img = cv2.resize(img, (image_x, image_y))

    # Realizar a previsão
    predict = predictor(img)

    # Exibir os resultados
    print('\n\n===========================\n')
    print('Imagem: ', path_img)
    print('Vetor de resultado: ', predict[0])
    print('Classe: ', predict[1])
    print('\n===========================\n')

if __name__ == "__main__":
    main()