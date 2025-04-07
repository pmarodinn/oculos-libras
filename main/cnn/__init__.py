"""
@author: Pedro Schuves Marodin, João Victor Castex, Rafael Merchiori

Descrição:
Este módulo define a classe `Convolucao`, que implementa uma arquitetura de rede neural convolucional (CNN) 
para classificação de imagens. A arquitetura segue o padrão:
    INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT

Dependências:
- TensorFlow 2.x
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU

class Convolucao(object):
    """
    Classe para construir uma arquitetura de rede neural convolucional (CNN).
    """

    @staticmethod
    def build(width, height, channels, classes):
        """
        Constrói o modelo CNN com base nos parâmetros fornecidos.

        :param width: Largura em pixels da imagem.
        :param height: Altura em pixels da imagem.
        :param channels: Quantidade de canais da imagem (1 para escala de cinza, 3 para RGB).
        :param classes: Quantidade de classes para o output.

        :return: Modelo CNN com a seguinte arquitetura:
            INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT
        """
        inputShape = (height, width, channels)

        # Criar o modelo sequencial
        model = Sequential()

        # Primeira camada convolucional + ReLU + MaxPooling
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=inputShape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Segunda camada convolucional + ReLU + MaxPooling
        model.add(Conv2D(filters=32, kernel_size=(3, 3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Terceira camada convolucional + ReLU + MaxPooling
        model.add(Conv2D(filters=64, kernel_size=(3, 3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Achatamento das features
        model.add(Flatten())

        # Camada densa + Dropout
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        # Camada de saída com ativação softmax
        model.add(Dense(classes, activation='softmax'))

        return model