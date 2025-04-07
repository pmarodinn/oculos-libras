"""
@author: Pedro Schuves Marodin, João Victor Castex, Rafael Merchiori

Descrição:
Este script redimensiona imagens em um diretório de entrada para um tamanho especificado 
e salva as imagens redimensionadas em um diretório de saída. As imagens são convertidas 
para o formato PNG durante o processo.

Dependências:
- Pillow (PIL)
- Python 3.x
"""

from PIL import Image
import os
import sys

def readf():
    """
    Função principal para redimensionar imagens.
    """
    try:
        # Verificar se os argumentos foram fornecidos corretamente
        if len(sys.argv) < 4:
            print("Erro: Forneça os argumentos corretamente.")
            print("Uso: python resize_img.py <diretório_entrada> <tamanho_imagem> <diretório_saída>")
            sys.exit(1)

        # Ler os argumentos da linha de comando
        input_dir = sys.argv[1]  # Diretório de entrada
        img_size = int(sys.argv[2])  # Tamanho desejado para as imagens (largura e altura)
        output_dir = sys.argv[3]  # Diretório de saída

        print("Iniciando...")
        print(f"Coletando dados de {input_dir}")

        # Listar as classes (subdiretórios) no diretório de entrada
        tclass = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

        counter = 0  # Contador de classes processadas

        for x in tclass:
            # Caminho para a pasta de origem e destino da classe
            list_dir = os.path.join(input_dir, x)
            list_tuj = os.path.join(output_dir, x)

            # Criar o diretório de destino se ele não existir
            if not os.path.exists(list_tuj):
                os.makedirs(list_tuj)

            # Processar cada imagem na pasta de origem
            for d in os.listdir(list_dir):
                try:
                    # Abrir a imagem
                    img_path = os.path.join(list_dir, d)
                    img = Image.open(img_path)

                    # Redimensionar a imagem
                    img = img.resize((img_size, img_size), Image.ANTIALIAS)

                    # Separar o nome do arquivo e a extensão
                    fname, extension = os.path.splitext(d)

                    # Salvar a imagem no formato PNG
                    newfile = f"{fname}.png"
                    output_path = os.path.join(list_tuj, newfile)
                    img.save(output_path, "PNG", quality=90)

                    print(f"Redimensionando arquivo: {x}/{d} -> {newfile}")
                except Exception as e:
                    print(f"Erro ao redimensionar arquivo: {x}/{d}")
                    print(f"Detalhes do erro: {e}")

            counter += 1  # Incrementar o contador de classes processadas

    except Exception as e:
        print("Erro durante o processamento:", e)
        sys.exit(1)

if __name__ == "__main__":
    readf()