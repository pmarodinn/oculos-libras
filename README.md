# Redes Neurais Convolucionais - LIBRAS 🖐 ✋ ✊ ☝

## **Projeto de Reconhecimento de Gestos em LIBRAS com Óculos Automatizado**
Este projeto faz parte de um sistema maior que envolve a construção de um óculos automatizado equipado com câmera e saída de som. O objetivo é traduzir gestos da Língua Brasileira de Sinais (LIBRAS) para áudio de forma offline e o mais próximo possível do tempo real.

---

## **Objetivo**
O projeto implementa uma arquitetura CNN para reconhecer gestos das letras do alfabeto em LIBRAS. A arquitetura segue o padrão:

```
INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT
```

Além disso, o sistema gera saída de áudio fonético para a letra reconhecida, permitindo feedback imediato ao usuário.

---

## **Requisitos**

### **Dependências de Software**
- Python 3.9+
- Conda ou Miniconda (para gerenciamento de ambiente virtual)
- Bibliotecas Python:
  - TensorFlow 2.x
  - OpenCV
  - NumPy
  - pyttsx3 (para texto para fala)
  - Outras dependências listadas no arquivo `environment.yml`.

### **Dependências de Sistema**
- **Linux/Mac/Windows**: Certifique-se de que os seguintes pacotes estão instalados no seu sistema:
  - `espeak` (para reprodução de áudio):
    ```bash
    sudo apt-get update
    sudo apt-get install espeak
    ```
  - Webcam funcional conectada ao sistema ou câmera integrada ao óculos.

---

## **Instalação**

### **1. Clone o Repositório**
Clone o repositório para o seu ambiente local:
```bash
git clone https://github.com/seu_usuario/nome_do_repositorio.git
cd nome_do_repositorio
```

### **2. Crie o Ambiente Virtual**
Use o arquivo `environment.yml` para criar o ambiente virtual com todas as dependências necessárias:
```bash
conda env create -f environment.yml
```

### **3. Ative o Ambiente**
Ative o ambiente virtual criado:
```bash
conda activate cnn_libras
```

### **4. Instale Dependências Adicionais**
Se necessário, instale pacotes adicionais fora do `environment.yml` (por exemplo, `espeak` no Linux):
```bash
sudo apt-get install espeak
```

---

## **Execução**

### **1. Treinamento do Modelo**
Para treinar o modelo usando o dataset fornecido:
```bash
python main/train.py
```
Certifique-se de que o dataset esteja organizado corretamente nas pastas `dataset/training` e `dataset/test`.

### **2. Execução em Tempo Real**
Para testar o modelo em tempo real com sua webcam ou câmera integrada ao óculos:
```bash
python main/app_64x64x3.py
```
O sistema exibirá a imagem capturada pela câmera.
Quando uma letra for reconhecida, o som fonético será emitido automaticamente.

### **3. Teste com Imagens Estáticas**
Você também pode testar o modelo com imagens estáticas:
```bash
python main/app_imgpath.py /caminho/para/imagem.png
```

---

## **Estrutura do Projeto**

```
├── dataset/               # Dataset e scripts para processamento de imagens
│   ├── pre-processed/     # Imagens pré-processadas (treinamento e teste)
│   ├── training/          # Dataset de treinamento
│   ├── test/              # Dataset de teste
│   ├── resize_img.py      # Script para redimensionar imagens
│   └── capture.py         # Script para capturar imagens da câmera
├── demo/                  # Demonstração do projeto (GIFs, vídeos, etc.)
├── logs/                  # Logs de execução (histórico de treinamento, etc.)
├── main/                  # Código principal do projeto
│   ├── cnn/               # Implementação da CNN
│   │   └── __init__.py    # Definição da arquitetura da CNN
│   ├── train.py           # Script para treinar o modelo
│   ├── app_64x64x3.py     # Script para reconhecimento em tempo real
│   └── app_imgpath.py     # Script para reconhecimento com imagens estáticas
├── models/                # Modelos treinados e gráficos
│   ├── graphics/          # Gráficos gerados durante o treinamento
│   ├── image/             # Representações visuais da arquitetura da CNN
│   └── model_epoch_61.h5  # Exemplo de modelo treinado
├── temp/                  # Arquivos temporários (imagens capturadas pela câmera)
├── environment.yml        # Arquivo para criar o ambiente Conda
└── README.md              # Este arquivo
```

---

## **Referências**

### **Documentação e Tutoriais**
- **Arquitetura CNN**: [CS231n - Convolutional Neural Networks](http://cs231n.stanford.edu/)
- **Keras**: [Documentação Oficial do Keras](https://keras.io/)
- **TensorFlow**: [Documentação Oficial do TensorFlow](https://www.tensorflow.org/)

### **Ferramentas Usadas**
- **OpenCV**: [Documentação Oficial do OpenCV](https://opencv.org/)
- **pyttsx3**: [Texto para Fala em Python](https://pyttsx3.readthedocs.io/en/latest/)
- **Conda**: [Gerenciamento de Ambientes com Conda](https://docs.conda.io/en/latest/)

---


