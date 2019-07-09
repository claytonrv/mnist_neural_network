import numpy as np
import matplotlib.pyplot as plt

class ImagesHandler():
    def __init__(self):
        self.tamanho_pixel_imagens = 28 # Altura e e largura 
        self.numero_de_labels_diferentes = 10 #  Sequência de números de 0 à 9 (Ex.: 0, 1, 2, ..., 9)
        self.pixels_da_imagem = self.tamanho_pixel_imagens * self.tamanho_pixel_imagens
        self.caminho_dataset = "dados/mnist/"
        self.gerar_dados_treinamento_e_teste()

    def gerar_dados_treinamento_e_teste(self):
        dados_treinamento = np.loadtxt(self.caminho_dataset + "mnist_train.csv", delimiter=",")
        dados_teste = np.loadtxt(self.caminho_dataset + "mnist_test.csv", delimiter=",") 
        dados_teste[:10]
        dados_teste[dados_teste==255]
        dados_teste.shape
        fac = 0.99 / 255
        self.images_treinamento = np.asfarray(dados_treinamento[:, 1:]) * fac + 0.01
        self.imagens_teste = np.asfarray(dados_teste[:, 1:]) * fac + 0.01
        self.labels_treinamento = np.asfarray(dados_treinamento[:, :1])
        self.labels_teste = np.asfarray(dados_teste[:, :1])
        self.transformar_dados()

    def transformar_dados(self):
        numeros = np.arange(10)
        for label in range(10):
            representacao_one_hot = (numeros==label).astype(np.int)
            print("Número: ", label, " na representação one-hot: ", representacao_one_hot)
        self.numeros = np.arange(self.numero_de_labels_diferentes)
        # Codifica os labels de representação dos números na notação one hot
        self.labels_treino_one_hot = (self.numeros==self.labels_treinamento).astype(np.float)
        self.labels_teste_one_hot = (self.numeros==self.labels_teste).astype(np.float)
        # Removendo zeros e uns dos labels, se o label é zero troca por 0.01
        # Se o label é um troca por 0.99
        # Esta ação é importante para melhorar os calculos
        self.labels_treino_one_hot[self.labels_treino_one_hot==0] = 0.01
        self.labels_treino_one_hot[self.labels_treino_one_hot==1] = 0.99
        self.labels_teste_one_hot[self.labels_teste_one_hot==0] = 0.01
        self.labels_teste_one_hot[self.labels_teste_one_hot==1] = 0.99

    def apresentar_imagens(self):
        for i in range(10):
            imagem = self.images_treinamento[i].reshape((28,28))
            plt.imshow(imagem, cmap="Greys")
            plt.show()
        for i in range(10):
            img = self.imagens_teste[i].reshape((28,28))
            plt.imshow(img, cmap="Greys")
            plt.show()