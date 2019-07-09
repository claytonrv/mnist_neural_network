import numpy as np
import matplotlib.pyplot as plt
import pickle
import manipulador_de_dataset as md
from manipulador_de_imagens import ImagesHandler
from rede_neural import RedeNeuralNumeros

def transformar_dados():
    manipulador_imagens = ImagesHandler()
    md.transformar_dados(manipulador_imagens)

def configurar_rede():
    with open("dados/mnist/pickled_mnist.pkl", "br") as fh:
        data = pickle.load(fh)
    images_treinamento = data[0]
    imagens_teste = data[1]
    labels_treinamento = data[2]
    labels_teste = data[3]
    labels_treino_one_hot = data[4]
    labels_teste_one_hot = data[5]
    tamanho_pixel_imagens = 28 # Altura e e largura 
    numero_de_labels_diferentes = 10 #  Sequência de números de 0 à 9 (Ex.: 0, 1, 2, ..., 9)
    pixels_da_imagem = tamanho_pixel_imagens * tamanho_pixel_imagens
    return images_treinamento, imagens_teste, labels_treinamento, labels_teste, labels_treino_one_hot, labels_teste_one_hot, tamanho_pixel_imagens, pixels_da_imagem, numero_de_labels_diferentes

def treinar_modelo(images_treinamento, labels_treinamento, pixels_da_imagem, labels_treino_one_hot):
    rede = RedeNeuralNumeros(num_nodos_entrada = pixels_da_imagem, num_nodos_saida = 10, num_nodos_escondidos = 100, taxa_aprendizagem = 0.1)
    for i in range(len(images_treinamento)):
        rede.treinar_rede(images_treinamento[i], labels_treino_one_hot[i])
    acertos, erros = rede.avaliar_desempenho(images_treinamento, labels_treinamento)
    print("Acurácia de treinamento: ", acertos / ( acertos + erros))
    matriz_confusao = rede.gerar_matriz_confusao(images_treinamento, labels_treinamento)
    print(matriz_confusao)
    for i in range(10):
        print("Número: ", i, "Precisão de reconhecimento: ", rede.calcular_precisao(i, matriz_confusao), "Taxa de recall: ", rede.calcular_taxa_recall(i, matriz_confusao))
    return rede

def testar_modelo(rede, imagens_teste, labels_teste):
    for i in range(20):
        res = rede.run(imagens_teste[i])
        print(labels_teste[i], np.argmax(res), np.max(res))
    acertos, erros = rede.avaliar_desempenho(imagens_teste, labels_teste)
    print("Acurácia de teste", acertos / ( acertos + erros))