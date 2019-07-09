import pickle

def transformar_dados(manipulador_imagens):
    with open("dados/mnist/pickled_mnist.pkl", "bw") as fh:
        dados = (manipulador_imagens.images_treinamento, 
                manipulador_imagens.imagens_teste, 
                manipulador_imagens.labels_treinamento,
                manipulador_imagens.labels_teste,
                manipulador_imagens.labels_treino_one_hot,
                manipulador_imagens.labels_teste_one_hot)
        pickle.dump(dados, fh)