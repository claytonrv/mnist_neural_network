import numpy as np
from scipy.stats import truncnorm

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

funcao_de_ativacao = sigmoid

def truncar_dados(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class RedeNeuralNumeros:
    '''
    Método de inicialização da rede neural.
    Recebe o número de nodos para cada camada (Camada inicial, camadas escondidas e camada de saída),
    além da taxa de aprendizagem.
    '''
    def __init__(self, num_nodos_entrada, num_nodos_saida, num_nodos_escondidos, taxa_aprendizagem):
        self.nodos_primeira_camada = num_nodos_entrada
        self.nodos_ultima_camada = num_nodos_saida
        self.nodos_escondidos = num_nodos_escondidos
        self.taxa_aprendizagem = taxa_aprendizagem 
        self.criar_matizes_peso()
        
    """
    Método para inicialização da matriz de pesos da rede neural 
    """
    def criar_matizes_peso(self):
        rad = 1 / np.sqrt(self.nodos_primeira_camada)
        X = truncar_dados(mean=0, sd=1, low=-rad, upp=rad)
        self.wih = X.rvs((self.nodos_escondidos, self.nodos_primeira_camada))
        rad = 1 / np.sqrt(self.nodos_escondidos)
        X = truncar_dados(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.nodos_ultima_camada, self.nodos_escondidos))

    '''
    Método de treino da rede neural.
    Recebe dois vetores, um com os valores para treino e o segundo com os valores de saída esperados.
    Ambos os vetores, de entrda e o esperado (validação), podem ser tuplas, listas
    ou um ndarray
    '''
    def treinar_rede(self, vetor_entrada, vetor_saida_esperado):
        vetor_entrada = np.array(vetor_entrada, ndmin=2).T
        vetor_saida_esperado = np.array(vetor_saida_esperado, ndmin=2).T
        saida_primeira_camada = np.dot(self.wih, vetor_entrada)
        saida_camada_escondida1 = funcao_de_ativacao(saida_primeira_camada)
        saida_camada_escondida2 = np.dot(self.who, saida_camada_escondida1)
        saida_rede = funcao_de_ativacao(saida_camada_escondida2)
        errors_saida = vetor_saida_esperado - saida_rede
        # Atualização dos pesos da rede:
        tmp = errors_saida * saida_rede * (1.0 - saida_rede)     
        tmp = self.taxa_aprendizagem  * np.dot(tmp, saida_camada_escondida1.T)
        self.who += tmp
        # Cálculo dos erros das camadas escondidas:
        erros_camadas_escondiddas = np.dot(self.who.T, errors_saida)
        # Atualização dos pesos:
        tmp = erros_camadas_escondiddas * saida_camada_escondida1 * (1.0 - saida_camada_escondida1)
        self.wih += self.taxa_aprendizagem * np.dot(tmp, vetor_entrada.T)

    '''
    Método de predição da rede neural.
    Recebe um vetor representando imagem do número para predição.
    '''
    def run(self, vetor_entrada):
        # O vetor de entrada pode ser uma tupla, lista ou um ndarray
        vetor_entrada = np.array(vetor_entrada, ndmin=2).T
        vetor_saida = np.dot(self.wih, vetor_entrada)
        vetor_saida = funcao_de_ativacao(vetor_saida)
        vetor_saida = np.dot(self.who, vetor_saida)
        vetor_saida = funcao_de_ativacao(vetor_saida)
        return vetor_saida

    '''
    Método para geração da matriz de confusão da rede neural para avaliação do modelo.
    '''        
    def gerar_matriz_confusao(self, data_array, labels):
        mat_conf = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            mat_conf[res_max, int(target)] += 1
        return mat_conf    

    '''
    Método para calculo da precisão do modelo, baseado na matriz de confusão.
    '''
    def calcular_precisao(self, label, matriz_confusao):
        col = matriz_confusao[:, label]
        return matriz_confusao[label, label] / col.sum()
    
    '''
    Método para calculo da taxa de recall do modelo, baseado na matriz de confusão.
    A taxa de recall pode ser resumida como o número de vezes em que a predição não
    estava correta, de acordo com o valor esperado.
    '''
    def calcular_taxa_recall(self, label, matriz_confusao):
        row = matriz_confusao[label, :]
        return matriz_confusao[label, label] / row.sum()
        
    '''
    Método para avaliação do desempenho geral da rede neural.
    O método calcula a relação entre acertos e erros do mesmo.
    '''
    def avaliar_desempenho(self, data, labels):
        acertos, erros = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                acertos += 1
            else:
                erros += 1
        return acertos, erros