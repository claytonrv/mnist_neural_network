import manipulador_de_rede_neural as mrn


images_treinamento = None 
imagens_teste = None
labels_treinamento = None
labels_teste = None
labels_treino_one_hot = None
labels_teste_one_hot = None
tamanho_pixel_imagens = None
pixels_da_imagem = None 
numero_de_labels_diferentes = None

mrn.transformar_dados()
images_treinamento, imagens_teste, labels_treinamento, labels_teste, labels_treino_one_hot, labels_teste_one_hot, tamanho_pixel_imagens, pixels_da_imagem, numero_de_labels_diferentes = mrn.configurar_rede()

rede_treinada = mrn.treinar_modelo(images_treinamento, labels_treinamento, pixels_da_imagem, labels_treino_one_hot)

mrn.testar_modelo(rede_treinada, imagens_teste, labels_teste)