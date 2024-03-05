# Predict_Spam_or_Ham_Email
Previsor de email legítmo ou Spam sobre um dataset proveniente da plataforma Kaggle (Link: https://www.kaggle.com/datasets/venky73/spam-mails-dataset), utilizando-se o algoritmo de Naive Bayes.

O código fornecido realiza uma série de etapas para análise e classificação de emails como spam ou legítimos (ham). Aqui está uma descrição detalhada do código:

Carregamento dos Dados:
Os dados são carregados de um arquivo CSV denominado 'spam_ham_dataset.csv' usando a biblioteca pandas.
A coluna "Unnamed: 0" é removida, presumivelmente porque não contém informações relevantes.

Análise Exploratória dos Dados:
São feitas algumas operações de análise inicial nos dados, como contar a quantidade de emails rotulados como spam e ham.
Os emails legítimos e spam são filtrados para uma análise mais detalhada, embora essas operações estejam atualmente comentadas.

Importação de Bibliotecas Necessárias:
Pandas para trabalhar com estruturação dos dados estatísticamente.
Numpy para fazer uma contagem dos emails ham e spam.
Stopwords é um conjunto de palavras que não agregam conhecimento, ou seja, não farão sentidos na analise e deixará o processamento mais custoso. Ex: preposições.
Strings para trabalhar com funções e cadeia de palavras.
Importam-se bibliotecas essenciais para processamento de linguagem natural (NLP) e aprendizado de máquina.

Download de Recursos do NLTK (Obrigatório):
É necessário fazer o download de recursos específicos do NLTK, como as stopwords uma única vez.
  nltk.download())
  nltk.download('stopwords')

Criação de uma Nova Coluna:
Uma nova coluna chamada 'length' (comprimento) é adicionada ao dataframe para representar o comprimento dos subject dos emails e assim poder trabalhar em cima desses números.
  data_spam_ham['length'] = data_spam_ham['text'].apply(len)

Análise Visual do Tamanho das Mensagens:
Um histograma é plotado para visualizar a distribuição do comprimento dos emails, dividido por spam(1) e ham(0) atravéz de gráficos.
  data_spam_ham.hist(column='length', by='label_num', bins=70,figsize=(15,6)) #Bins para o fator de agrupamento das informações. Figsize para o tamanho 

Pré-processamento de Texto:
É definida uma função para processar o texto dos emails, removendo pontuações e stopwords, e convertendo as palavras para minúsculas.
  "def processaTexto(texto):
    nopunc = [char for char in texto if char not in string.punctuation] #Remoção de pontuações
    nopunc = ''.join(nopunc) #agrupando o que foi desagrupado com o nopunc.
    CleanWords = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')] #Removendo palavras desnecessárias, separando as strings em sub-strings e transformando todos em letras minúsculas.
    return CleanWords"
Aplicando a função processaTexto nos textos do dataset e processando todos os dados.
  data_spam_ham['text'].apply(processaTexto)

Separação dos Dados:
Os previsores (X) e a classe (Y) são separados em variáveis distintas.
  X_spam_ham = data_spam_ham.iloc[:, 1].values #Previsores
  Y_spam_ham = data_spam_ham.iloc[:,2].values #Classe
Os dados são divididos em conjuntos de treinamento e teste usando a função 'train_test_split' do sklearn.
  X_data_spam_ham_treino, X_data_spam_ham_teste, Y_data_spam_ham_treino, Y_data_spam_ham_teste = train_test_split(X_spam_ham, Y_spam_ham, test_size=0.15, random_state=0)
  #Criei as variáveis, chamei a função de train_test_split, passei os paramentros de previsores(x) e classe(y) referente ao dataset, coloquei 15% do dataset para teste e o restante para treino.

Transferindo esses novos conjuntos para o naive bayes usando o conceito de pipeline
Pipeline de Pré-processamento e Classificação:
Um pipeline é criado para encadear várias etapas de processamento de texto e classificação.
As etapas incluem:
  CountVectorizer: para contar as ocorrências de palavras em cada email. Contando o Bag Of Words(Saco de palavras) para definir as palavras que são tipicas de spam ou as típicas de ham.
  TfidfTransformer: para converter contagens de palavras em pesos TF-IDF (Term Frequency-Inverse Document Frequency). Faz com que pegue as palavras que aparecem nos dois conjuntos(ham,spam) não seja tão levada em consideração para a analise final e balanceando essas palavras.
  MultinomialNB: um classificador Naive Bayes multinomial para realizar a classificação.

Treinamento do Modelo:
O pipeline é treinado com os dados de treinamento.
  pipeline.fit(X_data_spam_ham_treino, Y_data_spam_ham_treino)

Avaliação do Modelo:
O modelo treinado é usado para fazer previsões nos dados de teste.
  predict = pipeline.predict(X_data_spam_ham_teste)
A precisão (accuracy) do modelo é calculada e é obtida uma taxa de aproximadamente 92% de acerto total.
  accuracy_score(Y_data_spam_ham_teste, predict)
Um relatório de classificação é gerado, incluindo métricas como precision, recall e f1-score.
  classification_report(Y_data_spam_ham_teste, predict)
Uma matriz de confusão é exibida, fornecendo informações sobre os verdadeiros positivos e negativos, e falsos positivos e negativos.
confusion_matrix(Y_data_spam_ham_teste, predict)

Em resumo, o código realiza desde a preparação dos dados até a avaliação de um modelo de classificação de emails como spam ou legítimos, empregando técnicas de processamento de linguagem natural e aprendizado de máquina.

Adicionando às métricas previamente descritas, os resultados específicos para os emails legítimos (label_num 0) e spam (label_num 1) foram os seguintes:

Precisão (Legítimos): 0.89
Precisão (Spam): 1.00
Recall (Legítimos): 1.00
Recall (Spam): 0.71
F1-score (Legítimos): 0.94
F1-score (Spam): 0.83
Acurácia: 0.92
Essas métricas fornecem uma visão detalhada do desempenho do modelo na classificação de emails legítimos e spam, destacando a precisão, recall, f1-score e a acurácia para cada classe.

Este códido foi feito por mim (Christian Cauã Forte Barreto) com base nos estudos do curso da Udemy: "Machine Learning e Data Science com Python de A a Z", juntamente com meus conhecimentos adquiridos ao longo do tempo pela UFC (Universidade Federal do Ceará).
