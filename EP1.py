### Exercício Programa 1 ######################################################
#  AO PREENCHER ESSE CABEÇALHO COM O MEU NOME E O MEU NÚMERO USP,             #
#  DECLARO QUE SOU O ÚNICO AUTOR E RESPONSÁVEL POR ESSE PROGRAMA.             #
#  TODAS AS PARTES ORIGINAIS DESSE EXERCÍCIO PROGRAMA (EP) FORAM              #
#  DESENVOLVIDAS E IMPLEMENTADAS POR MIM SEGUINDO AS INSTRUÇÕES               #
#  DESSE EP E QUE PORTANTO NÃO CONSTITUEM DESONESTIDADE ACADÊMICA             #
#  OU PLÁGIO.                                                                 #
#  DECLARO TAMBÉM QUE SOU RESPONSÁVEL POR TODAS AS CÓPIAS                     #
#  DESSE PROGRAMA E QUE EU NÃO DISTRIBUI OU FACILITEI A                       #
#  SUA DISTRIBUIÇÃO. ESTOU CIENTE QUE OS CASOS DE PLÁGIO E                    #
#  DESONESTIDADE ACADÊMICA SERÃO TRATADOS SEGUNDO OS CRITÉRIOS                #
#  DIVULGADOS NA PÁGINA DA DISCIPLINA.                                        #
#  ENTENDO QUE EPS SEM ASSINATURA NÃO SERÃO CORRIGIDOS E,                     #
#  AINDA ASSIM, PODERÃO SER PUNIDOS POR DESONESTIDADE ACADÊMICA.              #
#                                                                             #
#  Nome : Gustavo Nogai Saito.                                                #
#  NUSP : 10730021.                                                           #
#  Turma: 2023145.                                                            #
#  Prof.: Ronaldo Fumio Hashimoto.                                            #
###############################################################################
""" programa ep01.py

Simular a diferença entre erro in e erro out a partir de um Universo (pré-estabelecido:
dados X pertencente a R² e Y: {-1,+1} com N pontos)
e um conjunto de hipóteses bem definido e finito!
LEIA O ENUNCIADO COMPLETO NO ARQUIVO EM  PDF

"""

import numpy as np

def standardization(X): # valor:0.5
  ''' 
    Given a np.array X returns X_std: mean = 0, std = 1 (not inplace - pure function)
  '''
  # seu código aqui

  center = np.mean(X, axis=0)
  std = np.std(X, axis=0)

  X_std = []

  for i in range(len(X)):
    xi = (X[i][0] - center[0])/std[0]
    yi = (X[i][1] - center[1])/std[1]

    X_std.append([xi, yi])

  return np.asarray(X_std)
  


def calc_error(Y,Y_hat): # valor:1.0
  ''' 
    Given Y (labels) and Y_hat(predicts), returns normalized error
    Inputs:
      Y: np.array or list
      Y_hat: np.array or list
  '''
  # seu código aqui

  n = len(Y_hat)

  norm_error = 0

  for i in range(n):
    if (Y[i] != Y_hat[i]):
      norm_error += 1

  norm_error /= n

  return norm_error
  


def sampling(N,X,Y,random_state=42): # valor:0.5
  '''
    Given the N #of samples (to sampling), X (np.array) and Y (labels - np.array)
    returns N random samples (X,y) type: np.array
  '''
  # seu código aqui

  np.random.seed(random_state)

  X_sample = []
  Y_sample = []

  for i in range(N):
    index = np.random.randint(len(X))
    X_sample.append(X[index])
    Y_sample.append(Y[index])

  return (np.asarray(X_sample), np.asarray(Y_sample))



def diagonais(X,M,b): # valor:2.5
  '''
    Função Diagonais: retas 45º (coeficiente angular +1 e -1  variando bias 
    um tanto para frente e um tanto para trás - passo do bias (b passado por parâmetro) 
    definido pelo intervalo [-M//4,M//4)

    Sabendo que: 
      x0 * w[0] + x1 * w[1] + bias = 0 e que
      w = [1,1] no caso da reta com inclinação negativa e
      w = [1,-1] no caso da reta com inclinação positiva

    A seguinte ordem deve ser utilizada:
      bias partindo de -(M//4) * b até (M//4) * b (exclusive)
      A reta com inclinação negativa (coef == -1), vetor w = [1,1] (perpendicular a reta), e bias é calculda primeiro 
      e a na sequência reta com inclinação positiva, vetor w = [1,-1], e o mesmo bias.
      Conforme mostrado nos plots!

	parâmetros:
		X: np.array
		M: número de hipóteses do universo (número inteiro) - espera-se um múltiplo de 4
	Retorna 
		predict: np.array de np.array de y_hat, um y_hat para cada hipótese (reta), deve ter tamanho M
   '''
  # seu código aqui

  bias = np.arange(-(M//4)*b, (M//4)*b, b)

  w_neg = [1, 1]
  w_pos = [1, -1]

  predicts = []
  for j in range(len(bias)):

    predicts_neg = []
    predicts_pos = []

    for k in range(len(X)):

      if (np.matmul(X[k], w_neg) + bias[j] >= 0):
        predicts_neg.append(1)
      else:
        predicts_neg.append(-1)
    
      if (np.matmul(X[k], w_pos) + bias[j] >= 0):
        predicts_pos.append(1)
      else:
        predicts_pos.append(-1)

    predicts.append(np.asarray(predicts_neg))
    predicts.append(np.asarray(predicts_pos))

  return np.asarray(predicts)
  


def euclidean_dist(p,q): # valor:0.5
  '''
    Given two points (np.array) returns the euclidean distance between them
  '''
  # seu código aqui

  dist = ((p[0]-q[0])**2 + (p[1]-q[1])**2)**(1/2)

  return dist
  


def egocentric(X,C,r): # valor:2.0
  '''
    Given a dataset X (np.array), C (np.array) are the points that will be used as centers, and a radius r: 
      For each point in C, Creates a circumference c, each center works as an hypothesis, and classify points inside c as +1
      otherwise -1.
      Returns all predicts (an list for each point (used as center) )
  '''
  # seu código aqui

  predicts = []
  for i in range(len(C)):
    predicts_c = []
    for j in range(len(X)):
      if (euclidean_dist(C[i], X[j]) < r):
        predicts_c.append(1)
      else:
        predicts_c.append(-1)
    predicts.append(np.asarray(predicts_c))

  return np.asarray(predicts)



def calc_freq(N,H_set,eps,X,Y,M=100,b=0.05,r=1,random_state = 42): # valor:3.0
  '''
  Given N # of samples(integer), H_set name of the hypotheses set 
  (string <diagonais> or <egocentric> error will be returned otherwise)
  eps: epsilon (abs(error_in - error_out) desired), X from the Universe data (np.array - complete dataset),
  Y is all label from theentire Universe(np.array), M # of hypotheses used if <diagonais> is chosen, 
  B: is the bias used when <diagonais> is chosen, r radius of the circumference if <egocentric> is chosen, 
  random_state to set the seed

  Returns:
    bound: theoretical bound for Pr[abs(error_in - error_out) > eps]
    probs: approximated probability of Pr[abs(error_in - error_out) <= eps] by the frequency 
      (# of occurancies (abs(error_in - error_out) <= eps) / # of hipotheses)
  '''
  # seu código aqui

  X_std = standardization(X)

  X_sampled, Y_sampled = sampling(N, X, Y, random_state)
  X_sampled_std = standardization(X_sampled)

  n_hipoteses = 0

  if (H_set == "diagonais"):
    predicts_in = diagonais(X_sampled_std, M, b)
    predicts_out = diagonais(X_std, M, b)
    n_hipoteses = M

  elif (H_set == "egocentric"):
    predicts_in = egocentric(X_sampled_std, X_sampled_std, r)
    predicts_out = egocentric(X_std, X_sampled_std, r)
    n_hipoteses = len(X_sampled_std)

  else:
    raise TypeError("H_set inválido.")

  probs = 0

  for i in range(n_hipoteses):
    error_in = calc_error(Y_sampled, predicts_in[i])
    error_out = calc_error(Y, predicts_out[i])

    if (abs(error_in - error_out) <= eps):
      probs += 1

  bound = (2 * n_hipoteses * np.exp(-2 * (eps**2) * N))

  probs /= n_hipoteses

  return bound, probs


############## Função principal - não será avaliada. ###########################
def main():
    # você pode criar ou importar um dataset qualquer e testar as funções
    # não importe o matplotlib no moodle porque dá erro pois não tem interface gráfica
    return



############## chamada da função main() ########################################
if __name__ == "__main__":
    main()