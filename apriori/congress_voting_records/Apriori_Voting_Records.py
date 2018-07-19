
# coding: utf-8

# In[1]:


#lib para facilitar o uso de dataframes
import pandas as pd
#lib para facilitar alguns cálculos
import numpy as np
#Lib para auxiliar o processo de count
from collections import Counter

import itertools


# In[2]:


#lendo o csv com o dataset pronto
df_voting = pd.read_csv('data/cleaned_voting.csv')


# In[3]:


#Confirmando o formato dos dados
df_voting.head()


# # Regras de associacao

# Regras de associacao sao tecnicas de mineiracao de dados usadas para descobrir associacoes interessantes entre atributos de um determinado banco de dados.
# A definicao classica de regras de associacao foi apresentada em Agrawal, Imielinski, & Swami (1993) and Han & Kamber, (2006)), e e definida como o seguinte:
# 
# Seja T {t1,t2...tn} um conjunto de transacoes e I {i1,i2...in} um conjunto de itens, definimos D como o dado relevante para a tarefa sendo um conjunto de transacoes onde cada T e um conjunto de itens de maneira que T seja um subconjunto de I.
# 
# Uma regra de associacao tem a forma X -> Y, onde X e Y sao conjuntos disjuntos

# Em nosso dataset as transacoes T representao o conjunto de votos de cada representante, e os Itens representao a reposta ao voto para cada projeto.

# O algoritmo aprior é usado para encontrar conjuntos frequentes de items em databases como o nosso.
# Ele funciona com a afirmativa que:
# > Um sub conjunto de um conjunto frequente provavelmente e trambem um conjunto frequente
# 
# Por exemplo:
#    se {i1,i2} é um conjunto frequente entao provavelmente {i1} e {i2} tambem devem ser conjuntos de items frequentes
#    
# - O algoritmo faz varias iteracoes em busca de itens frequentes e usa esses itens frequentes para gerar as regras de associacao
# 
# Abaixo vemos a implementacao do algoritmo passo a passo

# ---

# Antes de comecarmos com o algoritmo precisamos realizar a definicao das metricas

# In[4]:


def get_suporte(qtd_X, qtd_total):
    '''
    suporte = Numero de transacoes que um item aparece
              ---------------------------------------
              Numero total de transacoes
    '''
    return (qtd_X*1.0)/qtd_total

def calc_confidence(I, s):
    sup_s = gera_suporte_combinacoes(df_voting,[list(s)],SUP_THRESHOLD, create_rules=True)
    sup_i = gera_suporte_combinacoes(df_voting,[list(I)],SUP_THRESHOLD, create_rules=True)
    print
    return (sup_i*1.0) /sup_s
    


# ## Passo 1  e  2

# > Passo 1. No primeiro passo vamos criar uma tabela com todos os counts individuais de cada votacao
# positiva para cada item, calculando tambem seu suporte
# 
# > Passo 2. No segundo passo vamos apenas ficar com suportes acima de um determinado threshold para dar continuidade ao algoritmo

# In[5]:


SUP_THRESHOLD = 0.30
CONF_TRESHOLD = 0.80


# In[6]:


def itens_distintos(df,SUP_THRESHOLD):
    '''
    1. Conta a quantidade de y e n em todas as colunas, pegando apenas y no final
    e Transpondo para melhorar o entendimento da tabela
    2. Adiciona uma coluna com o suporte calculado
    3. Passa o threshold de suporte retornando apenas os itens acima do suporte
    4. Retorna uma lista dos itens filtrados no passo 3
    '''
    #1.
    countings = pd.DataFrame(df.apply(pd.Series.value_counts).T['y'])
    #2.
    countings['suporte'] = countings.apply(lambda k: get_suporte(k['y'], df_voting.shape[0]),axis=1)
    #3.
    countings = countings.loc[countings['suporte'] >= SUP_THRESHOLD]
    return list(countings.index)


# ## Passos 3 e 4

# > Passo 3 - a partir da lista gerada anteriormente geramos todas as combinacoes possiveis
# dentro do propio set (ordem nao importando) com numero de combinacoes = K

# In[7]:


def gera_combinacao(candidatos, k):
    '''
    A partir de uma lista gera todas as combinacoes possiveis de tamanho k
    Ex:
    para k = 2
    F3 = {{1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {1, 3, 5}, {2, 3, 4}}.
    Alguns grupos gerados
    1,2,3 e 1,2,4
    1,2,3 e 1,3,4
    1,2,3 e 1,3,5
    ...
    '''
    #geramos todas as combinacoes possiveis de i para o resto da lista
    #usando a funcao combinations de itertools
    return [list(x) for x in itertools.combinations(candidatos, k)]


# In[8]:


def merge_lists(lista):
    '''
    recebe uma lista de listas e transforma em uma lista unica para facilitar a comparacao
    '''
    merged = list(itertools.chain(*lista))
    return merged


# In[9]:


def compare_sets(f1,f2):
    '''
    compara dois sets de dados, retorna True se 
    os dois sets tem exatamente os mesmos items, caso contrario false
    '''
    #vamos verificar se tem apenas um elem diferente 
    #Temos apenas um elem diferente, vamos verificar se e o ultimo
    #se o tamanho dos sets for igual e apenas o ultimo elem diferir
    if (len(f1) == len(f2)) and (list(f1)[-1] != list(f2)[-1]) and ((list(f1)[:-1] == list(f2)[:-1])):
        return True
    else:
        return False
        


# In[10]:


def gera_suporte_combinacoes(df_voting,lista_combinacoes,SUP_THRESHOLD, create_rules=False):
    '''
    Essa funcao equivale as linhas 5 a 9 do slide 26 sobre regras de associacao
    Basicamente a partir de uma serie de candidatos
    itera um a um e mede o suporte, os mantendo apenas se a medicao ficar acima do threshold
    setado
    '''
    qtd_total_votos = df_voting.shape[0]
    combinacoes_uteis = []
    #crio copia do dataframe original
    for combinacao in lista_combinacoes:
        #Apenas garantindo, na primeira iteracao os tipos podem ser diferentes
        if type(combinacao) != list:
            combinacao = [combinacao]
        df = df_voting.copy()
        #Conto para cada coluna a quantidade de vezes que um voto aparece
        #usamos filtros sucessivos para isso
        for col in combinacao:
            df = df.loc[df[col]=='y']
        #Medimos o suporte
        suporte_par = get_suporte(df.shape[0],qtd_total_votos)
        ##Para criacao das regras
        if create_rules:
            return suporte_par
        if suporte_par >= SUP_THRESHOLD:
            combinacoes_uteis.append(combinacao)
    return combinacoes_uteis


# In[13]:


def candidate_gen(conj):
    '''
    Codigo representado no slide 27 (Regras de Associacao)
    '''
    #inicia um conjunto vazio
    Ck = []
    #Geramos todos os pares possiveis do conjunto em uso
    #checar comentario em gera_combinacao()
    todos_os_pares = gera_combinacao(conj, 2)
    #apenas para evitar erro na primeira iteracao pois os pares nao vem como listas
    if type(conj[0]) != list:
        return todos_os_pares
    #para cada para gerado comparamos usando a funcao compare_sets
    #eh a comparacao explicada no slide 28 Passo Join
    #Realiza passo de Juncao
    for f1,f2 in todos_os_pares:
        if compare_sets(f1,f2):
            f1 = set(f1)
            f2 = set(f2)
            #Caso a juncao seja permitida junto os dois conjuntos
            #slide 27, linha 7
            set_to_join = list(f1.union(f2))
            #Apenas um double check para evitar duplicidades no conjunto de candidatos
            if set_to_join not in Ck and len(set_to_join) == len(f1)+1:
                #Append na nova lista de candidatos possiveis
                Ck.append(set_to_join)
                #Passo de Poda
                #Determina se todos os subsets de Ck estao na lista original de pares
                #caso contrario remove da lista
                for subset_ck in gera_combinacao(set_to_join, k=len(todos_os_pares[0][0])):
                    if subset_ck not in merge_lists(todos_os_pares):
                        Ck.remove(set_to_join)
                        break
                        
    return Ck


# In[14]:


#primeira iteracao
candidatos = itens_distintos(df_voting,SUP_THRESHOLD)
Fk = []
#Enquanto tivermos possiveis candidatos
while candidatos:
    #Gera candidatos 
    candidatos = candidate_gen(candidatos)
    #Apenas candidatos com suporte acima do threshold sao selecionados
    valid_candidatos = gera_suporte_combinacoes(df_voting,candidatos,SUP_THRESHOLD)
    #Append na lista final de items frequentes com alto suporte
    Fk.append(valid_candidatos)
    #Atualiza candidatos para os novos serem gerados com base nesses
    candidatos = valid_candidatos


# In[16]:


print("Fk: ", Fk)


# In[17]:


def generate_rules(fk, CONF_TRESHOLD):
    final_rules = []
    for L in range(1, len(fk)):
        for subset in itertools.combinations(fk, L):
            I = set(fk)
            s = set(subset)
            confidence = calc_confidence(I,s)
            if confidence > CONF_TRESHOLD:
                rule = str('{} => {} = {}'.format(s, I-s, confidence))
                final_rules.append(rule)
    return final_rules
            


# In[18]:


print("Todas as regras geradas com confianca maior que {}".format(CONF_TRESHOLD))
all_rules = []
for item in merge_lists(Fk):
    all_rules.append(generate_rules(item,CONF_TRESHOLD))

for item in merge_lists(all_rules):
    print(item)

