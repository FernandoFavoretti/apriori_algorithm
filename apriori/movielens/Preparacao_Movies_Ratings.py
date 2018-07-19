
# coding: utf-8

# In[4]:


# -*- coding: utf-8 -*-


# ## Movie Lens Dataset

# O dataset escolhido é o famoso **movie lens dataset**, contendo diversas avaliações de usuários para diversos filmes. Disponível em diversas plataformas como kaggle, uci etc, para este estudo utilizamos a versao **recommended for education and development**, que é uma versão enxuta daa original que contém mais de 20 milhões de ratings, encontrada no site original do grupo de pesquisa responsável (https://grouplens.org/datasets/movielens/) ou podendo ser baixada diretamente em (http://files.grouplens.org/datasets/movielens/ml-latest-small.zip). Entretanto o dataset também está na pasta /data deste projeto.
# 
# 
# Escolhemos este dataset afim de possivelmente criar um pequeno sistema de recomendação baseado em regras de associacao, O dataset contem inicialmente 4 colunas:
# userId - Id do usuario repsonsavel por uma nota
# movieId  - Id do filme avaliado pelo usuario
# rating - nora dada pelo usuario a um determinado filme
# timestamp - o horario que a nota foi dada
# 
# Usaremos apenas as 3 primeiras colunas para formatar o dado

# In[5]:


#Libs utilizadas
#Pandas para trabalhar com dataframes
#Numpy para calculos
import pandas as pd
import numpy as np


# In[6]:


#lendo csv para dataframe
df_movies = pd.read_csv('data/ratings.csv')


# In[7]:


#Exemplo de dados
print('Exemplo dos dados:')
print(df_movies.head())


# In[8]:


#A primeira coisa a fazer sera criar uma pivot table
#Tornando o user id como linha, o movie id como coluna e o ratings como valor
#tornando o dataset com a cara transacional
df_movies = pd.pivot_table(df_movies, values='rating', index=['userId'],columns=['movieId'], aggfunc=np.max)


# In[9]:

print('Dados pivotados:')
print(df_movies.head())


# In[10]:


#Nosso dataset e muito grande e esparso, depois de varios testes
#prefirimos filtrar apenas os filmes que receberam maiores ratings (para melhorar as regras)
#Um dataset muito esparso impacta demais os valores de threshold de suporte
#por isso vamos filtrar os top 50 filmes
df_counter_ratings = pd.DataFrame(df_movies.sum(axis=0), columns=['sum_ratings'])
top_movies = list(df_counter_ratings.sort_values(by='sum_ratings', ascending=False)[:50].index)


# In[11]:


#Novo head dos dados
print('Apenas top filmes')
df_top_movies = df_movies[top_movies]
print(df_top_movies.head())


# In[12]:


#Iremos transformar em True usuarios que deram ratings maior que 3
#Tomando como verdade que eles 'indicariam' o filme
df_top_movies = df_top_movies.applymap(lambda k: k>=3)


# In[13]:


#Alem do movimento acima, vamos tambem filtrar usuarios que viram pelo menos
#10 filmes, para também melhorar as regras ao final
df_counter_views =  pd.DataFrame(df_top_movies.sum(axis=1), columns=['number_views'])
top_viewers = list(df_counter_views.loc[df_counter_views['number_views'] > 10].index.values)


# In[14]:


#nosso novo dataset contem 50 colunas e 373 linhas
df_top_movies = df_top_movies.loc[top_viewers]
print('Convertendo notas para 1 e 0...')
print(df_top_movies.shape)
print(df_top_movies.head())


# In[15]:


#Por fim, vamos tranformar o dataset no mesmo formato do
#dataset originalmente trabalhdo (Congressional_Voting_Records) para o algoritmo rodar sem problemas
print('Dado final...')
df_top_movies = df_top_movies.replace(True,'y')
df_top_movies = df_top_movies.replace(False,'n')
print(df_top_movies.head(3))

# In[17]:


#Salvamos para csv estamos prontos para rodar o algoritmo
df_top_movies.to_csv("data/cleaned_ratings.csv",index=False)

