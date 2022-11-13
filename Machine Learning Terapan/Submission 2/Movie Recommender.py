#!/usr/bin/env python
# coding: utf-8

# # Acknowledgements
# ---
# 
# Dataset ini diambil dari website [kaggle.com](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

# # Import Libraries
# ---

# In[1]:


# data manipulation
import pandas as pd
import numpy as np

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import express as px
from plotly import graph_objects as go
sns.set_style('whitegrid')

# data preprocessing
from ast import literal_eval

# modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


# # Data Loading
# ---

# In[2]:


metadatas = pd.read_csv('datasets/movies_metadata.csv', low_memory=False)
links = pd.read_csv('datasets/links_small.csv')
ratings = pd.read_csv('datasets/ratings_small.csv')


# Pertama kita akan melihat 5 data teratas dari dataset movies_metadata dan memunculkan informasi dari dataset

# In[3]:


metadatas.head()


# In[4]:


metadatas.info()


# Selanjutnya kita akan melihat 5 data teratas dari dataset link_small dan memunculkan informasi dari dataset

# In[5]:


links.head()


# In[6]:


links.info()


# Terakhir kita akan melihat 5 data teratas dari dataset ratings_small dan memunculkan informasi dari dataset

# In[7]:


ratings.head()


# In[8]:


ratings.info()


# # Exploratory Data Analysis
# ---
# 
# Pada proyek ini kita akan membuat 2 tipe model sistem rekomendasi yaitu Content Based Filtering dan Collaborative Filtering. Maka dari itu kita memerlukan 3 dataset yaitu movies_metadata, links_small, dan ratings_small.

# ##### movies_metadata
# * adult : merupakan keterangan apakah film termasuk film dewasa atau tidak
# * belongs_to_collection : merupakan koleksi dari film
# * budget : merupakan anggaran yang dibutuhkan untuk membuat film
# * genres : merupakan genre dari film
# * homepage : merupakan link website dari film
# * id : merupakan id dari film
# * imdb_id : merupakan id dari film di website imdb
# * original_language : merupakan bahasa asli dari film
# * original_title : merupakan judul awal dari film
# * overview : merupakan gambaran singkat dari film
# * popularity : merupakan popularitas dari film
# * poster_path : merupakan link poster dari film
# * production_companies : merupakan perusahaan yang memproduksi film
# * production_countries : merupakan negara dimana film di buat
# * release_date : merupakan tanggal film dirilis
# * revenue : merupakan pendapatan dari flim
# * runtime : merupakan durasi dari film
# * spoken_languages : merupakan bahasa yang digunakan pada film
# * status : merupakan status dari film (sudah dirilis atau belum)
# * tagline : merupakan tagline dari film
# * title : merupakan judul film saat ini
# * video : apakah film memiliki video atau tidak
# * vote_average : merupakan rata2 vote film
# * vote_count : merupakan jumlah vote dari film

# In[9]:


print(f'The dataset movies_metadata has {metadatas.shape[0]} records and {metadatas.shape[1]} column')


# In[10]:


metadatas.describe()


# ##### links_small
# * movieId : merupakan id dari film
# * imdbId : merupakan id dari film di website imdb
# * tmdbId : merupakan id dari film di website tmdb

# In[11]:


print(f'The dataset links_small has {links.shape[0]} records and {links.shape[1]} column')


# In[12]:


links.describe()


# ##### ratings_small
# * userId : merupakan id dari user
# * movieId : merupakan id dari film
# * rating : merupakan rating yang diberikan user
# * timestamp : merupakan waktu kapan rating diberikan

# In[13]:


print(f'The dataset ratings_small has {ratings.shape[0]} records and {ratings.shape[1]} column')


# In[14]:


ratings.describe()


# ### Data Visualization

# In[15]:


# membuat temporary dataframe untuk exploratory
temp_metadata = metadatas.drop(['belongs_to_collection', 'homepage', 'imdb_id', 'overview', 'poster_path', 'tagline'], axis=1)
temp_rating = ratings.dropna()


# In[16]:


temp_metadata.head()


# In[17]:


temp_rating.head()


# ##### Data Distribution

# Pertama kita akan melihat persebaran distribusi film yang termasuk kategori adult atau tidak. Dapat kita lihat dari visualisasi di bawah, ternyata terdapat data yang tidak valid karena tidak merepresentasikan apakah film tersebut termasuk film dewasa atau tidak. Dapat kita simpulkan dari visualisasi di bawah, sebesar 99.9% film bukan merupakan film dewasa

# In[18]:


def plot_distribution(series, title):
    """
    This function used to plot data distribution given pandas Series
    
    Parameters
    ----------
    series: pandas Series
    title: title for the plot
    
    Return
    ----------
    None
    """
    
    fig = px.pie(series, values=series.values, names=series.index, title=title)
    fig.show()


# In[19]:


is_adult = temp_metadata['adult'].value_counts()
plot_distribution(is_adult, title='Adult Movies Distribution')


# Selanjutnya kita akan melihat top 20 dari bahasa suatu film. Dapat kita lihat berdasarkan visualisasi di bawah sebagian besar dari film menggunakan bahasa english sebanyak 72.9%

# In[20]:


language = temp_metadata['original_language'].value_counts()[:20]
plot_distribution(language, title='Language of Movies Distribution')


# Untuk melihat distribusi yang terakhir kita akan melihat produksi film per tahun hingga saat ini. Dari visualisasi di bawah sebagian besar film diproduksi pada tahun 2014 dengan jumlah 1974 film.

# In[21]:


temp_metadata['year'] = pd.to_datetime(temp_metadata['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
year = temp_metadata['year'].value_counts()
fig = px.histogram(year, x=year.index, y=year.values, title='Movie Production per Year')
fig.show()


# ##### Checking for missing value

# In[22]:


print('Total missing value metadatas in dataframe:', metadatas.isnull().sum().sum(), 'records')


# In[23]:


print('Column with missing value in metadatas dataframe:', [col for col in metadatas.columns if metadatas[col].isnull().any()])


# In[24]:


print('Total missing value links in dataframe:', links.isnull().sum().sum(), 'records')


# In[25]:


print('Column with missing value in links dataframe:', [col for col in links.columns if links[col].isnull().any()])


# In[26]:


print('Total missing value ratings in dataframe:', ratings.isnull().sum().sum(), 'records')


# # Data Preparation
# ---

# Yang pertama kita lakukan adalah mengubah genres film kedalam bentuk list biasa. Karena saat ini genres film masih berbentuk json object.

# In[27]:


metadatas['genres'] = metadatas['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
metadatas.head()


# In[28]:


# mengambil tahun perilisan film
metadatas['year'] = pd.to_datetime(metadatas['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
metadatas.head()


# In[29]:


# dropping invalid row
metadatas = metadatas.drop([19730, 29503, 35587])
metadatas['id'] = metadatas['id'].astype('int')
links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')
smd = metadatas[metadatas['id'].isin(links)]
smd.head()


# # Data Preprocessing
# ---

# In[30]:


smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
smd.head()


# # Modeling
# ---
# 
# Untuk proyek ini kita akan menggunakan 2 pendekatan sistem rekomendasi yaitu Content Based dan Colaborative Filtering.

# ### Content Based Filtering
# 
# Untuk pendekatan pertama kita akan menggunakan kesamaan deskripsi dari film untuk membuat rekomendasi. Dengan menggunakan Cosine Similarity Score dari dot product TfidfVectorier.

# In[31]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])


# In[32]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[33]:


smd = smd.reset_index()
titles = smd[['title', 'vote_average']]
indices = pd.Series(smd.index, index=smd['title'])


# In[34]:


def content_recommendations(title, top_n=10):
    """
    This function is used for getting movies recommendation based on other movie given.
    
    Parameters
    ----------
    title: title of the movie
    top_n (optional): total recommendation that will be taken
    
    Returns
    ----------
    pandas Series: top n movie recommendation
    """
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    return titles.iloc[movie_indices][:top_n], scores[:top_n]


# In[35]:


recommendation, scores = content_recommendations('The Godfather')
recommendation['similarity_score'] = scores
recommendation


# ### Colaborative User Filtering
# 
# Untuk pendekatan ini kita memerlukan rating dari user karena kita memerlukan preferensi dari user suka film seperti apa. Pendekatan ini kita akan menggunakan library dari Surprise yaitu Singular Value Decomposition (SVD) untuk meminimalisir RMSE yang dapat memberikan rekomendasi lebih baik

# In[36]:


reader = Reader()


# In[37]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# In[38]:


svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)


# In[39]:


def cast_to_int(num):
    """
    This function will cast string number to integer.
    
    Parameters
    ----------
    num: the number that want to be converted
    
    Returns
    ----------
    if success return converted num
    if failed return np.nan
    """
    
    try:
        return int(num)
    except:
        return np.nan


# In[40]:


# appling cast_to_int function to links_small dataset
id_map = pd.read_csv('datasets/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(cast_to_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')


# In[41]:


def collab_recommendation(userId, title, top_n=10):
    """
    This function is used for getting movies recommendation based on user preferences.
    
    Parameters
    ----------
    userId: id of user want to get recommendation
    title: title of the movies that want to get recommendation
    top_n (optional): total recommendation that will be taken
    
    Returns
    ----------
    pandas DataFrame: top_n movies recommendation
    """
    
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(top_n)


# In[42]:


collab_recommendation(1, 'Avatar')

