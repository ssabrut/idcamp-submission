#!/usr/bin/env python
# coding: utf-8

# ## Acknowledgement
# ---
# 
# Dataset ini diambil dari [kaggle.com](https://www.kaggle.com/datasets/rajgupta2019/medical-insurance-dataset)

# # Import libraries
# ---
# 

# In[1]:


# data manipulation
import pandas as pd
import numpy as np

# data visualisation
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')

# data preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder

# modeling
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# metrics
from sklearn.metrics import mean_squared_error


# # Data loading
# ---

# In[2]:


df = pd.read_csv('Train_Data.csv')
df.head()


# In[3]:


print('Dataframe memiliki total {} data dan {} kolom'.format(df.shape[0], df.shape[1]))


# Mengecek apakah dataframe mempunya missing value

# In[4]:


df.info()


# In[5]:


print('Dataframe memiliki total {} missing value'.format(df.isnull().sum().sum()))


# # Exploratory data analysis
# ---
# Sebelum masuk ke data preprocessing dan modeling, kita harus mengeksplor data agar kita dapat menemukan gambaran tentang data dan mendapat insight dari data yang dapat digunakan pada proses data preprocessing.
# 
# Dataframe mempunya 7 kolom dengan deskripsi sebagai berikut:
# * age : merupakan umur dari responden
# * sex : merupakan jenis kelamin dari responden
# * bmi : merupakan bmi dari responden (berat badan (kg) / tinggi badan<sup>2</sup> (m))
# * smoker : merupakan keterangan perokok atau tidak dari responden
# * region : merupakan negara responden tinggal
# * children : merupakan berapa banyak anak yang dimiliki responden
# * charges : merupakan biaya yang harus dibayar oleh responden

# In[6]:


df.describe()


# ## Data visualisation

# In[7]:


numerical_col = [col for col in df.columns if df[col].dtype != object]
fig, ax = plt.subplots(1, 2, figsize=(18,7))
sns.boxplot(data=df[numerical_col[:3]], ax=ax[0])
sns.boxplot(data=df[numerical_col[-1]])


# Dapat kita lihat dari visualisasi data di atas terdapat outlier pada fitur bmi dan charges. Biasanya kita dapat menggunakan IQR Method untuk menangani outlier, namun pada dataset kali ini kita tidak akan melakukannya.
# 
# ### Univariate analysis
# Untuk melakukan univariate analysis, kita akan membuat dataframe sementara agar dapat melakukan preprocessing pada data tanpa mengubah dataframe utama

# In[8]:


temp_df = df
temp_df.head()


# In[9]:


rows = 2
cols = 2
fig = plt.figure(figsize=(20,15))
for i, col in enumerate(numerical_col):
  ax = fig.add_subplot(rows, cols, i + 1)
  sns.histplot(x=temp_df[col], bins=30, kde=True, ax=ax)
fig.tight_layout()
plt.show()


# Dari visualisasi di bawah ini dapat disimpulkan bahwa semakin bertambahnya umur tidak terlalu mempengaruhi pada harga asuransi

# In[10]:


plt.title('Correlation between charges vs age', fontsize=15, pad=20)
sns.scatterplot(x='age', y='charges', data=temp_df)


# Dari visualisasi di bawah ini kita mendapatkan insight yang menarik. Karena orang dengan bmi tinggi belum tentu memiliki harga isuransi yang tinggi

# In[11]:


plt.title('Correlation between charges vs bmi', fontsize=15, pad=20)
sns.scatterplot(x='bmi', y='charges', data=temp_df)


# Dari visualisasi di bawah ini dapat kita lihat bahwa orang yang belum memiliki anak cenderung mempunya harga asuransi yang lebih tinggi

# In[12]:


plt.title('Correlation between charges vs children', fontsize=15, pad=20)
sns.scatterplot(x='children', y='charges', data=temp_df)


# Setelah melihat korelasi antara numerical features, selanjutnya kita akan mengeksplor persebaran data pada categorical features

# In[13]:


categorical_col = [col for col in df.columns if df[col].dtype == object]
fig, ax = plt.subplots(1, len(categorical_col), figsize=(25,7))
for i, col in enumerate(categorical_col):
    sns.countplot(data=temp_df, x=col, ax=ax[i])


# Setelah melihat persebaran data secara universal sekarang kita akan melihat persebaran data berdasarkan region

# In[14]:


plt.title('Sex distribution grouped by region', fontsize=15, pad=20)
sns.countplot(data=temp_df, x='sex', hue='region')
plt.legend(loc='lower right')


# Dari persebaran data jenis kelamin di atas, dapat kita lihat bahwa kebanyakan laki laki berasal dari daerah southeast dan kebanyakan perempuan berasal dari northwest

# In[15]:


plt.title('Smoker distribution grouped by region', fontsize=15, pad=20)
sns.countplot(data=temp_df, x='smoker', hue='region')
plt.legend(loc='upper right')


# Dari persebaran data perokok di atas, dapat kita lihat bahwa jumlah orang yang perokok pada tiap daerah memiliki rasio yang lebih sedikit dari pada yang tidak merokok

# In[16]:


plt.title('Children distribution grouped by region', fontsize=15, pad=20)
sns.countplot(data=temp_df, x='children', hue='region')
plt.legend(loc='upper left')


# Dari persebaran data anak di atas, dapat kita lihat bahwa daerah yang memiliki jumlah anak terbanyak terdapat pada daerah southeast. Selanjutnya kita akan melihat persebaran data perokok berdasarkan jenis kelamin

# In[17]:


plt.title('Sex distribution grouped by smoker or not', fontsize=15, pad=20)
sns.countplot(data=temp_df, x='smoker', hue='sex')
plt.legend(loc='upper left')


# Selanjutnya kita akan melihat total anak anak pada setiap daerah

# In[18]:


children = temp_df.groupby('region')[['children']].sum()
plt.title('Total children distribution grouped by region', fontsize=15, pad=20)
sns.barplot(data=children, x=children.index, y='children')


# ### Multivariate analysis
# 
# Selanjutnya kita akan melihat korelasi antar fitur

# In[19]:


sns.pairplot(df[numerical_col], diag_kind='kde')


# Untuk lebih jelasnya kita dapat menggunakan heatmap agar dapat melihat antar fitur memiliki korelasi kuat atau lemah

# In[20]:


plt.title('Correlation matrix for numerical features', fontsize=15, pad=20)
sns.heatmap(df[numerical_col].corr(), annot=True, vmin=-1, vmax=1, cmap='viridis', linewidth=1)


# # Data Preparation
# ---
# 
# Karena terdapat categorical features pada data, maka pertama kita akan melakukan encoding pada categorical features. Kita dapat melakukan encoding dengan menggunakan OrdinalEncoder dari sklearn

# In[21]:


df.head()


# In[22]:


encoder = OrdinalEncoder()
encoded_df = df
encoded_df[categorical_col] = pd.DataFrame(encoder.fit_transform(df[categorical_col]))
encoded_df.head()


# Setelah melakukan encoding pada categorical features, selanjutnya kita akan membulatkan features age. Karena tidak mungkin kan umur desimal

# In[23]:


encoded_df['age'] = encoded_df['age'].apply(np.ceil).astype(np.int64)
encoded_df['bmi'] = encoded_df['bmi'].apply(lambda x: np.round(x, 2))
encoded_df.head()


# Setelah melakukan semua data preprocessing selanjutnya kita akan menentukan dependan dan independen variabel

# In[24]:


X = encoded_df.iloc[:, :-1].values
y = encoded_df.iloc[:, -1].values


# Setelah menentukan variabel independen dan dependen, selanjutnya kita akan melakukan splitting dataset ke dalam train set dan test set

# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)


# In[26]:


print('Total X_train:', len(X_train), 'records')
print('Total y_train:', len(y_train), 'records')
print('Total X_test:', len(X_test), 'records')
print('Total y_test:', len(y_test), 'records')


# In[27]:


print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


# In[28]:


models = pd.DataFrame(columns=['train_mse', 'test_mse'], index=['SVR', 'LinearRegression', 'GradientBoost'])


# # Modeling
# ---
# 
# ### Hyperparameter tuning
# Hyperparameter tuning adalah salah satu teknik yang dilakukan akan model dapat berjalan dengan performa terbaik. Biasanya dalam hyperparameter tuning, hyperparameter akan ditentukan secara acak oleh teknisi. Namun jika tidak ingin mencoba coba hyperparameter mana yang terbaik, kita dapat menggunakan GridSearch. GridSearch merupakan sebuah teknik yang memungkinkan kita untuk menguji beberapa hyperparameter sekaligus pada sebuah model

# In[29]:


def grid_search(model, hyperparameter):
    result = GridSearchCV(
        model,
        hyperparameter,
        cv=5,
        verbose=1,
        n_jobs=6
    )
    
    return result


# In[30]:


svr = SVR()
hyperparameters = {
    'kernel': ['rbf'],
    'C': [0.001, 0.01, 0.1, 10, 100, 1000],
    'gamma': [0.3, 0.03, 0.003, 0.0003]
}

svr_search = grid_search(svr, hyperparameters)
svr_search.fit(X_train, y_train)
print(svr_search.best_params_)
print(svr_search.best_score_)


# In[31]:


linear_regression = LinearRegression()
hyperparameters = {}

linear_search = grid_search(linear_regression, hyperparameters)
linear_search.fit(X_train, y_train)
print(linear_search.best_params_)
print(linear_search.best_score_)


# In[32]:


gradient_boost = GradientBoostingRegressor()
hyperparameters = {
    'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
    'n_estimators': [250, 500, 750, 1000],
    'criterion': ['friedman_mse', 'squared_error']
}

gradient_boost_search = grid_search(gradient_boost, hyperparameters)
gradient_boost_search.fit(X_train, y_train)
print(gradient_boost_search.best_params_)
print(gradient_boost_search.best_score_)


# ### Model Training
# 
# Kita akan melakukan training pada semua model untuk melihat model mana yang memiliki mse terendah

# In[33]:


svr = SVR(C=1000, gamma=.3, kernel='rbf')
svr.fit(X_train, y_train)


# In[34]:


linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)


# In[35]:


gradient_boost = GradientBoostingRegressor(criterion='squared_error', learning_rate=1e-1, n_estimators=500)
gradient_boost.fit(X_train, y_train)


# ### Model Evaluation

# In[36]:


model_dict = {
    'SVR': svr,
    'LinearRegression': linear_regression,
    'GradientBoost': gradient_boost
}

for name, model in model_dict.items():
    models.loc[name, 'train_mse'] = mean_squared_error(y_train, model.predict(X_train))
    models.loc[name, 'test_mse'] = mean_squared_error(y_test, model.predict(X_test))

models.head()


# In[37]:


models.sort_values(by='test_mse', ascending=False).plot(kind='bar', zorder=3)

