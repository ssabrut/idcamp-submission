# -*- coding: utf-8 -*-
"""Fake and real news dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Rl_I1Y8rFwsgiuFSysm63CYdtg93_y7v
"""

from google.colab import drive

drive.mount('/content/drive')

!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset

from zipfile import ZipFile

local_zip = 'fake-and-real-news-dataset.zip'
zip_ref = ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

import pandas as pd
import numpy as np

fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

fake.head()

true.head()

new_fake = fake.drop(['date'], axis=1) # drop unused column
new_fake['label'] = 0 # create new column label with 0 value
new_fake.head()

new_fake.info()

print('The total missing value in fake dataframe:', new_fake.isnull().sum().sum())

new_true = true.drop(['date'], axis=1)
new_true['label'] = 1 # create new column label with 1 value
new_true.head()

new_true.info()

print('The total missing value in true dataframe:', new_true.isnull().sum().sum())

df = pd.concat([new_fake, new_true])
df.head()

new_df = df.sample(frac=1).reset_index(drop=True) # shuffle the row and reset the index
new_df.head()

new_df.tail()

new_df.shape

# Commented out IPython magic to ensure Python compatibility.
from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_style('darkgrid')

new_df['length'] = new_df['text'].apply(len)
new_df.head()

sns.countplot(new_df['label'])

sns.displot(new_df['length'], kde=False)

plt.figure(figsize=(15,8))
plot = sns.countplot(x='subject', hue='label', data=new_df)
plot.set_xticklabels(plot.get_xticklabels(), rotation=45)

import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = stopwords.words('english')
punctuations = list(string.punctuation)
stop_words.extend(punctuations)

import re

def fill_title_in_text(index):
  text = df['title'].iloc[index]
  return text
  
def remove_stopwords(text):
  new_text = []
  for i in text.lower().split():
      word = i.strip()
      if word not in stop_words:
          new_text.append(i)
  return new_text

def denoise_text(text):
  index = text.name
  text = list(text)[0]
  if text == ' ':
    text = fill_title_in_text(index)
  text = re.sub(r'bit\S+', '', text) # removing bit.ly/*
  text = re.sub(r'\([^]]*\)', '', text) # removing parenthesis
  text = re.sub(r'\[[^]]*\]', '', text) # removing third brackets
  text = re.sub(r'([\.\\\+\*\?\[\^\]\$\(\)\{\}\!\<\>\|\:\-\,\"\”\“\‘])', '', text) # removing special characteres from the the words
  text = remove_stopwords(text)
  return ' '.join(text)

new_df['soup'] = (new_df['title'] + new_df['text']).to_frame(0).apply(denoise_text, axis = 1)
new_df.head()

X = new_df['soup']
y = new_df['label']

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

print('Total X_train:', len(X_train))
print('Total X_valid:', len(X_valid))
print('Total y_train:', len(y_train))
print('Total y_valid:', len(y_valid))

print('X_train shape:', X_train.shape)
print('X_valid shape:', X_valid.shape)
print('y_train shape:', y_train.shape)
print('y_valid shape:', y_valid.shape)

sns.countplot(y_train)

sns.countplot(y_valid)

vocab_size = 10000
maxlen = 1000
embedding_dim = 200
pad_trunc_type = 'post'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

train_sequences = tokenizer.texts_to_sequences(X_train)
valid_sequences = tokenizer.texts_to_sequences(X_valid)

padded_train = pad_sequences(train_sequences, maxlen=maxlen, padding=pad_trunc_type, truncating=pad_trunc_type)
padded_valid = pad_sequences(valid_sequences, maxlen=maxlen, padding=pad_trunc_type, truncating=pad_trunc_type)

glove_file = '/content/drive/MyDrive/glove.twitter.27B.200d.txt'

def get_coef(word, *arr):
  return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coef(*o.rstrip().rsplit(' ')) for o in open(glove_file, encoding='utf8'))

embed = np.stack(embedding_index.values())
embed_mean, embed_std = embed.mean(), embed.std()
embed_size = embed.shape[1]

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index))
embedding_matrix = embedding_matrix = np.random.normal(embed_mean, embed_std, (nb_words, embed_size))
for word, i in word_index.items():
  if i >= vocab_size:
    continue
  embedding_vector = embedding_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

model = Sequential([
  Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False),
  LSTM(128, return_sequences=True, dropout=0.2),
  LSTM(64, return_sequences=True, dropout=0.1),
  Dense(32, activation='relu'),
  Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import Callback

class Callback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91:
            print('\nStopping training. Accuracy has reached 91%')
            self.model.stop_training = True

callbacks = Callback()

epochs = 100
batch_size = 256

history = model.fit(
    padded_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(padded_valid, y_valid),
    callbacks=[callbacks]
)

def plot_history(history=history, type_='accuracy'):
  plt.plot(history.history[type_])
  plt.plot(history.history['val_' + type_])
  plt.title('model ' + type_)
  plt.ylabel(type_)
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

plot_history()

plot_history(type_='loss')