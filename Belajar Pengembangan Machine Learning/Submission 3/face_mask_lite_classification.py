# -*- coding: utf-8 -*-
"""Face Mask Lite Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YHyaEolF0zmZnQSCYlO3B5c29Xbkgu8X

# Acknowledgement

Dataset ini diambil dari [kaggle.com](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

# Import Dataset
"""

# from google.colab import drive

# drive.mount('/content/drive')

# !cp /content/drive/MyDrive/face-mask-12k-images-dataset.zip /content

# !unzip face-mask-12k-images-dataset.zip

"""Untuk struktur direktori kurang lebih seperti gambar dibawah ini.

![image.png](data:image/png;base64,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)

# Exploratory Data
"""

import os

base_dir = 'Face Mask Dataset'

"""Mengambil 1 gambar dari setiap kelas untuk melihat bentuk gambar"""

sample_img = []
total_image_per_class = []
sample_title = os.listdir(base_dir)

for dir in os.listdir(base_dir):
  img_dir = os.path.join(base_dir, dir)
  total_image_per_class.append(len(os.listdir(img_dir)))
  sample_img.append(os.path.join(img_dir, os.listdir(img_dir)[0]))

# Commented out IPython magic to ensure Python compatibility.
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
# %matplotlib inline
sns.set_style('whitegrid')

fig, ax = plt.subplots(1, 2, figsize=(15,7))
for i in range(2):
  img = cv2.imread(sample_img[i])
  ax[i].title.set_text(sample_title[i])
  ax[i].imshow(img)
fig.tight_layout()

"""Dari sampel gambar yang diambil kita bisa lihat ukurang dari gambar tidak seragam. Hal itu tidak menjadi masalah karena gambar akan kita kompres. Setelah melihat gambar sampel, selanjutnya kita akan melihat distribusi dari gambar pada setiap kelas"""

import pandas as pd

df = pd.DataFrame({
    'label': os.listdir(base_dir),
    'total_images': total_image_per_class
})

df

plt.xticks(rotation=45)
plt.title('The distribution of images', fontsize=15, pad=20)
sns.barplot(x='label', y='total_images', data=df)

"""# Data Preprocessing

Sebelum memasukan data ke dalam model, kita perlu melakukan pra pemrosesan data dengan melakukan image augmentation untuk menghasilkan lebih banyak gambar
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=.2,
    validation_split=.2
)

valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=.2)

img_size = 128
batch_size = 128
num_classes = len(os.listdir(base_dir))
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

valid_generator = valid_datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

"""# Modeling"""

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense

model = Sequential([
    Conv2D(64, 3, activation='relu', input_shape=(img_size, img_size, 3)),
    Conv2D(64, 3, activation='relu'),
    BatchNormalization(),
    Conv2D(128, 3, activation='relu'),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, 3, activation='relu'),
    Conv2D(256, 3, activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(.2),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import Callback

threshold = .98
class Callback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('accuracy') > threshold and logs.get('val_accuracy') > threshold:
      print(f'\nStopping training. Accuracy has reached {threshold * 100}%')
      self.model.stop_training = True

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=valid_generator,
    callbacks=[Callback()]
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
