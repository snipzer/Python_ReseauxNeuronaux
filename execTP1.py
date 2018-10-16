import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

print('TensorFlow %s, Keras %s, numpy %s'%(tf.__version__,keras.__version__, np.__version__))

## Configuration du réseau

model = keras.Sequential()
# Ajout d'une fonction linéaire.
# conseil : Pour passer en réseau de neurone, changer l'activation et créer plusieurs couches
model.add(keras.layers.Dense(64, activation='linear'))
# Ajout d'une fonction softmax avec 10 unités en sortie:
model.add(keras.layers.Dense(10, activation='softmax'))

## Choix de l'optimisation, de la fonction de cout, et de la métrique

regularisation_parameter = 0.001
model.compile(optimizer=tf.train.GradientDescentOptimizer(regularisation_parameter),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

## Création d'un dataset et apprentissage

#Il faudra remplacer ces données par celles du Titanic
data = pd.read_csv('Data/passagers.csv', header=0, dtype={'Age': np.float64})
labels = data["Survived"]

print(data)
print(data["Survived"])

## les val data sont récupérer de la première execution du learning
val_data = pd.read_csv('Data/test.csv', header=0, dtype={'Age': np.float64})
val_labels = val_data["survived"]

print(data.size)
print(val_data.size)

model.fit(data.values, labels.values, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
#
# ## Evaluation du modèle de prédiction
# x = np.random.random((100, 32))
# y = np.random.random((100, 10))
# #print(y)
# print(model.metrics_names)
# print(model.evaluate(x, y, batch_size=32, verbose=False))
# model.predict(x, batch_size=32)
