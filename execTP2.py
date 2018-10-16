import numpy as np
import pandas as pd
import re as re
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import History,LearningRateScheduler
from  tensorflow.keras.layers import Dropout

print('TensorFlow %s, Keras %s, numpy %s, pandas %s' % (tf.__version__, keras.__version__, np.__version__, pd.__version__))
__DEBUG__=False


## Structure du réseau et nombre d'epochs (nombre de fois où on passe sur le DataSet)
num_hidden_layers = 4
first_layer_size = 128
other_layer_size = 512
epochs = 50

###Valeurs A tester dans la cross validation
lst_init_learning_rate = [0.01, 0.003, 0.1]
lst_dropout_prob = [0.15, 0.05]
n_splits = 10


def get_columns_metadata(df, lst_cols):
    header_df = pd.DataFrame(data=lst_cols, columns=['var_name'])
    header_df['mean']=df[lst_cols].mean().values
    header_df['min']= df[lst_cols].min().values
    header_df['max']= df[lst_cols].max().values
    header_df.set_index('var_name', inplace=True)
    return header_df


def normalize(df, header_df):
    for col in df.columns:
        print(col, col in header_df.index)
        if col in header_df.index:
            if df[col].isnull and isinstance(df[col], float) or isinstance(df[col], int):
                df[col] = header_df.mean[col]
            else:
                df[col] = df[col]


train = pd.read_csv('Data/passagers.csv', header=0, dtype={'Age': np.float64})
test = pd.read_csv('Data/test.csv', header=0, dtype={'Age': np.float64})
full_data = [train, test]
finalfile_index = test.PassengerId    #Index des données de test pour le résultat final
train.info()

## Impact de la classe sur la survie
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

## Impact du sex sur la survie
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())

## Impact de la famille sur la survie
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

## Distinction sur les personnes seule
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

## Impact du Port d'embarquement sur la survie
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

## Impact du prix du ticket
for dataset in full_data:
    dataset.loc[dataset.Fare.isnull(), 'Fare'] = train['Fare'].mean()
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

## Impact de l'age
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    #dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)
print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

## Mise en forme des données
for dataset in full_data:
    # Traitement variable 'Sex'
    dataset['Sex'].replace('female', 0, inplace=True)
    dataset['Sex'].replace('male', 1, inplace=True)

    # Traitement variable 'Embarked'
    dataset['Embarked'].replace('S', 0, inplace=True)
    dataset['Embarked'].replace('C', 1, inplace=True)
    dataset['Embarked'].replace('Q', 2, inplace=True)

# Suppression des colonnes inutiles
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

### N'oubliez pas de mettre à jour la fonction normalize !
header_df=get_columns_metadata(train,list(train.columns.values))
normalize(train,header_df)



test  = test.drop(drop_elements, axis=1)
normalize(test,header_df)

print(train.head(10))
print(test.head(10))

## Création du modèle et initialisation Training
def set_model(init_learning_rate,dropout_prob):
    model = keras.Sequential()
    model.add(keras.layers.Dense(first_layer_size, activation='relu'))
    ### Ajouter ici une ligne  pour gérer le sur-apprentissage
    #Couche cachées
    for i in range(num_hidden_layers):
        # Adds a densely-connected layer  to the model:
        model.add(keras.layers.Dense(other_layer_size, activation='relu'))
    ### Ajouter ici une ligne  pour gérer le sur-apprentissage
    # Couche de Sortie:
    model.add(keras.layers.Dense(2, activation='softmax'))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,1000, 0.96, staircase=True)
    ### Ici vous pouvez essayer différents algos de descentes de gradients
    model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate),#RMSPropOptimizer(learning_rate), #GradientDescentOptimizer(learning_rate),AdamOptimizer
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

## Vérifiction du sur-apprentissage

###Essayez Différents jeu de paramètre pour réduire le sur-appentissage
init_learning_rate=0.1
dropout_prob= 0
epochs=200
pourcentage_validation= 0.2
lst_col=list(train.columns.values)
lst_col.remove('Survived')

X=train[lst_col]
y=train['Survived']

position_validation_data=int(train.shape[0] * (1-pourcentage_validation))
print('position_validation_data=',position_validation_data)
X_train, X_test = X[lst_col][:position_validation_data].values, X[lst_col][position_validation_data:].values
y_train, y_test = np.transpose([1-y[:position_validation_data], y[:position_validation_data]]), \
                  np.transpose([1-y[position_validation_data:], y[position_validation_data:]])


model = set_model(init_learning_rate,dropout_prob)
hist = History()
model.fit(X_train, y_train, epochs=epochs, batch_size=128,validation_data=(X_test, y_test),verbose=False, callbacks=[hist])

plt.rcParams["figure.figsize"] = (40,20)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(hist.history['val_loss'], color= 'g')
ax2.plot(hist.history['loss'], color= 'b')
ax1.set_xlabel('epochs')
ax1.set_ylabel('Validation data Error', color='g')
ax2.set_ylabel('Training Data Error', color='b')
plt.show()