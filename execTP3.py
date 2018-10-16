import numpy as np
import pandas as pd
import re as re

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import History,LearningRateScheduler
from  tensorflow.keras.layers import Dropout
print('TensorFlow %s, Keras %s, numpy %s, pandas %s'%(tf.__version__,keras.__version__, np.__version__,pd.__version__))


## Structure du réseau et nombre d'epochs (nombre de fois où on passe sur le DataSet)
num_hidden_layers = 4
first_layer_size = 128
other_layer_size = 512
epochs = 50

### Valeurs A tester dans la cross validation
lst_init_learning_rate = [0.01, 0.003, 0.1]
lst_dropout_prob = [0.15, 0.05]
n_splits = 10


def get_columns_metadata(df, lst_cols):
    header_df = pd.DataFrame(data=lst_cols, columns=['var_name'])
    print(header_df.size)
    print(df[lst_cols].mean().values.size)
    header_df['mean'] = df[lst_cols].mean().values
    header_df['min'] = df[lst_cols].min().values
    header_df['max'] = df[lst_cols].max().values
    header_df.set_index('var_name', inplace=True)
    return header_df


def normalize_value(value, header_df, col):
    return 2 * ((value - header_df.mean[col])/(value.max() - value.min()))


def normalize_number(df, header_df):
    for col in df.columns:
        if col in header_df.index:
            if df[col].isnull and isinstance(df[col], float) or isinstance(df[col], int):
                df[col] = normalize_value(df[col], header_df, col)
            else:
                df[col] = df[col]


def normalize_age(full_data):
    for dataset in full_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std() # Calcul de l'écart type
        age_null_count = dataset['Age'].isnull().sum() # nombre de valeur nulle
        # On génère une valeur aléatoire pour chaque valeur nulle, puis on l'arrondit à l'entier
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)


def normalize_sex_and_embarked(full_data):
    for dataset in full_data:
        dataset['Sex'].replace('female', 0, inplace=True)
        dataset['Sex'].replace('male', 1, inplace=True)
        dataset['Embarked'].replace('S', 0, inplace=True)
        dataset['Embarked'].replace('C', 1, inplace=True)
        dataset['Embarked'].replace('Q', 2, inplace=True)


def delete_useless_column(values, drop_array):
    return values.drop(drop_array, axis=1)


# La fonction pandas pd.read_csv permet de créer un objet Dataframe à partir d'un csv

# Données avec labels
train = pd.read_csv('Data/passagers.csv', header=0, dtype={'Age': np.float64})

# Données de tests sans label. Les prédictions de survie seront envoyées à kaggle
test = pd.read_csv('Data/test.csv', header=0, dtype={'Age': np.float64})

# On réunit les données dans une liste (pour pouvoir boucler sur les 2 dataframes)
full_data = [train, test]

# On garde les passagers ID des données test
# on en aura besoin pour le fichiers résultats de kaggle (voir l'exemple gender_submission.csv)
finalfile_index = test.PassengerId


def show_class(train):
    print("")
    print("Impact de la classe sur la survie")
    print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


def show_gender(train):
    print("")
    print("Impact du genre sur la survie")
    print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())


def show_familly_size(full_data, train):
    print("")
    print("Impact de la taille de la famille sur la survie")
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


def show_alone_person(full_data, train):
    print("")
    print("Distinction d'une personne seule")
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


def show_embarquement_port(full_data, train):
    print("")
    print("Impact du port d'embarquement sur la survie")
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


def show_price(full_data, train):
    print("")
    print("Impact du prix du billet sur la survie")
    for dataset in full_data:
        dataset.loc[dataset.Fare.isnull(), 'Fare'] = train['Fare'].mean()
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
    print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


def show_age(full_data, train):
    print("")
    print("Impact de l'age sur la survie")
    normalize_age(full_data)
    train['CategoricalAge'] = pd.qcut(train['Age'], 5)
    print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

def set_model(init_learning_rate, dropout_prob):
    # Architecture du réseau
    model = keras.Sequential()
    model.add(keras.layers.Dense(first_layer_size, activation='relu'))

    ### Ajouter ici une ligne  pour gérer le sur-apprentissage

    # Couches cachées (Hidden Layers)
    for i in range(num_hidden_layers):
        # Adds a densely-connected layer  to the model:
        model.add(keras.layers.Dense(other_layer_size, activation='relu'))

    ### Ajouter ici une ligne  pour gérer le sur-apprentissage
    # Couche de Sortie (avec fonction Softmax):
    model.add(keras.layers.Dense(2, activation='softmax'))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 1000, 0.96, staircase=True)


    ### Ici vous pouvez essayer différents algos de descentes de gradients
    #Définiton de l'optimizer  en charge de la Gradient Descent, de la fonction de coût et de la métrique.
    model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate),#RMSPropOptimizer(learning_rate), #GradientDescentOptimizer(learning_rate),AdamOptimizer
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


show_class(train)
show_gender(train)
show_familly_size(full_data, train)
show_alone_person(full_data, train)
show_embarquement_port(full_data, train)
show_price(full_data, train)
show_age(full_data, train)


normalize_sex_and_embarked(full_data)

drop_elements_test = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
drop_elements_train = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize', 'CategoricalAge', 'CategoricalFare']

train = delete_useless_column(train, drop_elements_train)
test = delete_useless_column(test, drop_elements_test)

header_df = get_columns_metadata(train, list(train.columns.values))

normalize_number(train, header_df)
normalize_number(test, header_df)

## Apprentissage
###Essayez Différents jeu de paramètre pour réduire le sur-appentissage
init_learning_rate = 0.003
dropout_prob = 0.5
check_epochs = 20
pourcentage_validation = 0.2

#A partir des données Train, on sépare features (X)  et labels "Survived"
lst_col = list(train.columns.values)
lst_col.remove('Survived')
X = train[lst_col]
y = train['Survived']

# On calcule la position de la séparation pour une répartition 80/20
position_validation_data = int(train.shape[0] * (1-pourcentage_validation))
print("")
print('position_validation_data=', position_validation_data)

# Construction des Features pour l'apprentissage et la validation.
# Transformation du Dataframe Pandas en Numpy Array (attendu par Keras)
X_train = X[lst_col][:position_validation_data].values
X_val = X[lst_col][position_validation_data:].values

# Construction des Labels pour l'apprentissage et la validation.  Hot Encoding
y_train = np.transpose([1-y[:position_validation_data], y[:position_validation_data]])
y_val = np.transpose([1-y[position_validation_data:], y[position_validation_data:]])

#Construction du modèle en appelant la fonction set_model
model = set_model(init_learning_rate, dropout_prob)
#définition d'une fonction History pour récupérer la fonction de coût et la métrique à chaque epoch.
hist = History()
model.fit(X_train, y_train, epochs=check_epochs, batch_size=512, validation_data=(X_val, y_val), verbose=False, callbacks=[hist])

print(hist.history.keys())

plt.rcParams["figure.figsize"] = (40, 20)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(hist.history['val_loss'], color= 'g')
ax2.plot(hist.history['loss'], color= 'b')
ax1.set_xlabel('epochs')
ax1.set_ylabel('Validation data Error', color='g')
ax2.set_ylabel('Training Data Error', color='b')
#plt.show()


## Cross validation
# sss => sklearn StratifiedShuffleSplit
def run_cross_validation(model, name, sss, acc_dict, loss_dict):
    loop = 1
    lst_col = list(train.columns.values)
    lst_col.remove('Survived')
    X = train[lst_col]
    y = train['Survived']
    position_validation_data = int(train.shape[0] * (1-pourcentage_validation))
    for train_index, test_index in sss.split(X, y):
        X_train = X[lst_col][:position_validation_data].values
        X_val = X[lst_col][position_validation_data:].values

        y_train = np.transpose([1-y[:position_validation_data], y[:position_validation_data]])
        y_val = np.transpose([1-y[position_validation_data:], y[position_validation_data:]])

        # Apprentissage et évaluation
        hist = History()
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val),verbose=False, callbacks=[hist])
        [loss, acc] = model.evaluate(X_val, y_val, batch_size=32, verbose=False)

        # Ajout de la performance dans les dictionnaires "loss_dict" et "acc_dict"
        if name in acc_dict:
            acc_dict[name] += acc
            loss_dict[name] += loss
        else:
            acc_dict[name] = acc
            loss_dict[name] = loss
        # Affichage de l'avancement
        print(loop, ':', [loss, acc])
        loop = loop + 1


## HyperParametrage
#Données utilisées pour la méthode split de l'objet StratifiedShuffleSplit
X = train.values[0::, 1::]
y = train.values[0::, 0]

# Création d'un dictionnaire pour stocker les modèles
model_dict = {}

sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=0)

# Création d'un dataframe pour logger les résultats
log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

# Boucle sur des valeurs de init_learning_rate et de dropout_prob
for init_learning_rate in lst_init_learning_rate:
    for dropout_prob in  lst_dropout_prob :
        #Initialisation des dictionnaires utilisés dans la cross validation
        acc_dict = {}
        loss_dict = {}
        #Construction du nom du modèle, en fonction des paramètres
        name="lr_%s_do_%s"%(init_learning_rate,dropout_prob)
        #Création de l'objet modèle
        model = set_model(init_learning_rate,dropout_prob)
        #Ajout du modèle au dico pour sélectionner le meilleur dans le suivant
        model_dict[name]=model
        run_cross_validation(model, name, sss, acc_dict, loss_dict)
        # Calcul de la performance du modèle comme moyenne pour chaque itération dans cross-validation
        for clf in acc_dict:
            acc_dict[clf] = acc_dict[clf] / n_splits
            log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
            log = log.append(log_entry)
print(log.values)

## Prediction
###A vous de completer les 3 lignes ci-dessous, sans oublier la normalisation !
### Analyser les résultats du bloc précédent pour choisir le meilleur paramètre
best_model = model_dict['lr_0.01_do_0.05']
# X = ???
# y = ???

y_hot = np.transpose([1-y, y])

#Apprentissage sur toutes les données, avec le modèle sélectionné
best_model.fit(X, y_hot, epochs=epochs, batch_size=32, verbose=False)
print(pd.DataFrame(best_model.evaluate(X, y_hot, batch_size=32, verbose=False), index=model.metrics_names))

#Inférence des données du fichier test et Construction du fichier à envoyer à Kaggle
prediction = best_model.predict(test.values, batch_size=32)
results = pd.DataFrame(np.argmax(prediction, axis=1), index=finalfile_index, columns=['Survived'])
results.to_csv('resultats.csv')
print(results.sum())
results.describe()