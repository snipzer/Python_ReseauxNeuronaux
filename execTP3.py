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
num_hidden_layers=4
first_layer_size = 128
other_layer_size = 512
epochs=50

### Valeurs A tester dans la cross validation
lst_init_learning_rate = [0.01,0.003, 0.1]
lst_dropout_prob=[0.15,0.05]
n_splits=10


def get_columns_metadata(df, lst_cols):
    header_df = pd.DataFrame(data=lst_cols, columns=['var_name'])
    header_df['mean']=df[lst_cols].mean().values
    header_df['min']= df[lst_cols].min().values
    header_df['max']= df[lst_cols].max().values
    header_df.set_index('var_name', inplace=True)
    return header_df


def normalize_number(df, header_df):
    for col in df.columns:
        print(col, col in header_df.index)
        if col in header_df.index:
            if df[col].isnull and isinstance(df[col], float) or isinstance(df[col], int):
                df[col] = header_df.mean[col]
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


def delete_useless_column(train, test):
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
    train = train.drop(drop_elements, axis=1)
    train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
    test = test.drop(drop_elements, axis=1)


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
    print("Impact de la classe sur la survie")
    print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


def show_gender(train):
    print("Impact du genre sur la survie")
    print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())


def show_familly_size(full_data, train):
    print("Impact de la taille de la famille sur la survie")
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


def show_alone_person():
    print("Distinction d'une personne seule")
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


def show_embarquement_port():
    print("Impact du port d'embarquement sur la survie")
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


def show_price():
    print("Impact du prix du billet sur la survie")
    for dataset in full_data:
        dataset.loc[dataset.Fare.isnull(), 'Fare'] = train['Fare'].mean()
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
    print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


def show_age():
    print("Impact de l'age sur la survie")
    normalize_age(full_data)
    train['CategoricalAge'] = pd.qcut(train['Age'],5)
    print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


normalize_sex_and_embarked(full_data)
delete_useless_column(train, test)

header_df = get_columns_metadata(train, list(train.columns.values))

normalize_number(train, header_df)
normalize_number(test, header_df)

print('\nTrain data:')
print(train.head(10))
print('\nTest data:')
print(test.head(10))