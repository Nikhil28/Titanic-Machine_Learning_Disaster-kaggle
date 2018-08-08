# -*- coding: utf-8 -*-
# Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# import datasets
dataset = pd.read_csv('train.csv')

###################### Feature engineering ######################

# Feature engineering for Title
NameSplit = dataset.Name.str.split('[,.]')
titles = [str.strip(name[1]) for name in NameSplit.values]
# New feature
dataset['Title'] = titles
dataset.Title.unique() # check the uniqueness
# redundancy: combine Mademoiselle and Madame into a single type
dataset.Title.values[dataset.Title.isin(['Miss', 'Mlle'])] = '0' # 'Miss'
dataset.Title.values[dataset.Title.isin(['Mme', 'Mrs'])] = '1' # 'Mrs'
dataset.Title.values[dataset.Title.isin(['Dona', 'Lady', 'the Countess', 'Jonkheer', 'Ms', 'Capt', 'Don', 'Major', 'Sir', 'Master', 'Mr','Dr', 'Rev', 'Col'])] = '2' # 'Other'


# Feature engineering for family size
dataset['FamilySize'] = dataset.SibSp.values + dataset.Parch.values + 1
# Feature engineering for family size Group
dataset['FamilySizeGroup'] = '2' #"Small"
dataset.FamilySizeGroup.values[dataset.FamilySize.values == 1] = '0' #'Alone'
dataset.FamilySizeGroup.values[dataset.FamilySize.values >= 5] = '1' # 'Big'


# Embarked used C as a most common
dataset.Embarked[61] = 'C'
dataset.Embarked[829] = 'C'


# Assign value to nan for cabins and we have cabin starts from A to G and T assigning value to them
dataset.Cabin = dataset.Cabin.fillna('9')
ShortlistingCabins = dataset.Cabin
import re
ShortlistingCabins.values[ShortlistingCabins.str.match(pat='^[A]+.*', case=True, flags=0, na='A', as_indexer=None)] = '0'
ShortlistingCabins.values[ShortlistingCabins.str.match(pat='^[B]+.*', case=True, flags=0, na='B', as_indexer=None)] = '1'
ShortlistingCabins.values[ShortlistingCabins.str.match(pat='^[C]+.*', case=True, flags=0, na='C', as_indexer=None)] = '2'
ShortlistingCabins.values[ShortlistingCabins.str.match(pat='^[D]+.*', case=True, flags=0, na='D', as_indexer=None)] = '3'
ShortlistingCabins.values[ShortlistingCabins.str.match(pat='^[E]+.*', case=True, flags=0, na='E', as_indexer=None)] = '4'
ShortlistingCabins.values[ShortlistingCabins.str.match(pat='^[F]+.*', case=True, flags=0, na='F', as_indexer=None)] = '5'
ShortlistingCabins.values[ShortlistingCabins.str.match(pat='^[G]+.*', case=True, flags=0, na='G', as_indexer=None)] = '6'
ShortlistingCabins.values[ShortlistingCabins.str.match(pat='^[T]+.*', case=True, flags=0, na='T', as_indexer=None)] = '7'



# Drop unrequired Features
dataset = dataset.drop(['PassengerId'], axis=1)
dataset = dataset.drop(['Name'], axis=1)
dataset = dataset.drop(['Ticket'], axis=1)
dataset = dataset.drop(['FamilySize'], axis=1)


# Convert Objects to Int64
dataset.Cabin = dataset.Cabin.infer_objects()
dataset.Title = dataset.Title.infer_objects()




# Dividibg data sets
X_dataset = pd.DataFrame(dataset.iloc[:, 1:11].values)
X = dataset.iloc[:, 1:11].values
y = dataset.iloc[:, 0].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
# Taking care of missing numaric data
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer = imputer.fit(X[:, [2]])
X[:, [2]] = imputer.transform(X[:, [2]])


# Encode categorical/string/character data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X =  LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1]) 
X[:, 7] = labelencoder_X.fit_transform(X[:, 7]) 
onehotencoder = OneHotEncoder(categorical_features=[7])
X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fetaure Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)







