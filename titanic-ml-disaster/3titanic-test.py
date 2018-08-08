# -*- coding: utf-8 -*-

#-----------------------------   Prediction   -----------------------------#
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.6)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





















# Predicting a single new observation
new_prediction = pd.read_csv('test.csv')

# Feature engineer for Title
new_NameSplit = new_prediction.Name.str.split('[,.]')
titles = [str.strip(name[1]) for name in new_NameSplit.values]
new_prediction['Title'] = titles
new_prediction.Title.unique() # check the uniqueness
new_prediction.Title.values[new_prediction.Title.isin(['Miss'])] = '0' # 'Miss'
new_prediction.Title.values[new_prediction.Title.isin(['Mrs'])] = '1' # 'Mrs'
new_prediction.Title.values[new_prediction.Title.isin(['Dona', 'Ms', 'Master', 'Mr','Dr', 'Rev', 'Col'])] = '2' # 'Other'

# Feature engineer for family size
new_prediction['FamilySize'] = new_prediction.SibSp.values + new_prediction.Parch.values + 1
# Feature engineer for family size Group
new_prediction['FamilySizeGroup'] = '2' #"Small"
new_prediction.FamilySizeGroup.values[new_prediction.FamilySize.values == 1] = '0' #'Alone'
new_prediction.FamilySizeGroup.values[new_prediction.FamilySize.values >= 5] = '1' # 'Big'

# Assign value to nan for cabins
new_prediction.Cabin = new_prediction.Cabin.fillna('9')
new_ShortlistingCabins = new_prediction.Cabin
import re
#CabinList.values[re.match("^[A]*", dataset.Cabin.values)] = 0 #'A'
new_ShortlistingCabins.values[new_ShortlistingCabins.str.match(pat='^[A]+.*', case=True, flags=0, na='A', as_indexer=None)] = '0'
new_ShortlistingCabins.values[new_ShortlistingCabins.str.match(pat='^[B]+.*', case=True, flags=0, na='B', as_indexer=None)] = '1'
new_ShortlistingCabins.values[new_ShortlistingCabins.str.match(pat='^[C]+.*', case=True, flags=0, na='C', as_indexer=None)] = '2'
new_ShortlistingCabins.values[new_ShortlistingCabins.str.match(pat='^[D]+.*', case=True, flags=0, na='D', as_indexer=None)] = '3'
new_ShortlistingCabins.values[new_ShortlistingCabins.str.match(pat='^[E]+.*', case=True, flags=0, na='E', as_indexer=None)] = '4'
new_ShortlistingCabins.values[new_ShortlistingCabins.str.match(pat='^[F]+.*', case=True, flags=0, na='F', as_indexer=None)] = '5'
new_ShortlistingCabins.values[new_ShortlistingCabins.str.match(pat='^[G]+.*', case=True, flags=0, na='G', as_indexer=None)] = '6'

# Drop unrequired Features
new_prediction = new_prediction.drop(['PassengerId'], axis=1)
new_prediction = new_prediction.drop(['Survived'], axis=1)
new_prediction = new_prediction.drop(['Name'], axis=1)
new_prediction = new_prediction.drop(['Ticket'], axis=1)
new_prediction = new_prediction.drop(['predictions'], axis=1)
new_prediction = new_prediction.drop(['FamilySize'], axis=1)


new_prediction_X = new_prediction.iloc[:, 0:10].values
# new dataset
# Taking care of missing data
from sklearn.preprocessing import Imputer
# Taking care of missing numaric data
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer = imputer.fit(new_prediction_X[:, [2,5]])
new_prediction_X[:, [2,5]] = imputer.transform(new_prediction_X[:, [2,5]])

# Encode categorical/string/character data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
new_labelencoder_X =  LabelEncoder()
new_prediction_X[:, 1] = new_labelencoder_X.fit_transform(new_prediction_X[:, 1]) 
new_prediction_X[:, 7] = new_labelencoder_X.fit_transform(new_prediction_X[:, 7]) 
onehotencoder = OneHotEncoder(categorical_features=[7])
new_prediction_X = onehotencoder.fit_transform(new_prediction_X).toarray()

# Fetaure Scalling
from sklearn.preprocessing import StandardScaler
sc_new_X = StandardScaler()
new_prediction_X = sc_new_X.fit_transform(new_prediction_X)

y_new_pred = classifier.predict(new_prediction_X)

prediction = pd.DataFrame(y_new_pred, columns=['Survived']).to_csv('prediction.csv')