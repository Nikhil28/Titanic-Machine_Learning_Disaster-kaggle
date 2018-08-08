#----------------------------- ALGOS -----------------------------#



#-----------------------------   K Neighbors Classifier   -----------------------------#
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
classifier.fit(X_train, y_train)


#-----------------------------   SVM Classifier   -----------------------------#
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


#-----------------------------   kernel SVM Classifier   -----------------------------#
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


#-----------------------------   Decision Tree Classifier   -----------------------------#
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


#-----------------------------   Random Forest Classifier   -----------------------------#
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 9, criterion = 'gini', random_state = 0, n_jobs = -1) #9
classifier.fit(X_train, y_train)
# mean = 0.8029; max = 1; std = 0.152; CM = 155/179 = 86.59


#-----------------------------   XG Boost Classifier   -----------------------------#
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=15, learning_rate=0.7, n_estimators=100, gamma=0, random_state=0, n_jobs = -1)
classifier.fit(X_train, y_train)


#-----------------------------   k-Fold Cross Validation   -----------------------------#
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 100, n_jobs = -1)
accuracies.mean()
accuracies.max()
accuracies.std()


#-----------------------------   Grid Search CV   -----------------------------#
# Applying Grid Search to find the best model and the best parameters for SVM
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100], 'kernel': ['linear']}]
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 100, 
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#-----------------------------   kernel SVM Classifier after GridSearchCV   -----------------------------#
from sklearn.svm import SVC
classifier = SVC(C = 100, kernel = 'rbf', gamma = 0.001, random_state = 0)
classifier.fit(X_train, y_train)


#-----------------------------   ANN using Keras   -----------------------------#
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22))
# Adding the second hidden layer
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100, epochs = 2000)






