import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

dataset = pd.read_csv('train.csv', na_values=['NA'])

#TAKE A LOOK AT THE DATA
dataset.head()
dataset.describe()

#CHOOSE RELEVANT FEATURES
'''
1. Pclass
2. Sex (Change to int)
3. Age
4. Parch + Sibsp (To be engineered)
'''

#LOOK FOR NULL VALUES IN THE RELEVANT FEATURES
dataset.isnull().any() #Returns a boolean dataframe showing columns with Null values as 'True'

#REPLACE NULL VALUES WITH THE MEAN OF THAT ATTR e.g. 'Age'
ageMean = dataset['Age'].mean()
dataset['Age'].fillna(ageMean, inplace=True)

#ENGINEER A FEATURE
newFeature = pd.Series(dataset['Parch'] + dataset['SibSp'])
dataset = dataset.assign(Relations = newFeature)

#CHANGE THE SEX INTO A BINARY FORMAT
dataset['Sex'].replace('male', 1, inplace=True)
dataset['Sex'].replace('female', 0, inplace=True)

#FROM THE CLEANED UP DATA, CREATE A NEW CSV FILE WITH ONLY THE RELEVANT ATTRS (id, pclass, sex, age, relations)
header = ["PassengerId", "Pclass", "Sex", "Age", "Relations", "Survived"]
dataset.to_csv('cleanedTitanic.csv', columns=header, index=False)

df = pd.read_csv('cleanedTitanic.csv')

#SPECIFY INPUTS AND OUTPUTS
inputs = df[['Pclass', 'Sex', 'Age', 'Relations']]
labels = df['Survived']

#INTIALISE A RANDOM FOREST CLASSIFIER
rfc = RandomForestClassifier()

#INITIALISE SOME PARAMS TO BE TESTED
rfcParameterGrid = {'n_estimators':[5,10,25,50],
					'criterion':['gini','entropy'],
					'max_features':[1,2,3,4],
					'warm_start':[True, False]
					}

#USE K-FOLD CROSS-VALIDATION
folds = StratifiedKFold(n_splits=10)

#FIND THE BEST PARAMETERS FOR THE MODEL
rfcgrid_search = GridSearchCV(rfc, param_grid = rfcParameterGrid, cv=folds)

#FIT THE MODEL TO THE DATA
rfcgrid_search.fit(inputs, labels)

#CHOOSE THE BEST CLASSIFIER
rfc = rfcgrid_search.best_estimator_

#TEST ACCURACY USING THE DATA
accuracy = cross_val_score(rfc, inputs, labels, cv=10)



#LOAD TEST DATA AND RUN MODEL
test = pd.read_csv('test.csv')

#CLEAN AND ENGINEER FEATURES
test['Age'].fillna(test['Age'].mean(), inplace=True)

newFeature = pd.Series(test['Parch'] + test['SibSp'])
test = test.assign(Relations = newFeature)

#CHANGE THE SEX INTO A BINARY FORMAT
test['Sex'].replace('male', 1, inplace=True)
test['Sex'].replace('female', 0, inplace=True)


#GENERATE PREDICTIONS OF SURVIVAL THEN CREATE A PREDICTIONS CSV
testInputs = test[['Pclass', 'Sex', 'Age', 'Relations']]
predictions = rfc.predict(testInputs)

predicted = pd.Series(predictions)
test = test.assign(Survived = predicted)

testHeader = ["PassengerId", "Survived"]
test.to_csv('predictions.csv', columns=testHeader, index=False)
