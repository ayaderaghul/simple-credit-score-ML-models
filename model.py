# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

df_train = pd.read_csv('/kaggle/input/klpcreditscoring2internaldataset/train.csv')
df_test = pd.read_csv('/kaggle/input/klpcreditscoring2internaldataset/test.csv')                

c_train = df_train.select_dtypes("object").columns
df_train[c_train] = df_train[c_train].astype('category').apply(lambda x: x.cat.codes)

c_test = df_test.select_dtypes("object").columns
df_test[c_test] = df_test[c_test].astype('category').apply(lambda x: x.cat.codes)

y_train = df_train.iloc[:,1]
X_train = df_train.iloc[:,2:]

y_test = df_test.iloc[:,1]
X_test = df_test.iloc[:,2:]

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)

my_imputer_test = SimpleImputer()
X_test = my_imputer_test.fit_transform(X_test)

LR =LogisticRegression()
LR.fit(X_train,y_train)
LR.predict(X_test)
#LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
#LR.predict(X_test)
#round(LR.score(X_test,y_test), 4)

# SVM = svm.SVC(decision_function_shape="ovo").fit(X_train, y_train)
# SVM.predict(X_test)
# round(SVM.score(X_test, y_test), 4)

# RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)
# RF.predict(X_test)
# round(RF.score(X_test, y_test), 4)

# NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_train, y_train)
# NN.predict(X_test)
# round(NN.score(X_test, y_test), 4)
