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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sonar_df = pd.read_csv('/home/mind-graph/Documents/Collage_project/sonar.all-data.csv', header=None)
sonar_df.head()
sonar_df.describe() 
sonar_df[60].value_counts()
X = sonar_df.drop(columns=60, axis=1)
y = sonar_df[60]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)
log_regression_model = LogisticRegression()
log_regression_model.fit(X_train, y_train)
#accuracy on test data
X_pred = log_regression_model.predict(X_test)
accuracy = accuracy_score(X_pred, y_test)
print('Accuracy score: ', accuracy)