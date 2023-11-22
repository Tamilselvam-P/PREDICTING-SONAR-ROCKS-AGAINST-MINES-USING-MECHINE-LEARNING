# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Loading Dataset
df=pd.read_csv('/home/mind-graph/Documents/Collage_project/sonar.all-data.csv',header=None)
# Creating a list of headers names and adding those names as columns name
def adding_header(df):
    header = []

    for i in range(df.shape[1]-1):
        header.append(f"feature{i+1}")
        
    header.append('Output')
    df.columns=header
    return df
df=adding_header(df)
# Displaying first 5 rows of dataset
df.head(5)
# Shape of Dataset
df.shape
# Dataset Information
df.info()
# Descriptive Statistics Of Dataset
df.describe()
# Output Column Distribution
sns.countplot(data=df,x='Output')
plt.title("ROCK & MINE Distribution")
plt.show()
# Creating Boxplot
plt.figure(figsize=(30,15))
df[:-1].boxplot()
plt.show()
# Correlation
df.drop('Output',axis=1).corr()
# Spliting the dataset into Features and Output
X=df.drop(labels=['Output'],axis=1)
y=df['Output']
# Split dataset into train and test subsets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(f"Shape of X_train - {X_train.shape}")
print(f"Shape of y_train - {y_train.shape}")
print(f"Shape of X_test- {X_test.shape}")
print(f"Shape of y_test - {y_test.shape}")
models={
    "Logistics Regression":LogisticRegression(),
    "Support Vector Classifier":SVC(),
    "Decision Tree Classifier":DecisionTreeClassifier(),
    "Random Forest Classifier":RandomForestClassifier()
}

for name,model in models.items():
    clf=model
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    acc=accuracy_score(y_test,y_pred)
    print(f"Accuracy of {name} - {acc}")
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
SVC_grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)  
SVC_grid.fit(X_train, y_train)
print("Best Parameters -",SVC_grid.best_estimator_)
y_pred=SVC_grid.predict(X_test)
print(f"Accuracy Score of SVC after Hyperparameter Tuning- {accuracy_score(y_test,y_pred)}")
# Radom Forest Classifier
param_grid = {
    'n_estimators': [25, 50, 100],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}

RFC_grid= GridSearchCV(RandomForestClassifier(),param_grid=param_grid)
RFC_grid.fit(X_train, y_train)
print("Best Parameters -",RFC_grid.best_estimator_)
y_pred=RFC_grid.predict(X_test)
print(f"Accuracy Score of Random Forest Classifier after Hyperparameter Tuning- {accuracy_score(y_test,y_pred)}")

