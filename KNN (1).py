import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('/home/mind-graph/Documents/Collage_project/sonar.all-data.csv', header=None)

# Rename columns
newNames = []
for i in range(len(df.columns) - 1):
    colName = 'Freq_' + str(i + 1)
    newNames.append(colName)
newNames.append('OutputTag')
df.columns = newNames

# Create a correlation heatmap (excluding 'OutputTag' column)
plt.figure(figsize=(12, 8), dpi=200)
sns.heatmap(df.drop(columns=['OutputTag']).corr(), cmap='coolwarm')

# Map 'R' and 'M' to 0 and 1 in a new column 'Result'
df['Result'] = df['OutputTag'].map({'R': 0, 'M': 1})

# Split the dataset into features (X) and target (y)
X = df.drop(['OutputTag', 'Result'], axis=1)
y = df['OutputTag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a StandardScaler and KNeighborsClassifier pipeline
scale = StandardScaler()
KNN = KNeighborsClassifier()
flow = [('scale', scale), ('KNN', KNN)]
pipe = Pipeline(flow)

# Define the range of neighbors for grid search
kRanges = list(range(1, 30))
param_grid = {'KNN__n_neighbors': kRanges}

# Create a GridSearchCV classifier
classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
classifier.fit(X_train, y_train)

# Get the best estimator's parameters
best_params = classifier.best_estimator_.get_params()

# Plot the mean test scores for different neighbors
pd.DataFrame(classifier.cv_results_)['mean_test_score'].plot()

# Predict using the best estimator
y_pred = classifier.predict(X_test)

# Evaluate the model
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print("Confusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_rep)
print("\nAccuracy Score:", accuracy)
