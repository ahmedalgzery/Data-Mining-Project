# Credit Card Fraud Detection

## Overview

This code is designed to perform credit card fraud detection using machine learning techniques. The dataset used for this task is loaded from the 'creditcard.csv' file. The code covers data exploration, preprocessing, visualization, and the application of machine learning models.

## Prerequisites

Make sure you have the required Python libraries installed. You can install them using the following:

```bash
pip install pandas scikit-learn six pydotplus matplotlib seaborn
```

## Steps

### Step 1: Load and Explore the Dataset

```python
# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display the first 5 rows
df.head()

# Display dataset information
df.info()

# Check for missing values
df.isnull().sum()

# Display distribution of normal and fraudulent transactions
df['Class'].value_counts()
```

### Step 2: Data Preprocessing

```python
# Separate data for analysis
normal = df[df.Class == 0]
fraud = df[df.Class == 1]

# Statistical measures of the data
normal.Amount.describe()
fraud.Amount.describe()

# Compare the values for both transactions
df.groupby('Class').mean()

# Create a balanced dataset
normal_sample = normal.sample(n=492)
new_dataset = pd.concat([normal_sample, fraud], axis=0)
```

### Step 3: Data Visualization

```python
# Scatter plot before splitting
plt.scatter(X['Time'], X['Amount'], c=Y, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.title('Scatter Plot Before Splitting')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount ($)')
plt.show()
```

### Step 4: Train-Test Split

```python
# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

### Step 5: K-Nearest Neighbors (KNN) Model

```python
# Create and train KNN model
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, Y_train)

# Make predictions
knn_pred = knn_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(Y_test, knn_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### Step 6: Decision Tree Model

```python
# Create and train Decision Tree model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)

# Make predictions
tree_pred = dtree.predict(X_test)

# Evaluate accuracy
accuracy_score(Y_test, tree_pred)
```

### Step 7: Bagging Classifier

```python
# Create a BaggingClassifier with a DecisionTree base estimator
bagging_model = BaggingClassifier()
bagging_model.fit(X_train, Y_train)

# Make predictions
bagging_pred = bagging_model.predict(X_test)

# Evaluate accuracy
accuracy_bagging = accuracy_score(Y_test, bagging_pred)
print(f"Bagging Accuracy: {accuracy_bagging:.2f}")
```

### Step 8: Confusion Matrices and Visualization

```python
# Confusion matrix and visualization for KNN
conf_matrix = confusion_matrix(Y_test, knn_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix (KNN)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion matrix and visualization for Decision Tree
conf_matrix_tree = confusion_matrix(Y_test, tree_pred)
sns.heatmap(conf_matrix_tree, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix (Decision Tree)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion matrix and visualization for Bagging
conf_matrix_bagging = confusion_matrix(Y_test, bagging_pred)
sns.heatmap(conf_matrix_bagging, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix (Bagging)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### Step 9: Decision Tree Visualization

```python
# Visualize the Decision Tree
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, feature_names=X.columns, class_names=['Amount', 'Class'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
Image(graph.create_png())
```

## Conclusion

This code provides a comprehensive workflow for credit card fraud detection, including data loading, preprocessing, model training, and evaluation. The K-Nearest Neighbors, Decision Tree, and Bagging Classifier models are used and their performances are assessed using confusion matrices. Additionally, a Decision Tree is visualized for better interpretability.
