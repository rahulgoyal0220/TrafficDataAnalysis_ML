# K-NN implementation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



df = pd.read_csv("final_data.csv")

df.head()
scaler = preprocessing.StandardScaler()
y = df['Congestion_Status'].values


# data pre processing ENCODING
le = preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass

names = df.columns
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)
X = scaled_df.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Setup arrays to store training and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the model
    knn.fit(X_train, y_train)
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model
knn.fit(X_train, y_train)

# Get accuracy. Note: In case of classification algorithms score method represents accuracy.
knn.score(X_test, y_test)

# let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)

# Confusion Matrix
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

# Classification Report
print(classification_report(y_test,y_pred))
