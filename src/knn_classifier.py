# K-NN implementation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import metrics
import matplotlib.pyplot as plt


def readData(filename):
    df = pd.read_csv(filename)
    # data pre processing ENCODING
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])
        else:
            pass
    return df


def knnClassfier(data):
    y = data['Congestion_Status'].values
    X = data.drop('Congestion_Status', axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=6, n_jobs =-1)
    # Fit the model
    knn.fit(X_train, y_train)
    # Get accuracy. Note: In case of classification algorithms score method represents accuracy.
    knn.score(X_test, y_test)
    # let us get the predictions using the classifier we had fit above
    y_pred = knn.predict(X_test)
    print_result('Knn Classifier evaluation score', y_test, y_pred)
    display_confusion_matrix('Knn Classifier', y_test, y_pred)
    display_classification_report('Knn Classifier', y_test, y_pred)


def getBestK_vakueKNN(X_train, y_train, X_test, y_test):
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
    plotKNN_k_value_graph(neighbors, test_accuracy, train_accuracy)


def plotKNN_k_value_graph(neighbors, test_accuracy, train_accuracy):
    # Generate plot
    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()


def execute_knn(filename):
    data = readData(filename)
    knnClassfier(data)


def print_result(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(metrics.accuracy_score(true_value, pred)))
    # print("precision: {}".format(metrics.precision_score(true_value, pred)))
    # print("recall: {}".format(metrics.recall_score(true_value, pred)))
    # print("f1: {}".format(metrics.f1_score(true_value, pred)))


def display_confusion_matrix(headline, y_test, y_pred):
    # Confusion Matrix
    print("Confustion Matrix report: ", headline)
    confusion_matrix(y_test, y_pred)
    print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


def display_classification_report(headline, y_test, y_pred):
    # Classification Report
    print("Classfication report: ", headline)
    print(classification_report(y_test, y_pred))


execute_knn("dataset/chicago_data_process.csv")
