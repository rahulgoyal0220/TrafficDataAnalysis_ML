import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


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


def random_forest(data):
    df = data
    features = df.drop('Congestion_Status', axis=1).columns[:15]
    target_names = ['Green', 'Red', 'Yellow']
    target_names = array(target_names)

    # Step 2
    X = df[features]
    y = df['Congestion_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=5)

    # Try sclaing on data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_result("Random Forest", y_test, y_pred)
    display_confusion_matrix('Knn Classifier', y_test, y_pred)
    display_classification_report('Knn Classifier', y_test, y_pred)


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


def display_important_feature(features, clf):
    # plot feature importance
    feature_imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()


def execute_random_forest(filename):
    data = readData(filename)
    random_forest(data)


execute_random_forest("dataset/chicago_data_process.csv")
