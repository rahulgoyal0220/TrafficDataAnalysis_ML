from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics


def print_result(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(metrics.accuracy_score(true_value, pred)))
    print("precision: {}".format(metrics.precision_score(true_value, pred)))
    print("recall: {}".format(metrics.recall_score(true_value, pred)))
    print("f1: {}".format(metrics.f1_score(true_value, pred)))


def confusion_matrix(headline, y_test, y_pred):
    # Confusion Matrix
    print("Confustion Matrix report: ", headline)
    confusion_matrix(y_test, y_pred)
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


def classification_report(headline, y_test, y_pred):
    # Classification Report
    print("Classfication report: ", headline)
    print(classification_report(y_test, y_pred))
