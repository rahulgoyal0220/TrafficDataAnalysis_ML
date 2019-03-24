import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("final_data.csv")

df.head()

df  = df.drop('SEGMENT_ID', axis=1)
df  = df.drop('SPEED', axis=1)
df  = df.drop('STREET', axis=1)
df  = df.drop('FROM_STREET', axis=1)
df  = df.drop('TO_STREET', axis=1)

features = df.drop('Congestion_Status', axis =1).columns[:15]

target_names = ['Green', 'Red', 'Yellow']
target_names = array(target_names)

le = preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass


#Step 2
X = df[features]
y = df['Congestion_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=5)

#Try sclaing on data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf=RandomForestClassifier(n_estimators=100, n_jobs=-1)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



#plot feature importance

feature_imp = pd.Series(clf.feature_importances_,index=features).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
