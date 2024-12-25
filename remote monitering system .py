#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn


# In[ ]:


df=pd.read_csv("/content/drive/MyDrive/NIT_dataset.csv")
df.sample(5)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
corr_matrix = df.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(35, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
X=df.drop("Label_n",axis=1)
y=df["Label_n"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Step 3: Train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=80, random_state=32)
rf.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = rf.predict(X_test)

# Step 5: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)


# In[ ]:


rf.score(X_test,y_test)


# In[ ]:


# Hide Warnings

import warnings
warnings.filterwarnings('ignore') #Set it to default to receive warnings again


# In[ ]:


import time
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss,matthews_corrcoef,auc, accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

import xgboost as xgb
from xgboost import XGBClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier


# In[ ]:


# Using Logistic Regression

scores_train = []
scores_test = []
time_taken = []
precision_train = []
precision_test = []
recall_train = []
recall_test = []
f1_train = []
f1_test = []
MCC_test=[]
Log_test=[]

for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.333)

    start_time = time.time()
    model = LogisticRegression()
    model.fit(x_train, y_train)
    taken = time.time() - start_time
    time_taken.append(taken)

    y_pred_train = model.predict(x_train)
    score = accuracy_score(y_pred_train, y_train)
    scores_train.append(score)

    y_pred = model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    scores_test.append(score)

    score = precision_score(y_test, y_pred)
    precision_test.append(score)

    score = precision_score(y_train, y_pred_train)
    precision_train.append(score)

    score = recall_score(y_test, y_pred)
    recall_test.append(score)

    score = recall_score(y_train, y_pred_train)
    recall_train.append(score)

    score = f1_score(y_test, y_pred)
    f1_test.append(score)

    score = f1_score(y_train, y_pred_train)
    f1_train.append(score)

    score = matthews_corrcoef(y_test, y_pred)
    MCC_test.append(score)

    score = log_loss(y_test, y_pred)
    Log_test.append(score)

print("Average time taken is", mean(time_taken), "secs")
print("Average training data accuracy:",mean(scores_train)*100,)
print("Average test data accuracy:",mean(scores_test)*100)
print("Average training data precision:",mean(precision_train)*100,)
print("Average test data precision:",mean(precision_test)*100)
print("Average training data recall:",mean(recall_train)*100,)
print("Average test data recall:",mean(recall_test)*100)
print("Average training data f1 score:",mean(f1_train)*100,)
print("Average test data f1 score:",mean(f1_test)*100)
print("Average test data MCC:",mean(MCC_test)*100)
print("Average test data Log_loss:",mean(Log_test)*100)


# In[ ]:


# Using SVM

scores_train = []
scores_test = []
time_taken = []
precision_train = []
precision_test = []
recall_train = []
recall_test = []
f1_train = []
f1_test = []

for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.333)

    start_time = time.time()
    model = make_pipeline(StandardScaler(), LinearSVC(tol=1e-5))
    model.fit(x_train, y_train)
    taken = time.time() - start_time
    time_taken.append(taken)

    y_pred_train = model.predict(x_train)
    score = accuracy_score(y_pred_train, y_train)
    scores_train.append(score)

    y_pred = model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    scores_test.append(score)

    score = precision_score(y_test, y_pred)
    precision_test.append(score)

    score = precision_score(y_train, y_pred_train)
    precision_train.append(score)

    score = recall_score(y_test, y_pred)
    recall_test.append(score)

    score = recall_score(y_train, y_pred_train)
    recall_train.append(score)

    score = f1_score(y_test, y_pred)
    f1_test.append(score)

    score = f1_score(y_train, y_pred_train)
    f1_train.append(score)

    score = matthews_corrcoef(y_test, y_pred)
    MCC_test.append(score)

    score = log_loss(y_test, y_pred)
    Log_test.append(score)


print("Average time taken is", mean(time_taken), "secs")
print("Average training data accuracy:",mean(scores_train)*100,)
print("Average test data accuracy:",mean(scores_test)*100)
print("Average training data precision:",mean(precision_train)*100,)
print("Average test data precision:",mean(precision_test)*100)
print("Average training data recall:",mean(recall_train)*100,)
print("Average test data recall:",mean(recall_test)*100)
print("Average training data f1 score:",mean(f1_train)*100,)
print("Average test data f1 score:",mean(f1_test)*100)
print("Average test data MCC:",mean(MCC_test)*100)
print("Average test data Log_loss:",mean(Log_test)*100)


# In[ ]:


# Using Naive Bayes

scores_train = []
scores_test = []
time_taken = []
precision_train = []
precision_test = []
recall_train = []
recall_test = []
f1_train = []
f1_test = []

for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.333)

    start_time = time.time()
    model = GaussianNB()
    model.fit(x_train, y_train)
    taken = time.time() - start_time
    time_taken.append(taken)

    y_pred_train = model.predict(x_train)
    score = accuracy_score(y_pred_train, y_train)
    scores_train.append(score)

    y_pred = model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    scores_test.append(score)

    score = precision_score(y_test, y_pred)
    precision_test.append(score)

    score = precision_score(y_train, y_pred_train)
    precision_train.append(score)

    score = recall_score(y_test, y_pred)
    recall_test.append(score)

    score = recall_score(y_train, y_pred_train)
    recall_train.append(score)

    score = f1_score(y_test, y_pred)
    f1_test.append(score)

    score = f1_score(y_train, y_pred_train)
    f1_train.append(score)

    score = matthews_corrcoef(y_test, y_pred)
    MCC_test.append(score)

    score = log_loss(y_test, y_pred)
    Log_test.append(score)

print("Average time taken is", mean(time_taken), "secs")
print("Average training data accuracy:",mean(scores_train)*100,)
print("Average test data accuracy:",mean(scores_test)*100)
print("Average training data precision:",mean(precision_train)*100,)
print("Average test data precision:",mean(precision_test)*100)
print("Average training data recall:",mean(recall_train)*100,)
print("Average test data recall:",mean(recall_test)*100)
print("Average training data f1 score:",mean(f1_train)*100,)
print("Average test data f1 score:",mean(f1_test)*100)
print("Average test data MCC:",mean(MCC_test)*100)
print("Average test data Log_loss:",mean(Log_test)*100)


# In[ ]:


# Using Random Forest

scores_train = []
scores_test = []
time_taken = []
precision_train = []
precision_test = []
recall_train = []
recall_test = []
f1_train = []
f1_test = []

for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.333)

    start_time = time.time()

    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)

    taken = time.time() - start_time
    time_taken.append(taken)

    y_pred_train = rf_model.predict(x_train)
    score = accuracy_score(y_pred_train, y_train)
    scores_train.append(score)


    y_pred = rf_model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    scores_test.append(score)

    score = precision_score(y_test, y_pred)
    precision_test.append(score)

    score = precision_score(y_train, y_pred_train)
    precision_train.append(score)

    score = recall_score(y_test, y_pred)
    recall_test.append(score)

    score = recall_score(y_train, y_pred_train)
    recall_train.append(score)

    score = f1_score(y_test, y_pred)
    f1_test.append(score)

    score = f1_score(y_train, y_pred_train)
    f1_train.append(score)

    score = matthews_corrcoef(y_test, y_pred)
    MCC_test.append(score)

    score = log_loss(y_test, y_pred)
    Log_test.append(score)

print("Average time taken is", mean(time_taken), "secs")
print("Average training data accuracy:",mean(scores_train)*100,)
print("Average test data accuracy:",mean(scores_test)*100)
print("Average training data precision:",mean(precision_train)*100,)
print("Average test data precision:",mean(precision_test)*100)
print("Average training data recall:",mean(recall_train)*100,)
print("Average test data recall:",mean(recall_test)*100)
print("Average training data f1 score:",mean(f1_train)*100,)
print("Average test data f1 score:",mean(f1_test)*100)
print("Average test data MCC:",mean(MCC_test)*100)
print("Average test data Log_loss:",mean(Log_test)*100)


# In[ ]:


# Using Weighted-Average Bagging

scores_train = []
scores_test = []
time_taken = []
precision_train = []
precision_test = []
recall_train = []
recall_test = []
f1_train = []
f1_test = []

for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.333)

    start_time = time.time()

    models = list()
    models.append(('lr', LogisticRegression()))
    models.append(('cart', DecisionTreeClassifier()))
    models.append(('bayes', GaussianNB()))

    scores = list()
    for name, model in models:
        model.fit(x_train, y_train)
        yhat = model.predict(x_test)
        acc = accuracy_score(y_test, yhat)
        scores.append(acc)

    ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
    ensemble.fit(x_train, y_train)

    taken = time.time() - start_time
    time_taken.append(taken)


    y_pred_train = ensemble.predict(x_train)
    score = accuracy_score(y_pred_train, y_train)
    scores_train.append(score)

    y_pred = ensemble.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    scores_test.append(score)

    score = precision_score(y_test, y_pred)
    precision_test.append(score)

    score = precision_score(y_train, y_pred_train)
    precision_train.append(score)

    score = recall_score(y_test, y_pred)
    recall_test.append(score)

    score = recall_score(y_train, y_pred_train)
    recall_train.append(score)

    score = f1_score(y_test, y_pred)
    f1_test.append(score)

    score = f1_score(y_train, y_pred_train)
    f1_train.append(score)

    score = matthews_corrcoef(y_test, y_pred)
    MCC_test.append(score)

    score = log_loss(y_test, y_pred)
    Log_test.append(score)
print("Average time taken is", mean(time_taken), "secs")
print("Average training data accuracy:",mean(scores_train)*100,)
print("Average test data accuracy:",mean(scores_test)*100)
print("Average training data precision:",mean(precision_train)*100,)
print("Average test data precision:",mean(precision_test)*100)
print("Average training data recall:",mean(recall_train)*100,)
print("Average test data recall:",mean(recall_test)*100)
print("Average training data f1 score:",mean(f1_train)*100,)
print("Average test data f1 score:",mean(f1_test)*100)
print("Average test data MCC:",mean(MCC_test)*100)
print("Average test data Log_loss:",mean(Log_test)*100)


# In[ ]:


# Using XGBoost

scores_train = []
scores_test = []
time_taken = []
precision_train = []
precision_test = []
recall_train = []
recall_test = []
f1_train = []
f1_test = []

for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.333)

    start_time = time.time()
    from xgboost import XGBClassifier
    params = {
                'objective':'binary:logistic',
                'max_depth': 4,
                'alpha': 10,
                'learning_rate': 1.0,
                'n_estimators':100
             }
    xgb_clf = XGBClassifier()
    xgb_clf.fit(x_train, y_train)
    taken = time.time() - start_time

    time_taken.append(taken)


    y_pred_train = xgb_clf.predict(x_train)
    score = accuracy_score(y_pred_train, y_train)
    scores_train.append(score)

    y_pred = xgb_clf.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    scores_test.append(score)

    score = precision_score(y_test, y_pred)
    precision_test.append(score)

    score = precision_score(y_train, y_pred_train)
    precision_train.append(score)

    score = recall_score(y_test, y_pred)
    recall_test.append(score)

    score = recall_score(y_train, y_pred_train)
    recall_train.append(score)

    score = f1_score(y_test, y_pred)
    f1_test.append(score)

    score = f1_score(y_train, y_pred_train)
    f1_train.append(score)

    score = matthews_corrcoef(y_test, y_pred)
    MCC_test.append(score)

    score = log_loss(y_test, y_pred)
    Log_test.append(score)
print("Average time taken is", mean(time_taken), "secs")
print("Average training data accuracy:",mean(scores_train)*100,)
print("Average test data accuracy:",mean(scores_test)*100)
print("Average training data precision:",mean(precision_train)*100,)
print("Average test data precision:",mean(precision_test)*100)
print("Average training data recall:",mean(recall_train)*100,)
print("Average test data recall:",mean(recall_test)*100)
print("Average training data f1 score:",mean(f1_train)*100,)
print("Average test data f1 score:",mean(f1_test)*100)
print("Average test data MCC:",mean(MCC_test)*100)
print("Average test data Log_loss:",mean(Log_test)*100)


# In[ ]:


# Using LightGBM

scores_train = []
scores_test = []
time_taken = []
precision_train = []
precision_test = []
recall_train = []
recall_test = []
f1_train = []
f1_test = []

for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.333)

    start_time = time.time()
    model = lgb.LGBMClassifier()
    model.fit(x_train, y_train)
    taken = time.time() - start_time

    time_taken.append(taken)

    y_pred_train = model.predict(x_train)
    score = accuracy_score(y_pred_train, y_train)
    scores_train.append(score)

    y_pred = model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    scores_test.append(score)

    score = precision_score(y_test, y_pred)
    precision_test.append(score)

    score = precision_score(y_train, y_pred_train)
    precision_train.append(score)

    score = recall_score(y_test, y_pred)
    recall_test.append(score)

    score = recall_score(y_train, y_pred_train)
    recall_train.append(score)

    score = f1_score(y_test, y_pred)
    f1_test.append(score)

    score = f1_score(y_train, y_pred_train)
    f1_train.append(score)

    score = matthews_corrcoef(y_test, y_pred)
    MCC_test.append(score)

    score = log_loss(y_test, y_pred)
    Log_test.append(score)

print("Average time taken is", mean(time_taken), "secs")
print("Average training data accuracy:",mean(scores_train)*100,)
print("Average test data accuracy:",mean(scores_test)*100)
print("Average training data precision:",mean(precision_train)*100,)
print("Average test data precision:",mean(precision_test)*100)
print("Average training data recall:",mean(recall_train)*100,)
print("Average test data recall:",mean(recall_test)*100)
print("Average training data f1 score:",mean(f1_train)*100,)
print("Average test data f1 score:",mean(f1_test)*100)
print("Average test data MCC:",mean(MCC_test)*100)
print("Average test data Log_loss:",mean(Log_test)*100)


# In[ ]:


import matplotlib.pyplot as plt

models = ['Logistic Regression', 'SVM', 'Naive Bayes', 'Random Forest', 'Voting Classifier', 'XGBoost', 'LightGBM']
average_test_accuracy = [92.71, 100.0, 89.41, 100.0, 93.03, 100.0, 100.0]

plt.figure(figsize=(10, 6))
plt.bar(models, average_test_accuracy, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Average Test Accuracy (%)')
plt.title('Average Test Accuracy of Different Models')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




