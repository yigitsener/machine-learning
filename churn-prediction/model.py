"""Data Preparation Library"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""Models Library"""
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

"""Model Evaluation"""
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

"""Other"""
import warnings
warnings.filterwarnings('ignore')

# Importing dataset
dt = pd.read_csv("data/churn-dataset.csv")

# Remove RowNumber, CustomerId and Surnmae
dt = dt.drop(columns=["RowNumber","CustomerId","Surname"])

# Descriptive statistics
print("First 5 Rows",dt.head(5))

# Dataset information
print(dt.info())

# Basic description
print(dt.describe())
description = dt.describe()
dt.describe().to_excel("table.xlsx",sheet_name="description")

# Missing values checking
print(dt.isnull().sum())

# Exited -- CreditScore
sns.violinplot( x=dt["Exited"], y=dt["CreditScore"], linewidth=5)
plt.show()

# Exited -- Age
sns.violinplot( x=dt["Exited"], y=dt["Age"], linewidth=5)
plt.show()

# Exited -- Tenure
sns.violinplot( x=dt["Exited"], y=dt["Tenure"], linewidth=5)
plt.show()

# Exited -- Balance
sns.violinplot( x=dt["Exited"], y=dt["Balance"], linewidth=5)
plt.show()

# Balance boxplot
dt.Balance.describe()
dt[["Balance"]].boxplot()
dt['Balance'].plot(kind='hist')

# Exited -- NumOfProducts
sns.violinplot( x=dt["Exited"], y=dt["NumOfProducts"], linewidth=5)
plt.show()

# Exited -- EstimatedSalary
sns.violinplot( x=dt["Exited"], y=dt["EstimatedSalary"], linewidth=5)
plt.show()

# Correlation Matrix
correlationColumns = dt[["CreditScore","Age","Tenure"
    ,"Balance","NumOfProducts","EstimatedSalary"]]

sns.set()
corr = correlationColumns.corr()
ax = sns.heatmap(corr
                 ,center=0
                 ,annot=True
                 ,linewidths=.2
                 ,cmap="YlGnBu")
plt.show()

"""Data Prepation"""
# Decomposition predictors and target
predictors = dt.iloc[:,0:10]
target = dt.iloc[:,10:]

# Gender encoder
predictors['isMale'] = predictors['Gender'].map({'Male':1, 'Female':0})
predictors.describe

# Geography one shot encoder
predictors[['France', 'Germany', 'Spain']] = pd.get_dummies(predictors['Geography'])

# Removal of unused columns.
predictors = predictors.drop(columns=['Gender','Geography','Spain'])
predictors.describe()

# Predictors Columns
predictors.describe()

# Transform data
normalization = lambda x:(x-x.min()) / (x.max()-x.min())
transformColumns = predictors[["Balance","EstimatedSalary","CreditScore"]]
predictors[["Balance","EstimatedSalary","CreditScore"]] = normalization(transformColumns)

# Train and test splitting
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.25, random_state=0)

"""Modelling"""
# Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred_dtc = dtc.predict(x_test)

dtc_cm = confusion_matrix(y_test,y_pred_dtc)
print("Decision Tree Confusion Matrix",dtc_cm)
dtc_acc = accuracy_score(y_test,y_pred_dtc)
print("Decision Tree Accuracy",dtc_acc)

# Logistic Regression
logr = LogisticRegression()
logr.fit(x_train,y_train)
y_pred_logr = logr.predict(x_test)

logr_cm = confusion_matrix(y_test,y_pred_logr)
print("Logistic Regression Confusion Matrix",logr_cm)
logr_acc = accuracy_score(y_test,y_pred_logr)
print("Logistic Regression Accuracy",logr_acc)

# Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred_gnb = gnb.predict(x_test)

gnb_cm = confusion_matrix(y_test,y_pred_gnb)
print("Naive Bayes Confusion Matrix",gnb_cm)
gnb_acc = accuracy_score(y_test,y_pred_gnb)
print("Naive Bayes Accuracy",gnb_acc)

# K Neighbors Classifier
knn = KNeighborsClassifier( metric='minkowski')
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)

knn_cm = confusion_matrix(y_test,y_pred_knn)
print("K Neighbors Classifier Confusion Matrix",knn_cm)
knn_acc = accuracy_score(y_test,y_pred_knn)
print("K Neighbors Classifier Accuracy",knn_acc)

# Random Forrest
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)

rfc_cm = confusion_matrix(y_test,y_pred_rfc)
print("Random Forrest Confusion Matrix",rfc_cm)
rfc_acc = accuracy_score(y_test,y_pred_rfc)
print("Random Forrest Accuracy",rfc_acc)

# Neural Network
clf = MLPClassifier()
clf.fit(x_train,y_train)
y_pred_clf = clf.predict(x_test)

clf_cm = confusion_matrix(y_test,y_pred_clf)
print("Random Forrest Confusion Matrix",clf_cm)
clf_acc = accuracy_score(y_test,y_pred_clf)
print("Random Forrest Accuracy",clf_acc)

# Xgboost Classifier
xgboast = xgb.XGBClassifier()
xgboast.fit(x_train, y_train)
print("Xgboast Classifier Accuracy",xgboast.score(x_test,y_test))

"""Model Evaluation"""
# prepare configuration for cross validation test harness
# all models including a list
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('xgboast', XGBClassifier()))

# evaluate each model in turning kfold results
results_boxplot = []
names = []
results_mean = []
results_std = []
p,t = predictors.values, target.values
for name, model in models:
    cv_results = cross_val_score(model, p,t, cv=10)
    results_boxplot.append(cv_results)
    results_mean.append(cv_results.mean())
    results_std.append(cv_results.std())
    names.append(name)
algorithm_table = pd.DataFrame({"Algorithm":names,
                                "Accuracy Mean":results_mean,
                                "Accuracy":results_std})
print(algorithm_table)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_boxplot)
ax.set_xticklabels(names)
plt.show()

#Grid Seach for XGboast
params = {
        'min_child_weight': [1, 2, 3],
        'gamma': [1.9, 2, 2.1, 2.2],
        'subsample': [0.4,0.5,0.6],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3,4,5]
        }
gd_sr = GridSearchCV(estimator=XGBClassifier(),
                     param_grid=params,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=1
                     )
gd_sr.fit(predictors, target)
best_parameters = gd_sr.best_params_
print(best_parameters)
best_result = gd_sr.best_score_
print(best_result)


# Surname investigation
# surname = dt["Surname"].value_counts().sort_values(ascending=False)
# surnameExited = dt[["Surname","Exited"]].groupby(["Surname"]).aggregate(["count","mean","sum"])
# surnameExited.columns = surnameExited.columns.droplevel()
# print(surnameExited.sort_values("count",ascending=False).head(20))
# moreThanOneSurname = surnameExited[surnameExited["count"] > 1]
# moreThanOneSurname[["count","sum"]].corr()
#
# x = moreThanOneSurname["count"]
# y = moreThanOneSurname["sum"]
#
# plt.scatter(x, y)
# z = np.polyfit(x, y, 1)
# p = np.poly1d(z)
# plt.plot(x,p(x),"r--")
# plt.show()

# Surname Exited Mean inner join to all data
# surnameExited_mean = surnameExited.reset_index()[["Surname","mean"]]
# dt = pd.merge(left=dt
#               ,right=surnameExited_mean
#              ,how="inner"
#              ,on="Surname")

# Reordering columns place
# dt = dt[['CustomerId', 'Surname', 'mean','CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited']]


# fig, axs= plt.subplots(2, 3)
# fig.suptitle('Sharing x per column, y per row')
# sns.violinplot(x=dt["Exited"], y=dt["CreditScore"], linewidth=5)
# sns.violinplot(x=dt["Exited"], y=dt["Age"], linewidth=5)
# sns.violinplot(x=dt["Exited"], y=dt["Tenure"], linewidth=5)
# sns.violinplot(x=dt["Exited"], y=dt["Balance"], linewidth=5)
# sns.violinplot(x=dt["Exited"], y=dt["NumOfProducts"], linewidth=5)
# sns.violinplot(x=dt["Exited"], y=dt["EstimatedSalary"], linewidth=5)
# plt.setp(axs, yticks=[])
# plt.tight_layout()
#
# f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
# names = ['CreditScore','Age','Tenure', 'Balance', 'NumOfProducts','EstimatedSalary']
# for i in names:
#     sns.violinplot(x=dt["Exited"], y=dt[i],ax=f)
#
# f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
# names = ['CreditScore','Age','Tenure', 'Balance', 'NumOfProducts','EstimatedSalary']
# fig, axes = plt.subplots(4, 2, figsize=(10, 16), sharey='row')
# axes_cols = (axes.flatten()[::2], axes.flatten()[1::2])
#
# for names, axes_col in zip(dt.groupby('Exited'), axes_cols):
#     for scale, ax in zip(['area', 'count', 'width'], axes_col[1:]):
#         sns.violinplot(x="day", y="total_bill", hue="smoker",
#             data=dt, split=True, ax=ax, scale=scale)
#         ax.set_title('scale = {}'.format(scale), y=0.95)
# sns.despine()
# fig.tight_layout()
