# importing libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Creat dataset about age and income
age = [20,21,22,24,25,25,
       26,27,27,28,28,28,
       29,30,31,31,32,33,
       33,34,35,36]
income = [x**3/np.random.randint(2,4) for x in yas]

# Dataset involves pandas DataFrame
df = pd.DataFrame({"age":age,
                   "income":income})

# Scatter plot
plt.scatter(df.age,df.income)
plt.title("Distribution age and income")
plt.xlabel("Age")
plt.ylabel("İncome")
plt.show()

# Correlation
df.corr()
# 0.90023

x = df.age # independent variable
y_true = df.income # target, dependent variable

"""STATSMODELS"""
# Add const for statsmodels linear model
x = sm.add_constant(x)
# Runing model
model = sm.OLS(y_true,x).fit()
# Summary of model result table
print(model.summary())

# Model prediction existing age columns for predicting income
y_pred_stat = model.predict(x)

# Combining real income values with estimated income values
comparison = pd.DataFrame({'Real_income': y_true, 'Estimated_income': y_pred_stat}).sort_index()

# Error between real value and estimated value
comparison["Error_Difference"] = comparison.Real_income - comparison.Estimated_income
print(comparison)

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_ccpr(model,"age" ,ax=ax)
plt.title("Age - Income Estimation with Simple Linear Regression")
plt.xlabel("Age(X): Independent Variable")
plt.ylabel("Income(Y): Dependent Variable")
plt.show()


"""SKLEARN"""
lr = LinearRegression()
lr.fit(x,y_true)
y_pred_sklearn = lr.predict(x)

print("coefficients ",lr.coef_)
print("intercept",lr.intercept_)
print("r2 score",metrics.r2_score(y_true,y_pred_sklearn))
print('Mean Squared Error:', metrics.mean_squared_error(y_true, y_pred_sklearn))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_true, y_pred_sklearn)))
