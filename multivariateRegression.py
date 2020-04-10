import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("DATA/Fish.csv")

print(df.info())
print(df.head(7))
print(df.describe())

# Species value count in dataset
print(pd.DataFrame(df.Species.value_counts()))

# Correlation
print(df.corr())

# Correlation heatmap
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df.corr(), annot=True,  linewidths=.5, ax=ax)
plt.show()

fig, axs = plt.subplots(5)
axs[0].scatter(df.Weight, df.Length1)
axs[1].scatter(df.Weight, df.Length2)
axs[2].scatter(df.Weight, df.Length3)
axs[3].scatter(df.Weight, df.Height)
axs[4].scatter(df.Weight, df.Width)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

x = [1,2,3,4,5]
y = [20,4,60,8,10]
z = [3,60,9,12,15]
t = [4,80,12,16,20]
w = [5,100,15,20,25]

fig = plt.figure()

plt.subplot(3, 2, 1)
plt.scatter(x, y)

plt.subplot(3, 2, 2)
plt.scatter(x, t)

plt.subplot(3, 2, 3)
plt.scatter(x, z)

plt.subplot(3, 2, 4)
plt.scatter(x, y)

plt.subplot(3, 2, 5)
plt.scatter(x, t)

plt.subplot(3, 2, 6)
plt.scatter(x, w)

plt.show()
