import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# adet - x
count = [5,10,15,20,27,35,45,65,95,150,200]

# fiyat - y
price = [5,10,15,20,25,30,35,40,45,50,55]

# dataframe objesi
df = pd.DataFrame({"count":count,
                   "price":price})

# scatter plot gösterimi

plt.scatter(df["count"],
            df.price,
            s=200,
            c="#6baed6",
            edgecolors='black'
            )

plt.xlabel("Count")
plt.ylabel("Price")
plt.legend()
plt.show()


