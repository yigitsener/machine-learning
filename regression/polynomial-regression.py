import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

quantity = [5,10,15,20,27,35,45,65,95,150,200]

price = [5,10,15,20,25,30,35,40,45,50,55]

df = pd.DataFrame({"quantity":quantity,
                   "price":price})

plt.scatter(df["quantity"],
            df["price"],
            s=100,
            c="red",
            edgecolors='red'
            )

plt.title("Quantity & Price")
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.show()

x = df[["quantity"]]

y = df[["price"]] # fiyat

# Lienar Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

def linearRegressionVisual():
    plt.scatter(df["quantity"],
                df["price"],
                s=100,
                c="red",
                edgecolors='red'
                )
    plt.plot(x, lin_reg.predict(x), color='blue')
    plt.title('Basit Doğrusal Regresyon Sonucu')
    plt.xlabel('Miktar')
    plt.ylabel('Fiyat')
    plt.grid(True)
    plt.show()
    return
linearRegressionVisual()

print(lin_reg.predict([[80]]))
# Out: 34.5537831

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)

X_poly = poly_reg.fit_transform(x)

pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

def polynomialRegressionVisual():
    plt.scatter(df["quantity"],
                df["price"],
                s=100,
                c="red",
                edgecolors='red'
                )
    plt.plot(x, pol_reg.predict(poly_reg.fit_transform(x)), color='blue')
    plt.title('Basit Doğrusal Regresyon Sonucu')
    plt.xlabel('Miktar')
    plt.ylabel('Fiyat')
    plt.grid(True)
    plt.show()
    return
polynomialRegressionVisual()

print(pol_reg.predict(poly_reg.fit_transform([[80]])))
# Out: 43.10018674

