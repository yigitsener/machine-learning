import pandas as pd
import numpy as np

# Creatin Example Dataframe
house = pd.DataFrame({"houseID":[2,4,6,8,10,12,14,16,17],
                   "numberOfRoom":[3,5,6,4,2,3,None,6,None],
                   "houseSquareMeter":[130,140,None,150,None,125,None,135,None],
                   "town": ["Manhattan", "Bronx",np.nan, "Queens", "Brooklyn","Staten Island","Malibu","Pacific Palisades",np.nan]})
print(house)

# Count of missing value based on variable
print(house.isnull().sum())

# Deletion listwise method
house1 = house.dropna()
print(house1)

# Deletion Variable Method
house2 = house.drop(columns="houseSquareMeter")
print(house2)

# Pairwise Method
# Add new column for pairwise method
house["housePrice"] =  [100000,125000,133000,138000,121000,105000,109000,108000,101000]
print(house)

# Only two variables selected for correlation analysis
print(house[["numberOfRoom","housePrice"]].dropna().corr())
# Correlation result = 0.301159

print(house[["houseSquareMeter","housePrice"]].dropna().corr())
# Correlation result  = 0.934123

# Last Observation Carried Forward
house.numberOfRoom.fillna(method="ffill")
# 0    3.0
# 1    5.0
# 2    6.0
# 3    4.0
# 4    2.0
# 5    3.0
# 6    3.0 Last Observation
# 7    6.0
# 8    6.0 Last Observation

# Next Observation Carried Backward
house.numberOfRoom.fillna(method="bfill")
# 0    3.0
# 1    5.0
# 2    6.0
# 3    4.0
# 4    2.0
# 5    3.0
# 6    6.0 Next Observation
# 7    6.0
# 8    NaN Next Observation There is no next

# Constant Value Imputation
house.numberOfRoom.fillna(value= "15")
# 0    3.0
# 1    5.0
# 2    6.0
# 3    4.0
# 4    2.0
# 5    3.0
# 6    15 Constant Value
# 7    6.0
# 8    15 Constant Value

# Mean Imputation
house.mean()
# houseID                  9.888889
# numberOfRoom             4.142857
# houseSquareMeter       136.000000
# housePrice          115555.555556
# dtype: float64
house_mean = house.fillna(house.mean())
print(house_mean.houseSquareMeter)
# 0    130.0
# 1    140.0
# 2    136.0
# 3    150.0
# 4    136.0 mean
# 5    125.0
# 6    136.0 mean
# 7    135.0
# 8    136.0 mean

# Median
print(house.numberOfRoom.fillna(house.numberOfRoom.median()))
# 0    3.0
# 1    5.0
# 2    6.0
# 3    4.0
# 4    2.0
# 5    3.0
# 6    4.0 median
# 7    6.0
# 8    4.0 median

# mode
print(house.numberOfRoom.fillna(house.numberOfRoom.mode()[1]))
# 0    3.0
# 1    5.0
# 2    6.0
# 3    4.0
# 4    2.0
# 5    3.0
# 6    6.0 mode
# 7    6.0
# 8    6.0 mode

# Lineer regression
print(house.houseSquareMeter.interpolate(method="linear"))
# 0    130.0
# 1    140.0
# 2    145.0 linear
# 3    150.0
# 4    137.5 linear
# 5    125.0
# 6    130.0 lineer
# 7    135.0
# 8    135.0 linear

# interplolate Methods
# method : {‘linear’, ‘time’, ‘index’, ‘values’,
#           ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’,
#           ‘cubic’, ‘barycentric’, ‘krogh’, ‘polynomial’,
#           ‘spline’, ‘piecewise_polynomial’,
#           ‘from_derivatives’, ‘pchip’, ‘akima’}

# All pandas null/missing values fonctions
# isnull()
# notnull()
# dropna()
# fillna()
# replace()
# interpolate()
