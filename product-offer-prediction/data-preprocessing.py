import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

line = "-------------------------------------"

#Data Importing
cust_1 = pd.read_csv("machine-learning/data/product-offer-data/customers_1.txt")
cust_2 = pd.read_csv("machine-learning/data/product-offer-data/customers_2.txt")
target = pd.read_excel("machine-learning/data/product-offer-data/return.xlsx")
score_cust = pd.read_csv("machine-learning/data/product-offer-data/test_customers.txt")

#Append cust_1 and cust_2
customers = cust_1.append(cust_2,ignore_index=True)

#convert dataframe
data_df = pd.DataFrame(customers)
tg = pd.DataFrame(target)

#Inner join; customers and target data
df = data_df.merge(tg,on="id",how= "inner")

#df overview
print(df.columns,df.shape)
print(line)
print(df.head())

print(line)
#df description
print(df.describe())

print(line)
#Değişkenlerdeki tekil veri sayıları
print(df.nunique())
print("""
uniqe values checking
1- ID column is uniqe
2- Age and Income are continuous
3- Gender, Region, Married, Children, Car, Product1, Product2 and return are categorical variables
""")

print(line)
#missing value analysis
print(df.isnull().sum())
print("there is no missing value")

print(line)
#Tüm değişkenlerin dağılımları
print("Gender variable...")
plt.bar(["MALE","FEMALE"],df["gender"].value_counts())
plt.xlabel("GENDER")
plt.show()

print(line)
print("Region variable...")
plt.bar(["INNER CITY","TOWN","RURAL","SUBURBAN"],df["region"].value_counts())
plt.xlabel("BÖLGE")
plt.show()

print(line)
print("Married variable...")
plt.bar(["Married","Single"],df["married"].value_counts())
plt.xlabel("MARRIED")
plt.show()

print(line)
print("Children variable...")
plt.bar(["0","1","2","3"],df["children"].value_counts())
plt.xlabel("Number Of Children")
plt.show()

print(line)
print("Car Variable...")
plt.bar(["YES","NO"],df["car"].value_counts())
plt.xlabel("Car owner")
plt.show()

print(line)
print("product1 and product2 variables")
print(df["product1"].value_counts(normalize=True))
print(df["product2"].value_counts(normalize=True))

print(line)

#numeric variables
print("Income variable...")
sns.distplot(df['income'])
plt.xlabel("Income")
plt.show()
print("skewness and kurtosis values")
print(df["income"].skew(),df["income"].kurt())

print(line)

print("Age Values")
sns.distplot(df['age'])
plt.show()
print("skewness and kurtosis values")
print(df["age"].skew(),df["age"].kurt())
