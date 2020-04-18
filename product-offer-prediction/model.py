# Kütüphanelerin Kurulumu
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder  # değişkenleri çeviren kütüphane
from sklearn.preprocessing import OneHotEncoder  # numeric değerleri başka kolonlara transform ederek flag atıyor
from sklearn.metrics import confusion_matrix
from sklearn import tree
import pickle
import warnings
warnings.filterwarnings("ignore")

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

# değişkenler ayrıştırılsın diye parçalanıyor
ikili_veriler = df.iloc[:, [2, 5, 7, 8, 9]].values
diger_veriler = df.iloc[:, [0, 1, 4, 6]].values
bolge = df.iloc[:, [3]].values

print("ikili/kategorik verilerin numpy arrayında replace edilerek numeric hale getirilmesi")
ikili_numeric = np.where(ikili_veriler == "NO", 0, ikili_veriler)
ikili_numeric = np.where(ikili_numeric == "MALE", 0, ikili_numeric)
ikili_numeric = np.where(ikili_numeric == "YES", 1, ikili_numeric)
ikili_numeric = np.where(ikili_numeric == "FEMALE", 1, ikili_numeric)
print(ikili_numeric)

print("region(bölge) değişkeninin numeric olarak tanımlanması")
ohe = OneHotEncoder(categorical_features="all")
bolge_n = ohe.fit_transform(bolge).toarray()

print("""
musteri datası parçalanarak kategorik veriler numerik hale getirildikten sonra tekrar
tek bir veri kümesi haline çeviriyoruz
""")
bolge_df = pd.DataFrame(data=bolge_n, index=range(300), columns=["INNER_CITY", "RURAL", "SUBURBAN", "TOWN"])
diger_veriler_df = pd.DataFrame(data=diger_veriler, index=range(300), columns=["id", "age", "income", "children"])
ikili_df = pd.DataFrame(data=ikili_numeric, index=range(300),
                        columns=['gender', 'married', 'car', 'product1', 'product2'])

print("tüm veriler birleşiyor")
df_all = pd.concat([diger_veriler_df, bolge_df, ikili_df], axis=1)

# train ve test kümelerimizi oluşturulması
print("ıd kolonlarını çıkararak sadece değişkenleri bırakıyoruz")
x = df_all.iloc[:, 1:13].values  # bağımsız değişken
y = df.iloc[:, [10]].values  # bağımlı değişken string

print("test ve train verileri hazırlanıyor...")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# karar ağacının oluşturulması
from sklearn.tree import DecisionTreeClassifier
DecisionTreeClassifier()
dtc = DecisionTreeClassifier(criterion="entropy", min_samples_split=8)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('DTC')
print(cm)

dosya = "model.save"
pickle.dump(dtc,open(dosya,"wb"))

model = pickle.load(open("NEW_PRODUCT_PREDICTION/model.save","rb"))

dtc_a = DecisionTreeClassifier(random_state=42)
# parametremetre optimizasyonu ve algoritma seçimi
from sklearn.model_selection import GridSearchCV

p = [{'min_samples_split': range(5, 20),
      "criterion":["entropy"],
      "min_samples_leaf": [17],
      "min_weight_fraction_leaf": [0.0]
      }]

'''
GSCV parametreleri
estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
param_grid : parametreler/ denenecekler
scoring: neye göre skorlanacak : örn : accuracy
cv : kaç katlamalı olacağı
n_jobs : aynı anda çalışacak iş
'''

gs = GridSearchCV(estimator=dtc_a,
                  param_grid=p,
                  scoring='accuracy',
                  cv=10,)

grid_search = gs.fit(x_train, y_train)
eniyisonuc = grid_search.best_score_
eniyiparametreler = grid_search.best_params_
print(eniyiparametreler)
print(eniyisonuc)
print(cm)

def yeniurun():
    age = int(input("Yaşınızı giriniz: "))
    income = int(input("Gelirinizi giriniz: "))
    children = int(input("Kaç çocuğunuz var:  "))
    bolge_input = int(input("""lütfen bir bolge seciniz\n
                  INNER_CITY için 1'e
                  RURAL için 2'e
                  SUBURBAN için 3'e
                  TOWN için 4'e basınız: """))
    if bolge_input == 1:
        region = [1, 0, 0, 0]
    elif bolge_input == 2:
        region = [0, 1, 0, 0]
    elif bolge_input == 3:
        region = [0, 0, 1, 0]
    else:
        region = [0, 0, 0, 1]

    cinsiyet = {"e": 0, "k": 1}
    cins = input("cinsiyetiniz e/k: ")
    gender = cinsiyet[cins]

    medeni = {"e": 1, "h": 0}
    evli = input("Evlimisiniz e/h: ")
    married = medeni[evli]

    araba = {"e": 1, "h": 0}
    arabasor = input("Aracınız var mı e/h: ")
    car = araba[arabasor]

    urunbir = {"e": 1, "h": 0}
    urbir = input("Ürün 1 var mı e/h: ")
    product1 = urunbir[urbir]

    uruniki = {"e": 1, "h": 0}
    uriki = input("Ürün 2 var mı e/h: ")
    product2 = uruniki[uriki]

    print("age {} income {} children {} region {} gender {} married {} car {} product1 {} product1 {}".format(
        age, income, children, region, gender, married, car, product1, product2))
    f = [age, income, children, gender, married, car, product1, product2]
    print(f)
    for i in region:
        f.insert(3, i)
    print(f)
    a = np.array([np.array(f)])
    print(a)
    sonuc = dtc.predict(a)
    print(sonuc)


# Skorlancak veri kümesini hesaplama
iv = score_cust.iloc[:, [2, 5, 7, 8, 9]].values
dv = score_cust.iloc[:, [0, 1, 4, 6]].values
bv = score_cust.iloc[:, [3]].values

print("ikili/kategorik verilerin numpy arrayında replace edilerek numeric hale getirilmesi")
inum = np.where(iv == "NO", 0, iv)
inum = np.where(inum == "MALE", 0, inum)
inum = np.where(inum == "YES", 1, inum)
inum = np.where(inum == "FEMALE", 1, inum)
print(inum)

print("region(bölge) değişkeninin numeric olarak tanımlanması")
ohe = OneHotEncoder(categorical_features="all")
bvn = ohe.fit_transform(bv).toarray()

b_df = pd.DataFrame(data=bvn, index=range(25), columns=["INNER_CITY", "RURAL", "SUBURBAN", "TOWN"])
dv_df = pd.DataFrame(data=dv, index=range(25), columns=["id", "age", "income", "children"])
iv_df = pd.DataFrame(data=inum, index=range(25), columns=['gender', 'married', 'car', 'product1', 'product2'])

print("tüm veriler birleşiyor")
df_skor_all = pd.concat([dv_df, b_df, iv_df], axis=1)

# train ve test kümelerimizi oluşturulması
print("ıd kolonlarını çıkararak sadece değişkenleri bırakıyoruz")
a = df_skor_all.iloc[:, 1:13].values  # bağımsız değişken

skorlama = dtc.predict(a)
durum = pd.DataFrame(skorlama, index=range(25), columns=["return"])
tum_data = pd.concat([score_cust, durum], axis=1)
print(tum_data)
tum_data.to_csv("NEW_PRODUCT_PREDICTION/DATA/output.csv", sep=',', encoding='utf-8')

a = np.array(['33' '3454.0' '3' '0' '0' '1' '0' '0' '1' '1' '1' '1'])
