# Exported notebook code; may contain IPython commands.
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re

from sklearn.neighbors import KernelDensity
pd.options.display.max_columns = None
from sklearn import preprocessing

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, Ridge  , LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor , RandomForestRegressor , HistGradientBoostingRegressor 
from sklearn.metrics import mean_absolute_percentage_error , r2_score
from sklearn.model_selection import train_test_split, cross_validate

# from xgboost import XGBRegressor


from sklearn.tree import DecisionTreeRegressor
from geopy.geocoders import Nominatim
from concurrent.futures import ThreadPoolExecutor

# %%
data= pd.read_csv('train.csv')
data_test= pd.read_csv('test.csv')

data

# %%
print(data.iloc[0]['description'])

# %%
data.info()

# %%
data['price']= data['price'].apply(lambda x: re.sub(r'[^\d]', '', x))
data['price']=data['price'].replace('' , '0')
data['price']=data['price'].astype('int64')

data['age']= data['age'].apply(lambda x: 0 if  x=='تازه ساز' else x)
data['age']=data['age'].apply(lambda x: str(x).replace('سال ساخت' , ''))
data['age']=data['age'].replace('nan' , np.nan)
data['age']=data['age'].astype('float64')

data['floor']=data['floor'].apply(lambda x: str(x).replace('همکف' , '0'))
data['floor']=data['floor'].astype('int64')

# %%
data.info()

# %%
#drop or fill 
data=data.dropna(subset=['bedrooms' , 'latitude' , 'longitude' , 'age' , 'neighborhood'])
data.reset_index(inplace=True , drop=True)


# %%
data.info()

# %%
#خونه هایی هست که سرش مغازه داره! :)
j=0
for i in range(data.index.max()):
    if 'مغازه' in str(data['description'][i]) and data['floor'][i]==0:
        print(data['description'][i] + '=======>' + str(i))
        print('///////////////////////////////////////////////////////////////////////////////////////////////////////')
        
        j+=1
        
print('sum of shop : '  + str(j))

# %%
#خونه هایی هست که سرش مغازه داره! :) in test data

j=0
for i in range(data_test.index.max()):
    if 'مغازه' in str(data_test['description'][i]) and data['floor'][i]==0 :
        print(data_test['description'][i] + '=======>' + str(i))
        print('///////////////////////////////////////////////////////////////////////////////////////////////////////')
        
        j+=1
        
print('sum of shop : '  + str(j))

# %%
data.describe()

# %%
#cleaning "age" column
data['age']=data['age'].apply(lambda x: 1402-x if x>100 else x)
data[data['age']==98]=1402-1398

# %%
index = data[(data['price']%1000)!=0].index
data.iloc[index]

# %%
j=0
for i in data[(data['price']%1000)!=0]['price'].index:
    print(str(data['price'][i]) + '====>' + str(i))
    j+=1
print(j)

# %%
data= data[~((data['price']%100000)!=0)]
data.reset_index(inplace=True , drop=True)
data

# %%
(data[data['type']=='فروش ویلایی']['price']==0).sum()

# %%
(data['price']==0).sum()

# %%
data['type'].value_counts()

# %%
data= data[~(data['price']==0)]
data.reset_index(inplace=True , drop=True)
data

# %%
data['price'].describe()

# %%
(data[data['area']>=1000]['type']=='فروش آپارتمان').sum()

# %%
data.info()

# %%
#drop data with area value more than 1000 AND type is apartemani
data = data[~((data['area'] >= 1000) & (data['type'] =='فروش آپارتمان'))]
data.reset_index(inplace=True , drop=True)
data

# %%
#drop data with area value less than 30
data= data[~(data['area']< 30)]
data.reset_index(inplace=True , drop=True)
data

# %%
data.describe()

# %%
#extract 
data['region']= data['neighborhood'].apply(lambda x: (str(x).strip().split(" ")[1])  if (len(str(x).strip().split(" ")))>1 else np.nan).astype('float32')

# %%
#drop record by region and price

#data =data[~((data['price'] > data['price'].quantile(0.999)) & (data['type']=='فروش آپارتمان') & (data['region'] != 9) & (data['region'] != 1))]
data =data[~((data['price'] > data['price'].quantile(0.99)))]
#data =data[~((data['price'] > data['price'].quantile(0.999)) & (data['type']=='فروش آپارتمان'))]
data.reset_index(inplace=True , drop=True)

# %%
data

# %%
#drop record by value < 0 in floor
data = data[~(data['floor']<0)]
data.reset_index(inplace=True , drop=True)
data

# %%
data.describe()

# %%
#new feature
data['price_meter']=data['price']/data['area']

# %%
data.describe()

# %%
data['type'][0] 

# %%
# 1 : vilaea & 0 : apartemani
data['type']=data['type'].apply(lambda x: 0 if x=='فروش آپارتمان' else 1)

# %%
data['type'][0] 

# %%
plt.figure(figsize=(10,10))
sns.scatterplot(data=data[data['type']==0], x="area", y="price", hue="region" , legend=True)

# %%
plt.figure(figsize=(10,10))
sns.scatterplot(data= data[(data['type']==1) &  (data['area'] < data['area'].quantile(0.99)) ] , x="area", y="price", hue="region" , legend=True)

# %%
plt.figure(figsize=(10,10))
sns.scatterplot(data=data[data['type']==0], x="age", y="price", hue="region" , legend=True)

# %%
plt.figure(figsize=(10,10))
sns.scatterplot(data= data[(data['type']==1) &  (data['area'] < data['area'].quantile(0.99)) ] , x="age", y="price", hue="region" , legend=True)

# %%
plt.figure(figsize=(10,10))
sns.scatterplot(data=data[data['type']==0], x="floor", y="price", hue="region" , legend=True)

# %%
plt.figure(figsize=(10,10))
sns.scatterplot(data= data[(data['type']==1) &  (data['area'] < data['area'].quantile(0.99)) ] , x="floor", y="price", hue="region" , legend=True)

# %%
data.info()

# %%
data.head(1)

# %%
data['area'].describe()

# %%
data['type'].value_counts()

# %%


# %%
data

# %%
X = data.drop(['price' , 'description' , 'neighborhood' , 'price_meter' ], axis=1)
#X = data.drop(['price' , 'description' , 'neighborhood'  ], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# %%
def fit_and_evaluate(model, X_train, y_train, X_test, y_test, tolerance=0.2):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # R2 Score
    r2 = r2_score(y_test, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape= mean_absolute_percentage_error(y_test, y_pred)
    
    #range accuracy
    correct_count = 0
    total_count = len(y_test)

    for true, predicted in zip(y_test, y_pred):
        if abs(true - predicted) <= tolerance*true:
            correct_count += 1

    accuracy = (correct_count / total_count)
    
    #Scatterplot
    plt.scatter(y_test, y_pred)
    plt.show()
    
    return r2, mape , accuracy

# %%
model = LinearRegression()

r2, mape , accuracy = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)

# %%
model = DecisionTreeRegressor(criterion='absolute_error',  max_depth=20 , random_state=5 )

r2, mape , accuracy = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)

# %%
model = DecisionTreeRegressor(criterion='squared_error',  max_depth=20 , random_state=5 )

r2, mape , accuracy = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)

# %%
model = GradientBoostingRegressor(loss='huber' , n_estimators=250 ,  max_depth=10 , random_state=5 )

r2, mape , accuracy = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)

# %%
data_test

# %%
model = RandomForestRegressor(n_estimators=500 ,  max_depth=300 , random_state=5 , n_jobs=-1 )

r2, mape , accuracy = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)

# %%


# %%
X_A = data[data['type']==0].drop(['price' , 'description' , 'neighborhood' , 'price_meter' , 'type' ], axis=1)
y_A = data[data['type']==0]['price']
price_bin_A=pd.cut(data[data['type']==0]['price'] , 10)
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A , y_A , test_size=0.05, random_state=42 , stratify=price_bin_A)


X_V = data[data['type']==1].drop(['price' , 'description' , 'neighborhood' , 'price_meter' , 'type'], axis=1)
y_V = data[data['type']==1]['price']
price_bin_V=pd.cut(data[data['type']==1]['price'] , 10)
X_train_V , X_test_V, y_train_V, y_test_V = train_test_split(X_V , y_V , test_size=0.05, random_state=42 , stratify=price_bin_V)



# %%
model_A = GradientBoostingRegressor(loss='huber' , n_estimators=250 ,  max_depth=10 , random_state=5 ,  )
model_V = GradientBoostingRegressor(loss='huber' , n_estimators=250 ,  max_depth=10 , random_state=5 ,  )

r2_A , mape_A , accuracy_A = fit_and_evaluate(model_A, X_train_A , y_train_A , X_test_A , y_test_A)
r2_V , mape_V , accuracy_V = fit_and_evaluate(model_V, X_train_V , y_train_V , X_test_V , y_test_V )

print("APARTEMAN R2 Score:", r2_A)
print("APARTEMAN MAPE:", (1-mape_A))
print("APARTEMAN acc:", accuracy_A)
print('APARTEMAN final score: ' ,  (r2_A + (1-mape_A) + accuracy_A )/3)


print("VILAEA R2 Score:", r2_V)
print("VILAEA MAPE:", (1-mape_V))
print("VILAEA acc:", accuracy_V)
print('VILAEA final score: ' ,  (r2_V + (1-mape_V) + accuracy_V )/3)


# %%
X
