import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path="/content/Bengaluru_House_Data.csv"
df_raw=pd.read_csv(path)
df_raw.shape

df_raw.head()

df=df_raw.copy()

df.info()

df.describe()

sns.pairplot(df)

def value_count(df):
  for var in df.columns:
    print(df[var].value_counts())
    print("----------------------------------")

value_count(df)

num_vars=["bath","balcony","price"]
sns.heatmap(df[num_vars].corr(),cmap="coolwarm",annot=True)

df.isnull().sum()

df.isnull().mean()*100

plt.figure(figsize=(16,9))
sns.heatmap(df.isnull())

df2 = df.drop('society',axis='columns')
df2.shape

df2['balcony']=df2['balcony'].fillna(df2['balcony'].mean())
df2.isnull().sum()

df3=df2.dropna()
df3.shape

df3.isnull().sum()

df.head()

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

df3['total_sqft'].value_counts()

total_sqft_int = []
for str_val in df3['total_sqft']:
  try:
    total_sqft_int.append(float(str_var))
  except:
    try:
      temp = []
      temp = str_val.split('-')
      total_sqft_int.append((float(temp[0])+float(temp[-1]))/2)
    except:
      total_sqft_int.append(np.nan)

df4 = df3.reset_index(drop=True)

df5 = df4.join(pd.DataFrame({'total_sqft_int':total_sqft_int}))
df5.head()

df.tail()

df5.isnull().sum()

df6=df5.dropna()
df6.shape

df6.info()

df6['size'].value_counts()

size_int=[]
for str_val in df6['size']:
  temp=[]
  temp = str_val.split(" ")
  try:
    size_int.append(int(temp[0]))
  except:
    size_int.append(np.nan)
    print("Noice =",str_val)

df6=df6.reset_index(drop=True)

df7=df6.join(pd.DataFrame({'bhk':size_int}))
df7.shape

df7.tail()

import scipy.stats as stats

def diagnostic_plots(df,variable):
  plt.figure(figsize=(16,4))
  plt.subplot(1,3,1)
  sns.distplot(df[variable],bins=30)
  plt.title('histogram')
  plt.subplot(1,3,2)
  stats.probplot(df[variable],dist="norm",plot=plt)
  plt.ylabel('Variable quantiles')
  plt.subplot(1,3,3)
  sns.boxplot(y=df[variable])
  plt.title('Boxplot')
  plt.show()

num_var = ["bath","balcony","total_sqft_int","bhk","price"]
for var in num_var:
  print("******* {} *******".format(var))
  diagnostic_plots(df7, var)

df7[df7['total_sqft_int']/df7['bhk']<350].head()

df8 = df7[~(df7['total_sqft_int']/df7['bhk'] < 350)]
df8.shape

df8['price_per_sqft'] = df8['price']*100000 / df8['total_sqft_int']
df8.head()


df8.price_per_sqft.describe()

def remove_pps_outliers(df):
  df_out = pd.DataFrame()
  for key, subdf in df.groupby('location'):
    m=np.mean(subdf.price_per_sqft)
    st=np.std(subdf.price_per_sqft)
    reduced_df = subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
    df_out = pd.concat([df_out, reduced_df], ignore_index = True)
  return df_out

df9 = remove_pps_outliers(df8)
df9.shape

def plot_scatter_chart(df,location):
  bhk2 = df[(df.location==location) & (df.bhk==2)]
  bhk3 = df[(df.location==location) & (df.bhk==3)]
  plt.figure(figsize=(16,9))
  plt.scatter(bhk2.total_sqft_int, bhk2.price, color='Blue', label='2 BHK', s=50)
  plt.scatter(bhk3.total_sqft_int, bhk3.price, color='Red', label='3 BHK', s=50, marker="+")
  plt.xlabel("Total Square Feet Area")
  plt.ylabel("Price")
  plt.title(location)
  plt.legend()

plot_scatter_chart(df9, "Rajaji Nagar")

plot_scatter_chart(df9, "Hebbal")

def remove_bhk_outliers(df):
  exclude_indices = np.array([])
  for location, location_df in df.groupby('location'):
    bhk_stats = {}
    for bhk, bhk_df in location_df.groupby('bhk'):
      bhk_stats[bhk]={
          'mean':np.mean(bhk_df.price_per_sqft),
          'std':np.std(bhk_df.price_per_sqft),
          'count':bhk_df.shape[0]}
    for bhk, bhk_df in location_df.groupby('bhk'):
      stats=bhk_stats.get(bhk-1)
      if stats and stats['count']>5:
        exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
  return df.drop(exclude_indices, axis='index')

df10 = remove_bhk_outliers(df9)
df10.shape

plot_scatter_chart(df10, "Hebbal")

df10.bath.unique()

df10[df10.bath > df10.bhk+2]

# here we are considering data only total no. bathroom =  bhk + 1
df11 = df10[df10.bath < df10.bhk+2]
df11.shape

plt.figure(figsize=(16,9))
for i,var in enumerate(num_var):
  plt.subplot(3,2,i+1)
  sns.boxplot(df11[var])

df11.head()

df12 = df11.drop(['area_type', 'availability',"location","size","total_sqft"], axis =1)
df12.head()

df12.to_csv("clean_data.csv", index=False)

df13 = df11.drop(["size","total_sqft"], axis =1)
df13.head()

df14 = pd.get_dummies(df13, drop_first=True, columns=['area_type','availability','location'])
df14.shape

df14.head()

df14.to_csv('oh_encoded_data.csv', index=False)

df13['area_type'].value_counts()

df15 = df13.copy()
# appy Ohe-Hot  encoding on 'area_type' feature
for cat_var in ["Super built-up  Area","Built-up  Area","Plot  Area"]:
  df15["area_type"+cat_var] = np.where(df15['area_type']==cat_var, 1,0)
df15.shape

df15.head(2)

df15["availability"].value_counts()

df15["availability_Ready To Move"] = np.where(df15["availability"]=="Ready To Move",1,0)
df15.shape

df15.tail()

location_value_count = df15['location'].value_counts()
location_value_count

location_gert_20 = location_value_count[location_value_count>=20].index
location_gert_20

df16 = df15.copy()
for cat_var in location_gert_20:
  df16['location_'+cat_var]=np.where(df16['location']==cat_var, 1,0)
df16.shape

df16.head()

df17 = df16.drop(["area_type","availability",'location'], axis =1)
df17.shape

df17.head()

df17.to_csv('ohe_data_reduce_cat_class.csv', index=False)

