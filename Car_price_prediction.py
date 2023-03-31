#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import io


# In[70]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import scipy.stats
import scipy.optimize
import scipy.spatial


# In[6]:


df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')


# In[7]:


df.head(-5)


# In[8]:


df.describe()


# In[9]:


df.corr(method="pearson")


# There is little to no correlation between year and selling price and there is an inverse correlation between kilometers driven and selling price.

# In[10]:


print(df["owner"].unique())
print(df["transmission"].unique())
print(df["seller_type"].unique())
print(df["fuel"].unique())


# In[11]:


#check missing values
df.isnull().sum()


# In[12]:


df.columns


# In[13]:


car_data=df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner']]


# In[14]:


car_data.head(-5)


# In[15]:


car_data["current_year"]=2023


# In[16]:


car_data.head()


# In[17]:


car_data["usage_years"]=car_data["current_year"]- car_data["year"]


# In[18]:


car_data.head()


# In[19]:


car_data.drop(['year'],axis=1,inplace=True)


# In[20]:


car_data.drop(['current_year'],axis=1,inplace=True)


# In[21]:


car_data.head()


# In[22]:


car_data=pd.get_dummies(car_data,drop_first=True)


# In[23]:


car_data.head()


# In[24]:


car_data.corr()


# In[25]:


sns.pairplot(car_data)


# In[26]:


corrmap=car_data.corr()
top_corr_feat=corrmap.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(car_data[top_corr_feat].corr(),annot=True,cmap="RdYlGn")


# In[27]:


#Indepedent(X) and Dependent (Y) Features 
X=car_data.iloc[:,1:]
Y=car_data.iloc[:,0]


# In[28]:


X.head()


# In[29]:


Y.head()


# In[71]:


#Featue Importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,Y)


# In[72]:


print(model.feature_importances_)


# In[32]:


#plot graph of feature importance for visualisation
feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(8).plot(kind='barh')
plt.show()


# In[33]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[34]:


X_train.shape


# In[73]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()


# Hyperparameters:
# RandomForestRegressor(n_estimators=100,*,criterion='mse',max_depth=None,min_samples_split=2,min_samples_leaf=1,
# min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,
# bootstrap=True,oob_score=False,n_jobs=None,random_state=None,verbose=0,warm_start=False,ccp_alpha=0.0,max_samples=None,)

# In[75]:


#Hyperparameters
#Number of trees in random forest
n_estimators=[int(x) for x in np.linspace(start=100, stop= 1200,num=12)]
print(n_estimators)


# In[76]:


#Randomized Search CV (Heperparameter Tuning)
#number of features to consider at every split
max_features=["auto","sqrt"]
#Maximun number of levels in tree
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
max_depth.append(None)
#minimum number of samples required to split a node
min_samples_split= [2,5 ,10]
#minimum number of sample required at each leaf node
min_sample_leaf=[1,2,4]


# In[77]:


from sklearn.model_selection import RandomizedSearchCV


# In[78]:


#Create the random Grid
rf=RandomForestRegressor(random_state=42)
random_grid={"n_estimators":n_estimators,
            "max_features":max_features,
            "max_depth":max_depth,
            "min_samples_split":min_samples_split,
            "min_sample_leaf":min_sample_leaf}
print(random_grid)


# In[40]:


#rf=RandomForestRegressor()


# In[50]:


#rf_random=RandomizedSearchCV(estimator=rf, param_distributions=random_grid ,n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[79]:


rf_random=RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[80]:


rf_random.fit(X_train, Y_train)


# In[ ]:


estimator.get_params().keys()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


sorted(sklearn.metrics.SCORERS.keys())


# In[ ]:


estimator.get_params().keys()


# In[ ]:





# In[ ]:




