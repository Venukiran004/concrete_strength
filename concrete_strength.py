#!/usr/bin/env python
# coding: utf-8

# # CONCRETE COMPRESSIVE STRENGTH:
# 
# 
# ## INPUT:
# Cement - kg in a m3 mixture
# Blast- Furnace Slag kg in a m3 mixture,
# Fly Ash- kg in a m3 mixture,
# Water -kg in a m3 mixture,
# Superplasticizer- kg in a m3 mixture,
# Coarse Aggregate - kg in a m3 mixture,
# Fine Aggregate- kg in a m3 mixture,
# Age - Day (1~365).
# ## OUTPUT:
# Concrete compressive strength -MPa(megapascal)

# In[7]:


import os
os.chdir(r"C:\Users\venuk\OneDrive\Desktop\cement")


# In[8]:


os.getcwd()


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## File Loading and EDA:

# In[10]:


##reading dataset
my_data=pd.read_csv(r"C:\Users\venuk\OneDrive\Desktop\cement\concrete_data.csv")


# In[11]:


my_data.shape


# In[12]:


my_data.head()


# In[13]:


my_data.isna().sum()


# In[14]:


col_data=my_data.columns
col_data


# ## univariate analysis:

# In[15]:


for x in col_data:
    plt.hist(my_data[x])
    plt.xlabel(x)
    plt.show()


# ## Bivariate Analysis

# In[16]:


for x in col_data[:8]:
    plt.scatter(my_data[x],my_data["concrete_compressive_strength"])
    plt.xlabel(x)
    plt.ylabel("concrete_compressive_strength")
    plt.show()


# 
# ## Box Plot:

# In[17]:


sns.boxplot(data = my_data,width=0.8,orient="h") 


# In[18]:


X=my_data.drop(['concrete_compressive_strength'],axis=1)


# In[19]:


#important of feature
X.head()


# In[20]:


y=my_data["concrete_compressive_strength"]


# In[21]:


y.head()


# ## Feature Importance:

# In[22]:


from sklearn.ensemble import ExtraTreesRegressor
selection=ExtraTreesRegressor()
selection.fit(X,y)


# In[23]:


selection.feature_importances_


# In[24]:


importance_values=(selection.feature_importances_)*100
col_names_data=X.columns


# In[25]:


col_names_data


# In[26]:


importance_values


# In[27]:


fig = plt.figure()
ax = fig.add_axes([0,0,2,1])
col1=['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer',
       'coarse_aggregate', 'fine_aggregate ', 'age']
col2=[26.27097514,  8.52657896,  3.98846297,  8.09916696,  9.81645574,
        3.12761999,  4.45023623, 35.72050401]
ax.bar(col1,col2)
plt.show()


# ## Heat Map:

# In[28]:


plt.figure(figsize=(18,18))
sns.heatmap(my_data.corr(),annot=True)
plt.show()


# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, random_state= 30)


# In[30]:


X_train.shape


# In[31]:


X_test.shape


# In[32]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=25,criterion='mse',random_state=50)
model.fit(X_train,y_train)


# In[33]:


model.score(X_train,y_train)


# In[34]:


model.score(X_test,y_test)


# In[35]:


import pickle


# In[36]:


filename = 'cement_model.pkl'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:




