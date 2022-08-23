#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('Salary.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.columns


# In[7]:


df.isnull().sum()


# In[8]:


import seaborn as sns


# In[9]:


sns.scatterplot(x="rank",y="salary",data=df)


# In[10]:


sns.scatterplot(x="discipline",y="salary",data=df)


# In[11]:


sns.scatterplot(x="yrs.since.phd",y="salary",data=df)


# In[12]:


sns.scatterplot(x="yrs.service",y="salary",data=df)


# In[13]:


sns.scatterplot(x="sex",y="salary",data=df)


# In[14]:


import matplotlib.pyplot as plt
sns.pairplot(df)
plt.savefig('pairplot.png')
plt.show()


# In[15]:


df.corr()


# In[16]:


df.corr()['salary'].sort_values()


# In[17]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),annot=True,linewidths=0.5,linecolor="black",fmt='.2f')


# In[18]:


df.describe()


# In[19]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
sns.heatmap(round(df.describe()[1:].transpose(),2),linewidth=2,annot=False,fmt='f')
plt.xticks(fontsize=18)
plt.yticks(fontsize=12)
plt.title("Variable summary")
plt.savefig('heatmaap.png')
plt.show()


# In[20]:


df.info()


# In[21]:


df.skew()


# In[24]:


sns.distplot(df["yrs.since.phd"])


# In[25]:


sns.distplot(df["yrs.service"])


# In[26]:


sns.distplot(df["salary"])


# In[57]:


df.corr()['salary']


# In[60]:


delete=pd.DataFrame([["0.334745","yrs.service","No","Alot"]],columns=["correlation with Target","Column NAme","Normalized","Outliers"])
delete


# In[61]:


df=df.drop(["yrs.service"],axis=1)


# In[62]:


df


# In[75]:


df.salary.value_counts()


# In[76]:


df.salary.unique()


# In[89]:


df.dtypes


# In[86]:


from scipy.stats import zscore


# In[87]:


import numpy as np


# In[88]:


np.abs(zscore(df))


# In[79]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
from sklearn.metrics import accuracy_score


# In[81]:


lr=LogisticRegression()
for i in range(0,1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=i,test_size=0.20)
    lr.fit(x_train,y_train)
    pred_train=lr.predict(x_train)
    pred_train=lr.predict(x_test)
    if round(accuracy_score(y_train,pred_train)*100,1)==round(accuracy_score(y_test,pred_test)*100,1):
        print("At random state",i,"The model perform very well")
        print("At random_state:-",i)
        print("Training accuracy score is:-",round(accuracy_score(y_train,pred_train)*100,1))
        print("Testing accuracy_score is:-",round(accuracy_score(y_test,pred_test)*100,1),'\n\n')


# In[ ]:




