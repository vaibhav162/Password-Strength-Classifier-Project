#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries & Dataset

# In[47]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[48]:


data= pd.read_csv(r"C:\Users\shruti\Desktop\data.csv", error_bad_lines= False)


# In[49]:


data.head()


# In[50]:


data.tail()


# # Exploratry Data Analysis

# In[51]:


data.info()


# In[52]:


# Cheking for Null values

data.isnull().sum()


# In[53]:


# Checking exact position of Null value

data[data["password"].isnull()]


# In[54]:


# Dropping Null value

data.dropna(inplace = True)


# In[55]:


data.isnull().sum()


# In[56]:


data.describe()


# # Analyzing Data

# In[57]:


data["strength"].unique()


# In[58]:


# Plotting against "strength"

sns.countplot(data["strength"])


# In[59]:


password_data= np.array(data)


# In[60]:


password_data


# In[61]:


# Shuffling randomly for robustness

import random
random.shuffle(password_data)


# In[62]:


x= [labels[0] for labels in password_data]
y= [labels[1] for labels in password_data]


# In[63]:


x


# In[70]:


# Creating a custom function to convert input into characters of list

def word_divide_character(inputs):
    character=[]
    for i in inputs:
        character.append(i)
    return character


# In[73]:


word_divide_character("edcmki90")


# # Importing TF-IDF Vectorizer to convert String into numerical data

# In[75]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[76]:


vector= TfidfVectorizer(tokenizer=word_divide_character)


# In[77]:


# Applying TD-IDF Vectorizer on data

x= vector.fit_transform(x)


# In[79]:


x.shape


# In[81]:


vector.get_feature_names()


# In[82]:


first_document_vector= x[0]
first_document_vector


# In[83]:


first_document_vector.T.todense()


# In[84]:


df= pd.DataFrame(first_document_vector.T.todense(), index= vector.get_feature_names(), columns=["TF-IDF"])
df.sort_values(by=["TF-IDF"], ascending= False)


# # Splitting data into Train & Test

# In[85]:


# Train- To learn the realtionship within data
# Test- To predict the data and this data will be unseen to model

from sklearn.model_selection import train_test_split


# In[86]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)


# In[87]:


x_train.shape


# In[88]:


from sklearn.linear_model import LogisticRegression


# In[89]:


clf= LogisticRegression(random_state=0, multi_class="multinomial")


# In[90]:


clf.fit(x_train, y_train)


# In[92]:


# Predicting special custom data

dt= np.array(["%@123abcd"])
pred= vector.transform(dt)
clf.predict(pred)


# In[93]:


# Predicting x_test data

y_pred= clf.predict(x_test)
y_pred


# # Checking Accuracy of Model using Accuracy_score & Confusion_matrix

# In[94]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[95]:


cm= confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


# # Creating report of Model

# In[97]:


from sklearn.metrics import classification_report


# In[99]:


print(classification_report(y_test, y_pred))


# In[ ]:




