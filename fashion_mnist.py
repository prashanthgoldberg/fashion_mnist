#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras


# In[3]:


import matplotlib.pyplot as plt


# In[9]:


f=keras.datasets.fashion_mnist


# In[11]:


(x_train_full,y_train_full),(x_test,y_test)=f.load_data()


# In[18]:


y_train_full[1]


# In[19]:


import tensorflow as tf


# In[46]:


x_train_n=x_train_full/255
x_test_n=x_test/255
x_valid,x_train=x_train_n[:50000],x_train_n[50000:]
y_valid,y_train=y_train_full[:50000],y_train_full[50000:]


# In[47]:


import numpy as np
np.random.seed(42)
tf.random.set_seed(42)


# In[48]:


ann=keras.models.Sequential()


# In[49]:


ann.add(keras.layers.Flatten(input_shape=[28,28]))
ann.add(keras.layers.Dense(300,activation='relu'))
ann.add(keras.layers.Dense(100,activation='relu'))
ann.add(keras.layers.Dense(10,activation='softmax'))
ann.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])


# In[50]:


ann.fit(x_train,y_train,epochs=50,validation_data=(x_valid,y_valid))


# In[51]:


y_pred=ann.predict(x_test)


# In[52]:


import pandas as pd


# In[56]:


d=pd.DataFrame(y_pred,y_test)


# In[57]:


d


# In[ ]:




