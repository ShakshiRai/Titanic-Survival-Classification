#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load Titanic dataset
df = sns.load_dataset("titanic")


# In[2]:


df.info()


# In[3]:


df.head()


# In[4]:


# Drop columns with too many missing values and non-numeric columns
df.drop(['deck', 'embark_town', 'alive', 'class'], axis=1, inplace=True)


# In[6]:


# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)


# In[7]:


# Convert categorical features to numerical
# le = LabelEncoder()
# df['sex'] = le.fit_transform(df['sex'])
# df['embarked'] = le.fit_transform(df['embarked'])
df=pd.get_dummies(df,drop_first=True)


# In[8]:


df.dropna(inplace=True)

# Define features and target
X = df.drop(['survived'], axis=1)
y = df['survived']


# In[9]:


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[10]:


# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}


# In[11]:


# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# Convert results to DataFrame for comparison
results_df = pd.DataFrame(results).T
print(results_df)


# In[ ]:




