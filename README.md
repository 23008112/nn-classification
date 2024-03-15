# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![Screenshot 2024-03-11 124954](https://github.com/23008112/nn-classification/assets/138972470/4e2cb0b1-7452-4bc3-8b81-c124a0233b63)

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:
Load and read the dataset

### STEP 3:
Perform pre processinga and clean the dataset

### STEP 4:
Normalize the values and split the values for x and y.

### STEP 5:
Build the deep learning model with appropriate layers and depth.

### STEP 6:
Plot a grah for Training Loss, Validation Loss Vs Iteration &nfor Accuracy, Validation Accuracy Vs Iteration.

### STEP 7:
Save the model using pickle.

## PROGRAM

### Name: R. SANJANA
### Register Number: 212223240148

```
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

customer_df = pd.read_csv('customers.csv')

customer_df.columns

customer_df.dtypes

customer_df.shape

customer_df.isnull().sum()

clean = customer_df.dropna(axis=0)

clean.isnull().sum()

clean.shape

clean.dtypes

clean['Gender'].unique()

clean['Ever_Married'].unique()

clean['Graduated'].unique()

clean['Profession'].unique()

clean['Spending_Score'].unique()

clean['Var_1'].unique()

clean['Segmentation'].unique()

categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)

customers_1 = clean.copy()

customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])

customers_1.dtypes

le = LabelEncoder()

customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])

customers_1.dtypes

customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)

customers_1.dtypes

corr = customers_1.corr()

sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)

sns.pairplot(customers_1)

sns.distplot(customers_1['Age'])

plt.figure(figsize=(10,6))
sns.countplot(customers_1['Family_Size'])

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=customers_1)

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=customers_1)

customers_1.describe()

customers_1['Segmentation'].unique()

X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']]

y1 = customers_1[['Segmentation']].values

one_hot_enc = OneHotEncoder()

one_hot_enc.fit(y1)

y1.shape

y = one_hot_enc.transform(y1).toarray()

y.shape

y1[0]

y[0]

X.shape

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)

X_train[0]

X_train.shape

scaler_age = MinMaxScaler()

scaler_age.fit(X_train[:,2].reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

ai_brain = Sequential([
  Dense(8,input_shape=(8,)),
  Dense(8,activation='relu'),
  Dense(8,activation='relu'),
  Dense(4,activation='softmax'),
])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)

ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )

metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)

x_test_predictions.shape

y_test_truevalue = np.argmax(y_test,axis=1)

y_test_truevalue.shape

print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

ai_brain.save('customer_classification_model.h5')

with open('customer_data.pickle', 'wb') as fh:

ai_brain = load_model('customer_classification_model.h5')

with open('customer_data.pickle', 'rb') as fh:

x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)

print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))




```

## Dataset Information
![Screenshot 2024-03-15 120802](https://github.com/23008112/nn-classification/assets/138972470/198ae4fe-52c9-480d-ac83-cf479a2cfb68)


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-03-15 120944](https://github.com/23008112/nn-classification/assets/138972470/ce87873a-b6ce-4b00-8065-45f8610a895d)

### Classification Report
![Screenshot 2024-03-15 121051](https://github.com/23008112/nn-classification/assets/138972470/23284387-a7c3-42d8-8dde-679336f1c8a0)


### Confusion Matrix
![Screenshot 2024-03-15 121123](https://github.com/23008112/nn-classification/assets/138972470/b209ca12-3559-4c0a-a0e8-0ead386279aa)



### New Sample Data Prediction
![Screenshot 2024-03-15 121205](https://github.com/23008112/nn-classification/assets/138972470/8b8d8e6d-8036-4382-9718-ea3941f731f4)

Include your sample input and output here

## RESULT
A neural network classification model is developed for the given dataset
