# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: P.Hemasonica
RegisterNumber: 22003246 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
```

## Output:
## Head

![Screenshot 2023-09-12 153046](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/49c4237b-a24c-4e6b-9a17-466966cccf6f)

## Tail

![Screenshot 2023-09-12 153120](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/5c8cf7cd-e506-4000-9e2e-e9a9c26fe17a)

## Array value of x

![Screenshot 2023-09-12 153157](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/dcd1f920-54e8-4882-8513-6aa5d2919289)

## Array value of y

![Screenshot 2023-09-12 153226](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/18ad968a-e82f-43dd-a4c3-65c565e546a3)

## Values of y prediction

![Screenshot 2023-09-12 153307](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/7a81f2ef-a958-472a-8a1f-b8652d30b891)

## Array value y test

![Screenshot 2023-09-12 153342](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/888b53ee-47f1-4944-bb30-e6bd819b88a2)

## Training set graph

![Screenshot 2023-09-12 153420](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/b424bf96-ba75-4d64-8768-eec0fc655ede)

## Test set graph

![Screenshot 2023-09-12 153456](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/d91f73e3-ca43-4fa3-b40b-7b72cb51e004)

## Values of MSE,MAE,RMSE

![Screenshot 2023-09-12 153521](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/baf61fb7-5894-4549-8cc1-5329fe956600)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
