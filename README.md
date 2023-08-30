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
![Screenshot 2023-08-24 083338](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/c4845eea-103e-4c4a-993d-bbd2481f3fd0)
![Screenshot 2023-08-24 083348](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/9043d9ca-e4d3-44f3-88d9-15a8ee95e582)
![Screenshot 2023-08-24 083355](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/c6a920a7-1c94-40dd-92dd-f8b0e4302411)
![Screenshot 2023-08-24 083406](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/d8542ce7-eefe-4f2b-8537-3342b9117bca)
![Screenshot 2023-08-24 083414](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/be21e611-429a-4ccb-a7b1-aafbfab115e4)
![Screenshot 2023-08-24 083422](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/90099227-fd69-474d-b8bb-f5972fb0f19a)
![Screenshot 2023-08-24 083429](https://github.com/Hemasonica774/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118361409/1eee6381-9bc0-421e-a3bf-d57da228295b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
