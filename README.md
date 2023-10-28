# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Assigning hours To X and Scores to Y
3. Plot the scatter plot
4. Use mse,rmse,mae formmula to find 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Namitha.S
RegisterNumber: 212221040110 
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as pit

dataset = pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())

#hours to X
X = dataset.iloc[:,:-1].values
print(X)
#scores to Y
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)

plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Test Set Graph")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print("Values of MSE, MAE and RMSE")
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

a = np.array([[10]])

Y_pred1=reg.predict(a)
print(Y_pred1)

```

## Output:
1) df.head()
2) df.tail()
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/2f1fcc17-b717-4f4a-8200-677e8b5afe80)
3) Array value of X
4) Array value of Y   
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/3cbbea60-0280-4630-bcae-35775e5c951f)
5) Values of Y prediction
6) Array values of Y test
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/0b741438-ece4-4a2b-a9b7-e65e2ce0ab48)
7) Training set graph
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/04c3d49a-1264-4433-b96e-960933a77eb0)
8) Test set graph
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/2b998a20-17cb-437d-93c6-92f486b73f41)
9)Values of MSE,MAE and RMSE
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/1404d657-a77b-48b3-9eca-3d31f68ec63b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
