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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print('df.head')

df.head()

print("df.tail")

df.tail()

X=df.iloc[:,:-1].values
print("Array value of X")
X

Y=df.iloc[:,1].values
print("Array value of Y")
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Values of Y prediction")
Y_pred

print("Array values of Y test")
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Test Set Graph")
plt.show()

print("Values of MSE, MAE and RMSE")
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/530e2e7a-2c42-43bd-8908-ba573d1f2f26)
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/d8e396a0-aa16-45c6-88c7-d2eab3ce9184)


![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/46380030-c855-4a5c-a6b7-4dd527920e16)


![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/66daa838-1bca-4ab1-afef-fe3673f16c5e)
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/509e9657-fb18-4e98-8c92-f0227936def4)
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/92c02cd7-f430-4717-9011-8eb0058b1358)
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/eddaa36c-8973-4951-8b42-135a89a0f7b2)
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/2f117fac-e23a-46d7-bdad-fa6f9e96a2a2)
![image](https://github.com/NamithaS2710/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133190822/e407b936-126b-4c37-bb11-f7fbc90c7161)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
