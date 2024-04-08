# EXP4-Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
   
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUNIL KUMAR T
RegisterNumber: 212223240164
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print(mse)

r2=metrics.r2_score(y_test,y_pred)
print(r2)

dt.predict([[5,6]])
```

## Output:
![Screenshot 2024-04-01 092450](https://github.com/MOHAMEDAHSAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331378/3ba3d398-6bfe-4ace-9970-46afdf78f006)
![Screenshot 2024-04-01 092509](https://github.com/MOHAMEDAHSAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331378/76ab28a5-b816-4150-a9b0-86ba766e803d)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
