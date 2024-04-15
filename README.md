# EXP4 - Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## Aim:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages and print the present data.
   
2. Print the placement data and salary data.
   
3. Find the null and duplicate values.

4. Using logistic regression find the predicted values of accuracy, confusion matrices.
   
5. Display the results.

## Program:
~~~
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUNIL KUMAR T
RegisterNumber:  212223240164
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1.head()

x=data1.iloc[:,:-1]
print(x)

y=data1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import accuracy_score
confusion=(y_test,y_pred)
print(confusion)

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
~~~
## Output:
# DATASET
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/c766aca1-20e5-494c-9bd2-bf9d613d73c0)

# DATASET AFTER DROPPING THE SALARY COLUMN
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/e5d4d549-564a-477b-9995-0dcf529b2855)

# CHECKING IF NULL VALUES ARE PRESENT
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/363ab4b1-1a13-4a80-9018-bf9a629638fb)

# CHECKING IF DUPLICATE VALUES ARE PRESENT
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/397fbb5d-12f8-44c1-98c6-cdf834b35a15)

# DATASET AFTER ENCODING
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/860d4c34-2690-44c1-bedf-6e0fab7100f5)

# X-VALUES
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/18e69f47-5962-45fa-8fd0-6b493daf3de7)

# Y-VALUES
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/640cf836-9b8f-422a-928b-392bc19e156e)

# Y_PRED VALUES
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/bf69785e-33d1-4401-9201-948d5bfbef84)

# ACCURACY
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/48ee4758-73e1-4808-b32b-325f98ad9100)

# CONFUSION MATRIX
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/7441087f-341f-400e-b6ec-6098b1501327)

# CLASSIFICATION_REPORT
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/20fdb76e-6cf3-45ec-b3a0-14bc1edfb606)

# lr.predict
![image](https://github.com/K-Dharshini/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139334830/554e98ef-e368-4c7e-94ae-198cb6192d47)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
