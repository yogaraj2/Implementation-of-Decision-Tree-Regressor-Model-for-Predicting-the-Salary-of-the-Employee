# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the libraries and read the data frame using pandas
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.calculate Mean square error,data prediction and r2. 
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: YOGARAJ .S
RegisterNumber:  212223040248
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[['Position','Level']]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```
## Output:

![Screenshot 2024-04-02 202112](https://github.com/yogaraj2/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/153482637/35ce067d-6f28-4f18-919f-f66eb2b55e85)
 
![Screenshot 2024-04-02 202044](https://github.com/yogaraj2/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/153482637/b75311fb-247a-4dcb-8a85-40cb806d3b8c)

![Screenshot 2024-04-02 202038](https://github.com/yogaraj2/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/153482637/4afa5c80-fbff-465c-a599-f916538bbee9)

 DATA PREDICT :

![Screenshot 2024-04-02 202029](https://github.com/yogaraj2/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/153482637/63b4356c-33a8-41e6-941f-c2219b53ad0d)

DECISION TREE :
![Screenshot 2024-04-02 202009](https://github.com/yogaraj2/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/153482637/d7be5329-83c5-4312-a549-9c6d10a7bfa9)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
