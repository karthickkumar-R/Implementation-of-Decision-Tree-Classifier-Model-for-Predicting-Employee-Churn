# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import pandas.
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KARTHICKKUMAR R
RegisterNumber:212223040087
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
data.head()

![370130923-b2b5a492-6404-4a1b-a997-16140eeda91c](https://github.com/user-attachments/assets/2f21913b-4af6-45b7-bf0f-f90fcd043e4d)

data.info()

![370130994-e7c54750-5886-4d13-81f5-1020e20978bb](https://github.com/user-attachments/assets/7f1048c6-9a0d-481c-acfe-506b1099f355)

data.isnull().sum()

![370131076-5b7616dd-2a1f-4974-9b29-6121a2e2aeb8](https://github.com/user-attachments/assets/31eb1764-2fca-4dba-9a0c-5ad8c758ea0d)

data value count

![370131131-a31ab0cd-dcdc-42fa-ae48-3cea2b366066](https://github.com/user-attachments/assets/e3da4062-9d76-437c-8872-6b5a0e2cfff4)


data.head() for salary

![370131182-4706d9d9-59fe-4169-86c9-c25707389296](https://github.com/user-attachments/assets/e5041166-e1e7-4c1c-aedd-b9af309ae557)

x.head()

![370131243-917f26b6-d9f0-4931-9240-34a7d36603b8](https://github.com/user-attachments/assets/ab619bc7-6e70-495e-bb2a-29cee2f4d416)

accuracy value

![370131304-718be8ac-49a4-400e-927f-7f788de5bc26](https://github.com/user-attachments/assets/a0124c8e-16c5-4dd7-8910-cf239cc9f597)

data prediction

![370131352-be015c7b-a6fe-46f2-8af4-778d7c033718](https://github.com/user-attachments/assets/89d31bf8-0cfd-446a-9cea-950982cbe1e0)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
