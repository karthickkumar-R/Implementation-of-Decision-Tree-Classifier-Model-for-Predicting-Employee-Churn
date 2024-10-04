# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import pandas.
   
2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:KARTHICKKUMAR R
RegisterNumber:212223040087
*/
```
```
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

![318640788-a5281883-eed0-439e-b4fa-922bbdfbed98](https://github.com/user-attachments/assets/b2b5a492-6404-4a1b-a997-16140eeda91c)

data.info()

![318640974-7c989f04-c2db-4949-a088-ad348bbec7c7](https://github.com/user-attachments/assets/e7c54750-5886-4d13-81f5-1020e20978bb)

data.isnull().sum()

![318641268-3d23a2f9-d8bf-4ca2-b00c-3e581ca80770](https://github.com/user-attachments/assets/5b7616dd-2a1f-4974-9b29-6121a2e2aeb8)

data value count

![318641864-10ee6c65-157b-4f16-95bc-0dc9664953a9](https://github.com/user-attachments/assets/a31ab0cd-dcdc-42fa-ae48-3cea2b366066)

data.head() for salary

![318642349-6492ed67-18af-46c1-8141-d168878dd59d](https://github.com/user-attachments/assets/4706d9d9-59fe-4169-86c9-c25707389296)

x.head()

![318642492-b3b08f53-9a93-4cdc-8403-6c601bae877e](https://github.com/user-attachments/assets/917f26b6-d9f0-4931-9240-34a7d36603b8)

accuracy value

![318642622-7026b909-59d8-4316-afaf-22270aa8dd90](https://github.com/user-attachments/assets/718be8ac-49a4-400e-927f-7f788de5bc26)

data prediction

![318642746-5f8fee59-9658-437e-8a98-8405059f69c6](https://github.com/user-attachments/assets/be015c7b-a6fe-46f2-8af4-778d7c033718)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
