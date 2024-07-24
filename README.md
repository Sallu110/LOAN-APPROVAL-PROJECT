
# Loan Approval Project
This project focuses on predicting loan approval status based on various applicant features using logistic regression.

## Steps of project
1. Introduction
2. Data Preprocessing
3. Model Training
4. Evaluation
5. Results 
6. Conclusion

### Introduction
In this project, I use a logistic regression model to predict whether a loan application will be approved based on applicant information such as income, loan amount, credit history, etc.

### Data Preprocessing
Import Libraries: Import pandas for data handling:
``` python
import pandas as pd

# Read and Clean Data: Read the dataset and handle missing values by dropping rows with NaN values:

loanData = pd.read_csv('loan_file.csv')
loanprep = loanData.dropna()
Drop Irrelevant Columns: Remove irrelevant columns like 'gender':

loanprep = loanprep.drop(['gender'], axis=1)
Create Dummy Variables: Convert categorical variables into dummy/indicator variables:

loanprep = pd.get_dummies(loanprep, drop_first=True)

# Normalize Data: Scale numeric features (income and loan amount) using StandardScaler:

from sklearn.preprocessing import StandardScaler
Scaler_ = StandardScaler()
loanprep['income'] = Scaler_.fit_transform(loanprep[['income']])
loanprep['loanamt'] = Scaler_.fit_transform(loanprep[['loanamt']])
```
### Model Training
``` python
# Split Data: Split the dataset into training and testing sets:

from sklearn.model_selection import train_test_split
X = loanprep.drop(['status_Y'], axis=1)
Y = loanprep[['status_Y']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

# Build Logistic Regression Model: Train the logistic regression model using scikit-learn:

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
```
### Evaluation
Confusion Matrix: Evaluate the model performance using a confusion matrix:
``` python
from sklearn.metrics import confusion_matrix
y_predict = lr.predict(X_test)
cm = confusion_matrix(Y_test, y_predict)

# Model Score: Calculate the accuracy score of the model:
score = lr.score(X_test, Y_test)
```

### RESULTS 

![Screenshot 2024-07-18 172517](https://github.com/user-attachments/assets/eef35ad0-244c-4d1a-87b9-7ee6f21305b5)

![Screenshot 2024-07-18 172548](https://github.com/user-attachments/assets/75ad3240-d7b5-4b22-acdb-a33515d05009)


### Conclusion
This project demonstrates the use of logistic regression for predicting loan approval status based on applicant features. The model's performance is evaluated using accuracy metrics and confusion matrix analysis.








