
# loan approval project 
# LOGISTIC REGRESSION MODEL 

# import libraries 

import pandas as pd

# read the data and create a copy

loanData = pd.read_csv('loan file.csv')

loanprep = loanData.copy()

# identify the missing values 

loanprep.isnull().sum(axis = 0)

# drop the rows with missing values 

loanprep = loanprep.dropna()

loanprep.isnull().sum(axis = 0)

# drop irrelevent columns 

loanprep = loanprep.drop(['gender'],axis = 1)

loanprep.dtypes

# create dummies variables 

loanprep = pd.get_dummies(loanprep, drop_first = True)

# Normalize the data for income and loan amount 

from sklearn.preprocessing import StandardScaler 

Scaler_ = StandardScaler()

loanprep['income'] = Scaler_.fit_transform(loanprep[['income']])

loanprep['loanamt'] = Scaler_.fit_transform(loanprep[['loanamt']])

# create X and Y dataframe 

Y = loanprep[['status_Y']]
X = loanprep.drop(['status_Y'],axis = 1)

# split the x and y dataset into training and testing set 

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1234, stratify=Y)


# build logistric Regresser

from sklearn.linear_model import LogisticRegression
 
lr = LogisticRegression()

lr.fit(X_train, Y_train)

y_predict = lr.predict(X_test)

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(Y_test, y_predict)

score = lr.score(X_test, Y_test)

# PROJECT END 


