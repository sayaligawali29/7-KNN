# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:37:47 2024

@author: user
"""

'''Prepare a classification model using the Naive Bayes algorithm
 for the salary dataset. Train and test
 datasets are given separately.
 Use both for model building
'''
'''Business Problem:
    Imagine you work for a company, and your HR department 
    is trying to streamline the hiring process.
    They have a dataset that includes information
    about employees, such as their education, work
    experience, and other relevant factors.
    The goal is to predict whether a candidate's salary will
    be above a certain threshold or not based on their
    attributes. 
    This prediction can be valuable for making informed 
    decisions during the hiring process and negotiating salaries.'''

'''Business Constraints:
    '''
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
salary=pd.read_csv("SalaryData_Train.csv")
salary.columns
'''O/p:
    Index(['age', 'workclass', 'education', 'educationno', 'maritalstatus',
           'occupation', 'relationship', 'race', 'sex', 'capitalgain',
           'capitalloss', 'hoursperweek', 'native', 'Salary'],
          dtype='object')'''
salary.shape
#It contain 30161 rows and 14 columns
#O/p:(30161, 14)
salary.describe()
'''o/p:
    age   educationno   capitalgain   capitalloss  hoursperweek
count  30161.000000  30161.000000  30161.000000  30161.000000  30161.000000
mean      38.438115     10.121316   1092.044064     88.302311     40.931269
std       13.134830      2.550037   7406.466611    404.121321     11.980182
min       17.000000      1.000000      0.000000      0.000000      1.000000
25%       28.000000      9.000000      0.000000      0.000000     40.000000
50%       37.000000     10.000000      0.000000      0.000000     40.000000
75%       47.000000     13.000000      0.000000      0.000000     45.000000
max       90.000000     16.000000  99999.000000   4356.000000     99.000000'''

#Checking for null values
salary.isnull().sum()
#It does not contain any null value

#Checking for duplicate values
salary.duplicated().sum()
#It has 3258 duplicate values

#Dropping of duplicates
salary.drop_duplicates(inplace=True)

#Now again checking for duplicates
salary.duplicated().sum()
#All duplicates are removed

#Converting data of game column into discrete data
#using one hot encoding

salary_new=pd.get_dummies(salary)
salary_new

salary_new.shape
#after creating dummy variables,we have 26903 rows and 104 columns

salary_new.columns

#Droping of one column as converting dummy variable of n columns 
#we need to drop one column to get n-1 columns 

salary_new.drop(salary_new['age'],axis=1,inplace=True)

#checking 5 number summery
des_salary=salary_new.describe()
des_salary


####Naive Bayes Algorithm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
###Loading Data
email_data=pd.read_csv("SalaryData_Train.csv",encoding="ISO-8859-1")
#####Cleaning of data
import re