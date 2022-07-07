# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\360 CLASSES\project\real data\Real_Data_collection.csv")
df.info()
duplicate = df.duplicated()
duplicate.value_counts()
sum(duplicate)
# There is No Any Duplicated Values

df= df.rename(columns = {'Course Duration (In Hours)': 'Duration_Hours', 'Course name (like python , data science etc...)' :'Course_name','Course rating(Like 4.1, 3.0 etc..)':'Course_rating','Price of course(In Rupees)': 'Price','Trainer Grade(Low,Medium,High)':'Trainer_grade',
                             'Doubt Sessions Available':'Doubt_session','Recorded videos Provided':'Recorded_video','Practice Assignments Included':'Practise_assignment','Plat form or Institute':'Institute','Level Of Course':'Level','Mode Of program(Online, Inperson, Both, Other ,Null)':'Mode',
                             'Certification Provided':'Certification_Provided','Placement Assistance':'Placement_Assistance','Course Category':'Course_Category'})
df.columns
df = df.drop(['Course_Category', 'Course_rating'] , axis = 1)
# checking for null values
df.isna().sum()
# null values are found in each column
# for deal with null values using Imputation Method 

from sklearn.impute import SimpleImputer
# for course duration
df.Duration_Hours.median()
df.Duration_Hours.mean()
df.Duration_Hours.mode()
#df.Duration_Hours.skew()
#df.Duration_Hours.kurt()
#df.Duration_Hours.var()
#df.Duration_Hours.std()

# Here we are using mode imputation
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df["Duration_Hours"] = pd.DataFrame(mode_imputer.fit_transform(df[["Duration_Hours"]]))
df["Country"]= pd.DataFrame(mode_imputer.fit_transform(df[["Country"]]))
df["City"] = pd.DataFrame(mode_imputer.fit_transform(df[["City"]]))
df["Trainer_grade"] = pd.DataFrame(mode_imputer.fit_transform(df[["Trainer_grade"]]))
df["Practise_assignment"] = pd.DataFrame(mode_imputer.fit_transform(df[["Practise_assignment"]]))
df["Doubt_session"] = pd.DataFrame(mode_imputer.fit_transform(df[["Doubt_session"]]))
df["Institute"] = pd.DataFrame(mode_imputer.fit_transform(df[["Institute"]]))
df["Course_name"] = pd.DataFrame(mode_imputer.fit_transform(df[["Course_name"]]))
df["Mode"] = pd.DataFrame(mode_imputer.fit_transform(df[["Mode"]]))
df["Level"] = pd.DataFrame(mode_imputer.fit_transform(df[["Level"]]))
df["Certification_Provided"] = pd.DataFrame(mode_imputer.fit_transform(df[["Certification_Provided"]]))
df["Internship_Provided"] = pd.DataFrame(mode_imputer.fit_transform(df[["Internship_Provided"]]))
df["Recorded_video"] = pd.DataFrame(mode_imputer.fit_transform(df[["Recorded_video"]]))
df["Placement_Assistance"] = pd.DataFrame(mode_imputer.fit_transform(df[["Placement_Assistance"]]))
df.isnull().sum() 

df.Price = df.Price.str.replace(',','')
df.Price = df.Price.astype('float32')

df.Price.median()
df.Price.mean()
df.Price.mode()
df.Price.skew()
df.Price.kurt()

# Here using mean imputation for price 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Price"] = pd.DataFrame(mean_imputer.fit_transform(df[["Price"]]))
df.isnull().sum() 
df.info()

# Doing Transformation of numerical column from object to int
df.Duration_Hours = df.Duration_Hours.astype('int64')
df.Price = df.Price.astype('int64')
df.info()

# in price column some values are 0 then replace it 
zero = np.where(df['Price'] == 0 , True, False)
zero.sum()
# replacing those zeros by median
df.Price = np.where(df.Price == 0 , df.Price.median() , df.Price)
df.Price = df.Price.astype('int')

#df.to_csv("F:/Projects/EduTech Product Cost Prediction/2/cleaned data.csv", index=False)

sns.boxplot(data = df)
sns.boxplot(df.Duration_Hours)
sns.boxplot(df.Price)
# there are outliers so do outlier treatment

# outlier Treatment for Duration_Hours coumn
IQR = df['Duration_Hours'].quantile(0.75) - df['Duration_Hours'].quantile(0.25)
lower_limit = df['Duration_Hours'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Duration_Hours'].quantile(0.75) + (IQR * 1.5)

outliers = np.where(df['Duration_Hours'] > upper_limit, True, np.where(df['Duration_Hours'] < lower_limit, True, False))
outliers.sum()
outliers_values = df.Duration_Hours[outliers]
outliers_values

# Trimming extreme values
df = df.loc[(~ outliers),]
sns.boxplot(df.Duration_Hours);plt.title("Boxplot of n_enrolled after Trimming"); plt.show()

# Replacing remaining outliers in course duration
IQR = df['Duration_Hours'].quantile(0.75) - df['Duration_Hours'].quantile(0.25)
lower_limit = df['Duration_Hours'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Duration_Hours'].quantile(0.75) + (IQR * 1.5)

outliers = np.where(df['Duration_Hours'] > upper_limit, True, np.where(df['Duration_Hours'] < lower_limit, True, False))
outliers.sum()
outliers_values = df.Duration_Hours[outliers]
outliers_values

df.Duration_Hours = np.where(df.Duration_Hours > upper_limit, upper_limit,np.where(df.Duration_Hours < lower_limit, lower_limit, df.Duration_Hours))
sns.boxplot(df.Duration_Hours);plt.title("Boxplot of Duration_Hours after replacing"); plt.show()

# Outlier Treatment for Price
sns.boxplot(df.Price)
IQR = df['Price'].quantile(0.75) - df['Price'].quantile(0.25)
lower_limit = df['Price'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Price'].quantile(0.75) + (IQR * 1.5)

outliers = np.where(df['Price'] > upper_limit, True, np.where(df['Price'] < lower_limit, True, False))
outliers.sum()
outliers_values = df.Price[outliers]
outliers_values

# Trimming extreme values
df = df.loc[(~ outliers),]
sns.boxplot(df.Price);plt.title("Boxplot of Price after Trimming"); plt.show()
# check remaining outlier values
IQR = df['Price'].quantile(0.75) - df['Price'].quantile(0.25)
lower_limit = df['Price'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Price'].quantile(0.75) + (IQR * 1.5)

outliers = np.where(df['Price'] > upper_limit, True, np.where(df['Price'] < lower_limit, True, False))
outliers.sum()
outliers_values = df.Price[outliers]
outliers_values
# Those values are retained

sns.boxplot(data = df)

# Exporting Data 
#df.to_csv('F:/Projects/EduTech Product Cost Prediction/2/Dataset after outlier treatment.csv' ,index = False)

sns.distplot(df['Price'])
sns.distplot(np.log(df['Price']))

#labeleccoding for categorical columns
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df["Institute"]=lb.fit_transform(df.Institute)
df["Course_name"]=lb.fit_transform(df.Course_name)
df["Mode"]=lb.fit_transform(df.Mode)
df["Level"]=lb.fit_transform(df.Level)
df["Country"]=lb.fit_transform(df.Country)
df["City"]=lb.fit_transform(df.City)
df["Certification_Provided"]=lb.fit_transform(df.Certification_Provided)
df["Practise_assignment"]=lb.fit_transform(df.Practise_assignment)
df["Doubt_session"]=lb.fit_transform(df.Doubt_session)
df["Internship_Provided"]=lb.fit_transform(df.Internship_Provided )
df["Placement_Assistance"]=lb.fit_transform(df.Placement_Assistance)
df["Recorded_video"]=lb.fit_transform(df.Recorded_video)
df["Trainer_grade"]=lb.fit_transform(df.Trainer_grade)

y = np.log(df.Price)
x = df.drop('Price' , axis = 1)
#predictors = df.loc[:, df.columns!="price"]
#type(predictors)
#target = df["Price"]
#type(target)
df.columns
# Train Test partition of the data and perfoming the adaboost regressor as it has given best result in automl by pycaret
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state=2)

from sklearn.ensemble import AdaBoostRegressor as AR
from sklearn import metrics
regressor=AR(base_estimator=None,learning_rate=1.0,loss="linear",n_estimators=100,random_state=2)

regressor.fit(x_train,y_train)
#predicting a new result
y_pred=regressor.predict(x_test)
## accuracy score
from sklearn import metrics
r_square=metrics.r2_score(y_test, y_pred)
print(r_square)
mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
print(mean_squared_log_error)

#plotting the actual price and the predicted price
plt.plot(y_test,color="blue",label="actual_price")
plt.plot(y_pred,color="red",label="predicted_price")
plt.title("Actual_price vs Predicted_price")
plt.xlabel("values")
plt.ylabel("price")
plt.legend()
plt.show()
#save the model_ar to the disk
import pickle
filename="project.pkl"
pickle.dump(regressor,open(filename,"wb"))
project_ar=pickle.load(open("project.pkl","rb"))
