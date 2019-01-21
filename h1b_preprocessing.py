
#Import the necessary modules

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

#Load the data file

df= pd.read_csv("h1b.csv")

#Data cleaning and preprocessing

#Convert all column headers to lower case 

df.columns= df.columns.str.lower()

#Remove all descriptive columns

cols= ["unnamed: 0","case_number","employer_address","employer_business_dba",
"employer_name","employer_phone","employer_phone_ext"]

df.drop(cols,1,inplace=True)

#Remove all redundant geographical data- just keep the state variable

cols= ["employer_city","employer_postal_code","employer_country",
"employer_province","agent_attorney_city","worksite_city","worksite_county","worksite_postal_code"]

df.drop(cols,1,inplace=True)


#For some of these, more than 50% of the data in the column is missing. I'm going to write a function to remove these

def delete_col(col):
    if df[col].isnull().sum()/df.shape[0] >0.4:
        df.drop([col],1,inplace=True)

#Apply this function to all columns in the data frame

for col in df.columns:
	delete_col(col)

#Now that we have excluded columns with more than 50% of the total data missing, let us deal with the other columns
#We will impute the mean for the numerical columns for all missing values
#We will impute the mode for the categorical columns for all missing values


#Two functions - one for numerical cols and one for categorical

def imputer_categorical(column):
        if df[column].isnull().sum()>0:
            df[column]= df[column].fillna(df[column].mode()[0])
        return df

def imputer_numerical(column):
        if df[column].isnull().sum()>0:
            df[column]= df[column].fillna(df[column].mean())
        return df


#Separating numerical columns 
int_cols= list(df.select_dtypes(include='int').columns)
float_cols= list(df.select_dtypes(include='float64').columns)
num_cols= int_cols+ float_cols

#Applying the imputing function on all numerical columns

for col in num_cols:
    imputer_numerical(col)

#Apply the imputing categorical variables function on all the columns
#This works because the function has a check to see if there is any data missing
#For the numerical columns there will no longer be any missing values so only the categorical columns will be imputed


for col in df.columns:
    imputer_categorical(col)


#Dealing with date columns

import datetime

:
#Converting all dates into datetime objects
def date_converter(column):
    df[column]= pd.to_datetime(df[column])
    return df

cols = ["case_submitted","decision_date","employment_start_date","employment_end_date"]

for x in cols:
    date_converter(x)

#A function to subtract dates to get two new variables- employment_length,application_duration
def date_diff(final_col, end_date, start_date):
    df[final_col]= end_date - start_date
    df[final_col]= df[final_col].apply(lambda x:str(x))
    df[final_col]=df[final_col].str.split(" ")
    df[final_col]=df[final_col].apply(lambda x:x[0])
    return df[final_col].value_counts()


#Applying the function to get a new feature- Application duration
end_date = df.decision_date
start_date= df.case_submitted
final_col= "application_duration"
date_diff(final_col,end_date,start_date)
df.employment_length= df.employment_length.astype(int)

#Applying the function to get a new feature-Employment length
end_date = df.employment_end_date
start_date= df.employment_start_date
final_col= "employment_length"
date_diff(final_col,end_date,start_date)
df.application_duration= df.application_duration.astype(int)

#Now we can drop the original date columns
df.drop(["case_submitted","decision_date","employment_start_date","employment_end_date"],1,inplace=True)

#Some additional cleaning

#pw_source_year seems to have the wrong data type
df.pw_source_year= df.pw_source_year.astype(str)
df.pw_source_year= df.pw_source_year.str.split(".")
df.pw_source_year=df.pw_source_year.apply(lambda x:x[0])


#Dropping job title and soc code as they are redundant 
df.drop(["soc_code","job_title"],1,inplace=True)

#We only need certified and denied cases- I will first remove the withdrawn categories

final= df.loc[df["case_status"]!="WITHDRAWN"]
finale = final.loc[final["case_status"]!="CERTIFIED-WITHDRAWN"]
df= finale

#Dealing with the large class imbalance- through under sampling

#I am going to manually under sample the dominant class to reduce the class imbalance
denied= df.loc[df["case_status"]=="DENIED"]
certified= df.loc[df["case_status"]=="CERTIFIED"]
certified_sample= certified.sample(frac=0.02)
df= pd.concat([denied,certified_sample])
df= df.sample(frac=1)

#Save the preprocessed file as a pickle file
df.to_pickle("preprocessed_h1b.pkl")



