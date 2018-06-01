# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

#Dataset 1: Stanford Open Policing Project  
ri = pd.read_csv('police.csv')
ri.head()
ri.shape  #(91741,15)
ri.dtypes
#object - can be arbitrary python object but usually - String
#NaN indicaes values are missing.

ri.isnull().sum() #count of missing data in each columns
# ri.isnull() - is a data frame method that returns df of 
#true and false values. sum() -  row wise with true == 1 
# 1 **************Removing columns************************************************
'Remove the column that only contains missing values'

ri.drop('county_name',axis='columns',inplace=True) #default is inplace = False
#or ri = ri.drop('county_name',axis='columns') is ok
ri.shape  #(91741,14)
ri.dropna(axis='columns',how='all') #alternate method
# del ri['county_name'] will also work but it is not pandas
# 2 ***************Comparing groups****************************************************
'Do men or women speed more often'
#- driver_gender, violations columns will give an answer.

ri[ri.violation =='Speeding'].driver_gender.value_counts() 
#value_count not count. What if I need it as %
ri[ri.violation =='Speeding'].driver_gender.value_counts(normalize=True)

#When man is pulled over, how often it is for speeding and for women
ri[ri.driver_gender=='M'].violation.value_counts(normalize=True)
ri[ri.driver_gender=='F'].violation.value_counts(normalize=True)

#How to get two lines above in one line of code ?
ri.groupby('driver_gender').violation.value_counts(normalize=True)
'Group by can be read as  = FOR EACH'
#Convert into df type by unstack(): 
ri.groupby('driver_gender').violation.value_counts(normalize=True).unstack()
#loc can be used to select paricular row/col 
ri.groupby('driver_gender').violation.value_counts(normalize=True).loc[:,'Speeding']
# 3  ****************Examining relationships********************************************
'Does gender affect who gets searched during a stop ?'
#Relevant Columns -  driver_gender, search_conducted
ri.search_conducted.value_counts(normalize=True) #% of total veichle searched during their stop
ri.search_conducted.mean() #gives the same value in case of boolean
#When we take a mean of 0's and 1's we get % of 1 in the series.
ri.groupby('driver_gender').search_conducted.mean()
#M searched 4%, F searched 2%
ri.groupby(['violation','driver_gender']).search_conducted.mean()
#Can group by multipe columns, put them in a list, and list them as a string
# -  Males are searched getting searched for about every type of violation
#Causation is difficult to conclude manytimes, so focus on relationships
# 4 ********************Handling missing values***********************************************
'Why is search_type missing so often?'
ri.isnull().sum() # search_type           88545
ri.search_type.value_counts()

ri.search_conducted.value_counts()

ri[ri.search_conducted == False].search_type.value_counts()
#search_type is always is always null so there is no value to count
#by default in pandas methods dropna = True

ri[ri.search_conducted == False].search_type.value_counts(dropna=False)
#NaN 88545
# 5 ********************Using string methods************************************************
'During the search, how often is the driver frisked? - string search'

ri.search_type.value_counts(dropna=False)

#We need to identify all the cases where search_type value contains
#the string 'Protective Frisk'

#what method we use to search stings ?
#there are Python string methods
# and there are Pandas string methods
#Pandas string methods we apply across entire series

ri.search_type.str.contains('Protective Frisk')
#We can create a new column for frisk
ri['frisk']=ri.search_type.str.contains('Protective Frisk')
ri.frisk.value_counts(dropna=False)

ri.frisk.sum() #274 #True Values
ri.frisk.mean() #% of true values
#8.5% of the time when there is a seach, there is frisk
#Note - Pandas calculations exclude missing values.
# 6 ****************'Combining dates and times'**********
'Which year had the least number of stops'
'all series string methods start with str'
ri.dtypes
ri.stop_date.value_counts() #how to filter date by year?

ri.stop_date.str[-4:].value_counts() #both works
ri.stop_date.str.slice(-4,).value_counts() #both works

#concatinate two columns
combined = ri.stop_date.str.cat(ri.stop_time,sep=' ')
combined
#There is a datetime datatype in pandas- many useful features
#save it in a new column
ri['stop_datetime'] = pd.to_datetime(combined) #takes combined and converts into datetime data type.
ri.dtypes
ri.stop_datetime.dt.year #to get year #weekday #month etc
#like string methods have series.str.something, datetime has series.dt.something

ri.stop_datetime.dt.year.value_counts()
ri.stop_datetime.dt.year.value_counts().sort_values().index[0]
#Lessions: 
#Consider removing chunks of data that may be biased
#Use the datetime data type for dates and times
# 7 *****************Plotting a time series****************************
'How does drung activity change by time of day?'
###stop_datetime,drugs_related_stop

ri.drugs_related_stop.dtype #dtype('bool')
ri.drugs_related_stop.mean()
#What % of time during a stop is considered as drug related.

ri.groupby('hour').drugs_related_stop.mean()
# Will not work - there is no column name called hour

ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean()
#can group by that is not column name if we specify it in full

#How to plot this ?

ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean().plot() # .count() rather than .mean() gives slightly different answer
#default plot for a pandas series is a line plot

ri.groupby(ri.stop_datetime.dt.time).drugs_related_stop.mean().plot()
# .plot() saves time rather than matplotlib.
# 8  **************Creating useful plots***********************************
# Do most stops occur at night

ri.stop_datetime.dt.hour.value_counts() #gives series
#series has two methods for sorting -  sort_index() and sort_value() -changes right left directions.
ri.stop_datetime.dt.hour.value_counts().plot() #not good
ri.stop_datetime.dt.hour.value_counts().sort_index().plot() #fixes it
#assume 10pm-4am is night

ri[(ri.stop_datetime.dt.hour>4)&(ri.stop_datetime.dt.hour<22)]
ri[(ri.stop_datetime.dt.hour>4)&(ri.stop_datetime.dt.hour<22)].shape #(68575,16)
ri.shape #(91741, 16)
#number of stops at night = 91741 - 68575

#Be careful and sorting before plottig. does not sort everything automatically. 
# 9 *********************Fixing bad data**********************************************
'Find the bad data in the stop_duration column and fix it'
ri.stop_duration.value_counts(dropna=False)
#what is 1 and 2 ? may be bad data. How to fix it ? 

ri[ri.stop_duration==1|ri.stop_duration==2].stop_duration='NaN'
#There are 4 things wrong with this code.
# - NaN is not a string
# when using multiple conditions use parenthesis ()
#stop_duration is a string so == '1'.check dtype
# run and see 4 the error (warning but code did not run)- set on a copy of a slice from df
#use .loc and tel rows and col
ri.loc[(ri.stop_duration=='1')|(ri.stop_duration=='2'),'stop_duration']='NaN'
ri.stop_duration.value_counts() #shows one count NaN ,(here NaN is a string)
ri.stop_duration.value_counts(dropna=False) #shows 2 different count of NaN in the list

#fix it
import numpy as np
ri.loc[ri.stop_duration=='NaN','stop_duration']=np.nan
ri.stop_duration.value_counts(dropna=False)
#***********************************************************************

























































