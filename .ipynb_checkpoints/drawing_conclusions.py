# DRAWING CONCLUSIONS

import pandas as pd 
df= pd.read_csv('cancer_data_edited.csv')
df.head()

df_m = df[df('diagnosis') == 'M']
df_m['area'].describe()
    ##create a new dataframe containing malignant data
    ##df('diagnosis') == 'M' returns series of booleans, indicating whether value in diagnosis column is = 'M' (malignant)

df_b = df[df('diagnosis') == 'B']
df_b['area'].describe()
    ## does the same as above for benign tumors

df_week = df[df('week') == '2016-03-13']
df_week.describe()


#DRAWING CONCLUSIONS QUIZ

# imports and load data
import pandas as pd
% matplotlib inline

df = pd.read_csv('store_data.csv')
df.head()

# explore data
df.hist(figsize=(8, 8));

df.tail(20)

# total sales for the last month
df.iloc[196:, 1:].sum()

# average sales
df.mean()

# sales on march 13, 2016
df[df['week'] == '2016-03-13']

# worst week for store C
df[df['storeC'] == df['storeC'].min()]

# total sales during most recent 3 month period
df.iloc[187:, 1:].sum()
# OR
last_three_months = df[df['week'] >= '2017-12-01']   
last_three_months.iloc[:, 1:].sum()  
    ##sum across last 3 months
    ## exclude sum of week column

last_three_months = df[df['week'] >= '2017-12-01']
print (last_three_months)
    ##to show last 3 months (not totals

print (df)
    ##to show all rows/columns


# COMMUNICATE FINDINGS

import pandas as pd 
% matplotlib inline

df_census = pd.read_csv('census_income_data.csv')

df_a = df_census[df_census['income'] == '>50k']
df_b = df_census[df_census['income'] == '<=50k']
    ## create 2 separate groups: those earning above 50k and those learning <= 50k

df_a['education'].value_counts().plot(kind='bar');
    ## plot bar graph of education levels for group 1 (>50k)
df_b['education'].value_counts().plot(kind='bar');
    ## plot bar graph of education levels for group 2 (<=50k)
    ## x axises are different however

ind = df_a['education'].value_counts()
df_a['education'].value_counts()[ind].plot(kind='bar')
df_b['education'].value_counts()[ind].plot(kind='bar')
    ## now both charts have the same index, y-axis

ind = df_a['workclass'].value_counts().index
df_a['workclass'].value_counts()[ind].plot(kind='pie', figsize=(8, 8))
df_b['workclass'].value_counts()[ind].plot(kind='pie', figsize=(8, 8))
    ## same as above but with pie charts

df_a['age'].hist()
df_b['age'].hist()
    ## show distributions for each group

df_a['age'].describe()
df_b['age'].describe()
    ## summary statistics for both groups

##DRAWING CONCLUSIONS QUIZ PT.2

# sales for the last month
df.iloc[196:, 1:].sum().plot(kind='bar');
    ## bar chart of sales for the last month (sum)

# average sales
df.mean().plot(kind='pie');
`   ## pie chart of average sales`

# sales for the week of March 13th, 2016
df[df['week'] == '2016-03-13'].plot(kind='bar');
#OR
sales = df[df['week'] == '2016-03-13']
sales.iloc[0, 1:].plot(kind='bar');
    ## bar chart of sales the week of 13.03.2016

# sales for the lastest 3-month periods
last_three_months = df[df['week'] >= '2017-12-01']
last_three_months.iloc[:, 1:].sum().plot(kind='pie')
    ## pie chart of sales for the latest 3-month period

