# Cleaning Column Labels

# load datasets
import pandas as pd
df_08 = pd.read_csv('all_alpha_08.csv')
df_18 = pd.read_csv('all_alpha_18.csv')
# view 2008 dataset
df_08.head(1)
# view 2018 dataset
df_18.head(1)

# Drop Extraneous Columns

# drop columns from 2008 dataset
df_08.drop(['Stnd', 'Underhood ID', 'FE Calc Appr', 'Unadj Cmb MPG'], axis=1, inplace=True)
# confirm changes
df_08.head(1)

# drop columns from 2018 dataset
df_18.drop(['Stnd', 'Stnd Description', 'Underhood ID', 'Comb CO2'], axis=1, inplace=True)
# confirm changes
df_18.head(1)

# Rename Columns

# rename Sales Area to Cert Region
df_08.rename(columns={'sales_area': 'cert_region'}, inplace=True)
    ## we want to rename the column sales_area to cert_region, and want this change to occur in the original table (not create a new column)
# confirm changes
df_08.head(1)

# replace spaces with underscores and lowercase labels for 2008 dataset
df_08.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
    ## use rename again to change all columns (therefore using x intead of specific column name)
    ## we want to lowercase all column names (x.lower()) and replace all spaces with underscores (.replace(" ", "_"))
    ## these changes should edit the existing list and not creat a new one
    ## lambda simplifies the rename code above, which works as we are not focusing only on particular columns
# confirm changes
df_08.head(1)

# confirm column labels for 2008 and 2018 datasets are identical
df_08.columns == df_18.columns

# make sure they're all identical like this
(df_08.columns == df_18.columns).all()

# save new datasets for next section
df_08.to_csv('data_08_v1.csv', index=False)
df_18.to_csv('data_18_v1.csv', index=False)


# Filter, Drop Nulls, Dedupe

# Filter by Certification Region
# filter datasets for rows following California standards
df_08 = df_08.query('cert_region == "CA"')
df_18 = df_18.query('cert_region == "CA"')

# confirm only certification region is California
df_08['cert_region'].unique()
df_18['cert_region'].unique()

# drop certification region columns form both datasets
df_08.drop(['cert_region'], axis=1, inplace=True)
df_18.drop(['cert_region'], axis=1, inplace=True)

# Drop Rows with Missing Values
# view missing value count for each feature in 2008
df_08.info()
df_18.info()

# drop rows with any null values in both datasets
df_08.dropna(axis=0, how='any', inplace=True)
df_18.dropna(axis=0, how='any', inplace=True)

# checks if any of columns in 2008 have null values - should print False
df_08.isnull().sum().any()
df_18.isnull().sum().any()

# Dedupe Data

# print number of duplicates in 2008 and 2018 datasets
print(df_08.duplicated().sum())
print(df_18.duplicated().sum())

# drop duplicates in both datasets
df_08.drop_duplicates(inplace=True)
df_18.drop_duplicates(inplace=True)

# print number of duplicates again to confirm dedupe - should both be 0
print(df_08.duplicated().sum())
print(df_18.duplicated().sum())

# save progress for the next section
df_08.to_csv('data_08_v2.csv', index=False)
df_18.to_csv('data_18_v2.csv', index=False)


# Fixing `cyl` Data Type
# - 2008: extract int from string
# - 2018: convert float to int

# load datasets
import pandas as pd
df_08 = pd.read_csv('data_08_v2.csv')
df_18 = pd.read_csv('data_18_v2.csv')

# check value counts for the 2008 cyl column
df_08['cyl'].value_counts()

# Extract int from strings in the 2008 cyl column
df_08['cyl'] = df_08['cyl'].str.extract('(\d+)').astype(int)
df_08.head()

# Check value counts for 2008 cyl column again to confirm the change
df_08['cyl'].value_counts()

# convert 2018 cyl column to int
df_18['cyl'] = df_18["cyl"].astype(int)

# save
df_08.to_csv('data_08_v3.csv', index=False)
df_18.to_csv('data_18_v3.csv', index=False)


# Fixing `air_pollution_score` Data Type
# - 2008: convert string to float
# - 2018: convert int to float
# Load datasets `data_08_v3.csv` and `data_18_v3.csv`. You should've created these data files in the previous section: *Fixing Data Types Pt 1*.
# load datasets
import pandas as pd
df_08 = pd.read_csv('data_08_v3.csv')
df_18 = pd.read_csv('data_18_v3.csv')

# try using pandas' to_numeric or astype function to convert the
# 2008 air_pollution_score column to float -- this won't work
df_08["air_pollution_score"] = df_08["air_pollution_score"].astype(float)

# Figuring out the issue
# Looks like this isn't going to be as simple as converting the datatype. 
# According to the error above, the air pollution score value in one of the rows is "6/4" - let's check it out.

df_08[df_08.air_pollution_score == '6/4']
    ## how to find value within a column

# It's not just the air pollution score!
# The mpg columns and greenhouse gas scores also seem to have the same problem - maybe that's why these were all saved as strings! 
# According to this link, which I found from the PDF documentation:
# "If a vehicle can operate on more than one type of fuel, an estimate is provided for each fuel type."
# Ohh... so all vehicles with more than one fuel type, or hybrids, like the one above (it uses ethanol AND gas) will have a string that holds two values - one for each. 
# This is a little tricky, so I'm going to show you how to do it with the 2008 dataset, and then you'll try it with the 2018 dataset.

# First, let's get all the hybrids in 2008
hb_08 = df_08[df_08['fuel'].str.contains('/')]
hb_08

# hybrids in 2018
hb_18 = df_18[df_18['fuel'].str.contains('/')]
hb_18

# We're going to take each hybrid row and split them into two new rows - 
# one with values for the first fuel type (values before the "/"), and the other with values for the second fuel type (values after the "/"). 
# Let's separate them with two dataframes!

# create two copies of the 2008 hybrids dataframe
df1 = hb_08.copy()  # data on first fuel type of each hybrid vehicle
df2 = hb_08.copy()  # data on second fuel type of each hybrid vehicle
​
# Each one should look like this
df1

# columns to split by "/"
split_columns = ['fuel', 'air_pollution_score', 'city_mpg', 'hwy_mpg', 'cmb_mpg', 'greenhouse_gas_score']

# apply split function to each column of each dataframe copy
for c in split_columns:
    df1[c] = df1[c].apply(lambda x: x.split("/")[0])
    df2[c] = df2[c].apply(lambda x: x.split("/")[1])

# # this dataframe holds info for the FIRST fuel type of the hybrid
# aka the values before the "/"s
df1

# this dataframe holds info for the SECOND fuel type of the hybrid
# aka the values after the "/"s
df2

# combine dataframes to add to the original dataframe
new_rows = df1.append(df2)

# now we have separate rows for each fuel type of each vehicle!
new_rows

# drop the original hybrid rows
df_08.drop(hb_08.index, inplace=True)

# add in our newly separated rows
df_08 = df_08.append(new_rows, ignore_index=True)

# check that all the original hybrid rows with "/"s are gone
df_08[df_08['fuel'].str.contains('/')]


# Repeat this process for the 2018 dataset

# create two copies of the 2018 hybrids dataframe, hb_18
df1 = hb_18.copy()
df2 = hb_18.copy()

# Split values for fuel, city_mpg, hwy_mpg, cmb_mpg
# You don't need to split for air_pollution_score or greenhouse_gas_score here because these columns are already ints in the 2018 dataset.
# list of columns to split
split_columns = ['fuel', 'city_mpg', 'hwy_mpg', 'cmb_mpg']

# apply split function to each column of each dataframe copy
for c in split_columns:
    df1[c] = df1[c].apply(lambda x: x.split("/")[0])
    df2[c] = df2[c].apply(lambda x: x.split("/")[1])

# append the two dataframes
new_rows = df1.append(df2)

# drop each hybrid row from the original 2018 dataframe
# do this by using pandas' drop function with hb_18's index
df_18.drop(hb_18.index, inplace=True)

# append new_rows to df_18
df_18 = df_18.append(new_rows, ignore_index=True)

# check that they're gone
df_18[df_18['fuel'].str.contains('/')]

# Now we can comfortably continue the changes needed for air_pollution_score! Here they are again:
# 2008: convert string to float
# 2018: convert int to float

# convert string to float for 2008 air pollution column
df_08["air_pollution_score"] = df_08["air_pollution_score"].astype(float)

# convert int to float for 2018 air pollution column
df_18["air_pollution_score"] = df_18["air_pollution_score"].astype(float)

# save
df_08.to_csv('data_08_v4.csv', index=False)
df_18.to_csv('data_18_v4.csv', index=False)


# Fixing Data Types Part 3
# In this last section, you'll fix datatypes of columns for mpg and greenhouse gas score.
# After you complete these final fixes, check the datatypes of all features in both datasets to confirm success for all the changes we specified earlier. Here they are again for your reference:

# load datasets
import pandas as pd
df_08 = pd.read_csv('data_08_v4.csv')
df_18 = pd.read_csv('data_18_v4.csv')

# convert mpg columns to floats
mpg_columns = ['city_mpg', 'hwy_mpg', 'cmb_mpg']
for c in mpg_columns:
    df_18[c] = df_18[c].astype(float)
    df_08[c] = df_08[c].astype(float)

# Fix greenhouse_gas_score datatype
# 2008: convert from float to int

# convert from float to int
df_08['greenhouse_gas_score'] = df_08['greenhouse_gas_score'].astype(int)

# All the dataypes are now fixed! Take one last check to confirm all the changes.¶
df_08.dtypes == df_18.dtypes

# Save your final CLEAN datasets as new files!
df_08.to_csv('clean_08.csv', index=False)
df_18.to_csv('clean_18.csv', index=False)


# Drawing Conclusions
# Use the space below to address questions on datasets clean_08.csv and clean_18.csv

import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline

# load datasets
df_08 = pd.read_csv('clean_08.csv')
df_18 = pd.read_csv('clean_18.csv')

Q1: Are more unique models using alternative sources of fuel? By how much?
Let's first look at what the sources of fuel are and which ones are alternative sources.

df_08.fuel.value_counts()
Gasoline    984
gas           1
ethanol       1
CNG           1
Name: fuel, dtype: int64
df_18.fuel.value_counts()
Gasoline       749
Gas             26
Ethanol         26
Diesel          19
Electricity     12
# Name: fuel, dtype: int64
# Looks like the alternative sources of fuel available in 2008 are CNG and ethanol, and those in 2018 ethanol and electricity. 
# (You can use Google if you weren't sure which ones are alternative sources of fuel!)

# how many unique models used alternative sources of fuel in 2008
alt_08 = df_08.query('fuel in ["CNG", "ethanol"]').model.nunique()
alt_08

# how many unique models used alternative sources of fuel in 2018
alt_18 = df_18.query('fuel in ["Ethanol", "Electricity"]').model.nunique()
alt_18

plt.bar(["2008", "2018"], [alt_08, alt_18])
plt.title("Number of Unique Models Using Alternative Fuels")
plt.xlabel("Year")
plt.ylabel("Number of Unique Models");

# Since 2008, the number of unique models using alternative sources of fuel increased by 24. We can also look at proportions.

# total unique models each year
total_08 = df_08.model.nunique()
total_18 = df_18.model.nunique()
total_08, total_18

prop_08 = alt_08/total_08
prop_18 = alt_18/total_18
prop_08, prop_18

plt.bar(["2008", "2018"], [prop_08, prop_18])
plt.title("Proportion of Unique Models Using Alternative Fuels")
plt.xlabel("Year")
plt.ylabel("Proportion of Unique Models");


# Q2: How much have vehicle classes improved in fuel economy?
# Let's look at the average fuel economy for each vehicle class for both years.

veh_08 = df_08.groupby('veh_class').cmb_mpg.mean()
veh_08

veh_18 = df_18.groupby('veh_class').cmb_mpg.mean()
veh_18

# how much they've increased by for each vehicle class
inc = veh_18 - veh_08
inc

# only plot the classes that exist in both years
inc.dropna(inplace=True)
plt.subplots(figsize=(8, 5))
plt.bar(inc.index, inc)
plt.title('Improvements in Fuel Economy from 2008 to 2018 by Vehicle Class')
plt.xlabel('Vehicle Class')
plt.ylabel('Increase in Average Combined MPG');


# Q3: What are the characteristics of SmartWay vehicles? Have they changed over time?
# We can analyze this by filtering each dataframe by SmartWay classification and exploring these datasets.

# smartway labels for 2008
df_08.smartway.unique()
# get all smartway vehicles in 2008
smart_08 = df_08.query('smartway == "yes"')
# explore smartway vehicles in 2008
smart_08.describe()