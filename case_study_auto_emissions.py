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
df_08.to_csv('data_08_v4.csv', index=False)
df_18.to_csv('data_18_v4.csv', index=False)