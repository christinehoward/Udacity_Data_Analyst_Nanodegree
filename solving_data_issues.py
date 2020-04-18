#CLEANING DATA

# Solving missing value issues by inputing means

mean = df['view_duration'].mean()
df['view_duration'] = df['view_duration'].fillna(mean)
    ## 

mean = df['view_duration'].mean()
df['view_duration'].fillna(mean, inplace=True)
    ## replaces null values in the affected column


# Solving duplicated data issues

df.duplicated()
    ## considers a duplicate only when all values in columns match

sum(df.duplicated())
    ## count of duplicate rows

df.drop_duplicates(inplace=True)
    ## drop duplicates where duplicate = true

# Solving problem of incorrect data types

df['timestamp'] = pd.to_datetime(df('timestamp'))
    ## change date type to datetime und timestamp column


#CLEANING PRACTICE

# use means to fill in missing values
import pandas as pd
df = pd.read_csv('cancer_data_means.csv')
mean = df['texture_mean'].mean()
df['texture_mean'].fillna(mean, inplace=True)
mean = df['smoothness_mean'].mean()
df['smoothness_mean'].fillna(mean, inplace=True)
mean = df['symmetry_mean'].mean()
df['symmetry_mean'].fillna(mean, inplace=True)
# confirm your correction with info()
df.info()

# check for duplicates in the data
df.duplicated()
# drop duplicates
df.drop_duplicates(inplace=True)
# confirm correction by rechecking for duplicates in the data
df.duplicated()


#Renaming Columns
##Since we also previously changed our dataset to only include means of tumor features, the "_mean" at the end of each feature seems unnecessary. It just takes extra time to type in our analysis later. Let's come up with a list of new labels to assign to our columns.

## remove "_mean" from column names
new_labels = []
for col in df.columns:
    if '_mean' in col:
        new_labels.append(col[:-5])  # exclude last 6 characters
    else:
        new_labels.append(col)
​
## new labels for our columns
new_labels
## assign new labels to columns in dataframe
df.columns = new_labels
​
# display first few rows of dataframe to confirm changes
df.head()
# save this for later
df.to_csv('cancer_data_edited.csv', index=False)