#GATHERING DATA

import pandas as pd 
df = pd.read_csv('winequality-red.csv', sep=';')
df.head()
    ##columns in csv file were separated by ; instead of , and this fixes the output

df = pd.read_csv('noheaders.csv')
df.head()
#OR
df = pd.read_csv('winequality-red.csv', sep=';', header = None)
df.head()

# ASSESSING DATA QUIZ

df = pd.read_csv('winequality-red.csv', sep=';')
df.shape
# rows 1599, columns 12

df = pd.read_csv('winequality-white.csv', sep=';')
df.shape
# rows 4898, columns 12

# Info about missing values
df.info()
winequality-red.csv
# 0 NaN
winequality-white.csv
# 0 NaN

#Info about duplicates
sum(df.duplicated())
winequality-red.csv
# 240 duplicates
winequality-white.csv
# 937 duplicates

# Info about unique values for quality
winequality-red.csv
# 6 unique
winequality-white.csv
# 7 unique

# Info about mean density
winequality-red.csv
# density 0.996747
winequality-white.csv
# density 0.994027

# Appending data



# import numpy and pandas
import pandas as pd
import numpy as np

# load red and white wine datasets
red_df = pd.read_csv('winequality-red.csv', sep=';')
white_df = pd.read_csv('winequality-white.csv', sep=';')
    ## created 2 datasets, for red and white wines

# Add arrays to the red and white dataframes
red_df['color'] = color_red
red_df.head()

white_df['color'] = color_white
white_df.head()

# Combine DataFrames with Append

# append dataframes
wine_df = red_df.append(white_df) 

# view dataframe to check for success
wine_df.head()

# Save combined dataset 
wine_df.to_csv('winequality_edited.csv', index=False)

# how to fix error due to different naming of same column in red_df.
# need to rename the misspelled column in red_df to match it in white_df.
red_df.rename(columns = {'total_sulfur-dioxide':'total_sulfur_dioxide'}, inplace=True)

# afterwards, we need to run the append statement again
wine_df = red_df.append(white_df)
# and we can check number of columns / rows in new table
wine_df.shape

# EDA with Visuals

# Load dataset
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('winequality_edited.csv')

# Histograms for Various Features
df.fixed_acidity.hist();
# OR
df['fixed_acidity'].hist();

df.total_sulfur_dioxide.hist();
# OR
df['total_sulfur_dioxide'].hist();

df.pH.hist();
# OR
df['pH'].hist();

df.alcohol.hist();
# OR
df['alcohol'].hist();

# Scatterplots of Quality Against Various Features

df.plot(x="volatile_acidity", y="quality", kind="scatter");
df.plot(x='alcohol', y='quality', kind='scatter'); etc.


# Pandas Groupby
## like GROUP BY SQL

wine_df = pd.read_csv('winequality_edited.csv')
wine_df.head()

wine_df.mean()
    ## mean for all columns in df

wine_df.groupby('quality').mean()
    ## to find the mean pH for each quality rating can use groupby along with the mean

wine_df.groupby(['quality', 'color']).mean()
## can also split the columns and provide multiple columns to group by

wine_df.groupby(['quality', 'color'], as_index=False).mean()

wine_df.groupby(['quality', 'color'], as_index=False)['pH'].mean()


# CONCLUSIONS GROUP BY

# Is a certain type of wine associated with higher quality?

# Find the mean quality of each wine type (red and white) with groupby
wine_df.groupby(['color'], as_index=False)['quality'].mean()
# OR
df.groupby('color').mean().quality


# What level of acidity receives the highest average rating?

# View the min, 25%, 50%, 75%, max pH values with Pandas describe
wine_df['pH'].describe()
# OR
df.describe().pH

# Bin edges that will be used to "cut" the data into groups
bin_edges = [2.72, 3.11, 3.21, 3.32, 4.01] # Fill in this list with five values you just found

# Labels for the four acidity level groups
bin_names = ['high', 'mod_high', 'medium', 'low'] # Name each acidity level category

# Creates acidity_levels column
df['acidity_levels'] = pd.cut(df['pH'], bin_edges, labels=bin_names)

# Checks for successful creation of this column
df.head()

# Find the mean quality of each acidity level with groupby
df.groupby('acidity_levels').mean().quality
# OR
wine_df.groupby(['acidity_levels'], as_index=False)['quality'].mean()

# Save changes for the next section
df.to_csv('winequality_edited.csv', index=False)


# Another useful function that we’re going to use is pandas' query function.

# In the previous lesson, we selected rows in a dataframe by indexing with a mask. 
# Here are those same examples, along with equivalent statements that use query().

# selecting malignant records in cancer data
df_m = df[df['diagnosis'] == 'M']
df_m = df.query('diagnosis == "M"')

# selecting records of people making over $50K
df_a = df[df['income'] == ' >50K']
df_a = df.query('income == " >50K"')

## The examples above filtered columns containing strings. 
## You can also use query to filter columns containing numeric data like this.

# selecting records in cancer data with radius greater than the median
df_h = df[df['radius'] > 13.375]
df_h = df.query('radius > 13.375')


# CONCLUSIONS USING QUERY

# Q1: Do wines with higher alcoholic content receive better ratings?
    ## To answer this question, use query to create two groups of wine samples:
    ## Low alcohol (samples with an alcohol content less than the median)
    ## High alcohol (samples with an alcohol content greater than or equal to the median)
    ## Then, find the mean quality rating of each group.

# get the median amount of alcohol content
df.alcohol.median()

# select samples with alcohol content less than the median
low_alcohol = df.query('alcohol < 10.3')

# select samples with alcohol content greater than or equal to the median
high_alcohol = df.query('alcohol >=10.3')

# ensure these queries included each sample exactly once
num_samples = df.shape[0]
num_samples == low_alcohol['quality'].count() + high_alcohol['quality'].count() # should be True

# get mean quality rating for the low alcohol and high alcohol groups
low_alcohol.quality.mean(), high_alcohol.quality.mean()


# Q2: Do sweeter wines (more residual sugar) receive better ratings?
    ## Similarly, use the median to split the samples into two groups by residual sugar and find the mean quality rating of each group.

# get the median amount of residual sugar
df.median().residual_sugar #OR 
df.residual_sugar.median()

# select samples with residual sugar less than the median
low_sugar = df.query('residual_sugar < 3.0')

# select samples with residual sugar greater than or equal to the median
high_sugar = df.query('residual_sugar >=3.0')

# ensure these queries included each sample exactly once
num_samples == low_sugar['quality'].count() + high_sugar['quality'].count() # should be True

# get mean quality rating for the low sugar and high sugar groups
low_sugar.quality.mean(), high_sugar.quality.mean()


# Plotting wine color and quality

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
% matplotlib inline
wine_df = pd.read_csv('winequality_edited.csv')
wine_df.head()

colors = ['red', 'white']
wine_df.groupby('color')['quality'].mean().plot(kind = 'bar', title='Average Wine Quality', alpha=.7)

    ## colors: changes colors of bars in graph
    ## alpha: transparency of bars
    ## graph shows average wine quality by color

plt.xlabel('Colors', fontsize=18)
plt.ylabel('Quality', fontsize=18)
    ## setting / editing x/y axis labels using pyplot

# can break up long line of code as follows:
color_means = wine_df.groupby('color')['quality'].mean()
color_means.plot(kind = 'bar', title='Average Wine Quality', alpha=.7)

counts = wine_df.groupby(['quality', 'color']).count()['pH']
counts.plot(kind = 'bar', title='Counts by Wine Color & Quality', color=colors, alpha=.7)
plt.xlabel('Quality and Color', fontsize=18)
plt.ylabel('Count', fontsize=18)
    ## plot a more detailed bar chart with ratings by color

totals = wine_df.groupby('color').count()['pH']
proportions = counts / totals
proportions.plot(kind = 'bar', title='Proportion by Wine Color & Quality', color=colors, alpha=.7)
plt.xlabel('Quality and Color', fontsize=18)
plt.ylabel('Proportion', fontsize=18)


# Creating a Bar Chart Using Matplotlib

import matplotlib.pyplot as plt
% matplotlib inline

# There are two required arguments in pyplot's bar function: the x-coordinates of the bars, and the heights of the bars.
plt.bar([1, 2, 3], [224, 620, 425]);

# You can specify the x tick labels using pyplot's xticks function, or by specifying another parameter in the bar function. The two cells below accomplish the same thing.

## plot bars
plt.bar([1, 2, 3], [224, 620, 425])​
## specify x coordinates of tick labels and their labels
plt.xticks([1, 2, 3], ['a', 'b', 'c']);

# plot bars with x tick labels
plt.bar([1, 2, 3], [224, 620, 425], tick_label=['a', 'b', 'c']);

# Set the title and label axes like this.
plt.bar([1, 2, 3], [224, 620, 425], tick_label=['a', 'b', 'c'])
plt.title('Some Title')
plt.xlabel('Some X Label')
plt.ylabel('Some Y Label');


# Plotting with Matplotlib
## Use Matplotlib to create bar charts that visualize the conclusions you made with groupby and query.

# Import necessary packages and load `winequality_edited.csv`
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
​
df = pd.read_csv('winequality_edited.csv')
df.head()

#1: Do wines with higher alcoholic content receive better ratings?
## Create a bar chart with one bar for low alcohol and one bar for high alcohol wine samples.

# Use query to select each group and get its mean quality
median = df['alcohol'].median()
low = df.query('alcohol < {}'.format(median))
high = df.query('alcohol >= {}'.format(median))
​   ## '{}'.format(median) is used here to refer to the median for alcohol
mean_quality_low = low['quality'].mean()
mean_quality_high = high['quality'].mean()
    ## ​low/high refer to queries for alcohol median above

# Create a bar chart with proper labels
locations = [1, 2]
heights = [mean_quality_low, mean_quality_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average Quality Ratings by Alcohol Content')
plt.xlabel('Alcohol Content')
plt.ylabel('Average Quality Rating');


#2: Do sweeter wines receive higher ratings?¶
## Create a bar chart with one bar for low residual sugar and one bar for high residual sugar wine samples.

# Use query to select each group and get its mean quality
median = df['residual_sugar'].median()
low = df.query('residual_sugar < {}'.format(median))
high = df.query('residual_sugar >= {}'.format(median))

mean_quality_low = low['quality'].mean()
mean_quality_high = high['quality'].mean()

# Create a bar chart with proper labels
locations = [1, 2]
heights = [mean_quality_low, mean_quality_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average Quality Ratings by Residual Sugar')
plt.xlabel('Residual Sugar')
plt.ylabel('Average Quality Rating');