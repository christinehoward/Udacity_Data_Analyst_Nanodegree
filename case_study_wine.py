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

# Apending data

# import numpy and pandas
import pandas as pd
import numpy as np

# load red and white wine datasets
red_df = pd.read_csv('winequality-red.csv', sep=';')
white_df = pd.read_csv('winequality-white.csv', sep=';')