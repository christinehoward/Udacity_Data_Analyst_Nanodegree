# Part I - Probability
# To get started, let's import our libraries.

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)

# 1. Now, read in the ab_data.csv data. Store it in df. Use your dataframe to answer the questions in Quiz 1 of the classroom.

# a. Read in the dataset and take a look at the top few rows here:

df = pd.read_csv('ab_data.csv')
df.head()
user_id	timestamp	group	landing_page	converted
0	851104	2017-01-21 22:11:48.556739	control	old_page	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0
4	864975	2017-01-21 01:52:26.210827	control	old_page	1

# b. Use the cell below to find the number of rows in the dataset.

df.shape
(294478, 5)

# c. The number of unique users in the dataset.

df['user_id'].nunique()
290584

# d. The proportion of users converted.

converted_df = df.query('converted == 1').user_id.nunique() / df['user_id'].nunique()
print(converted_df)
0.12104245244060237
# e. The number of times the new_page and treatment don't match.

count_newpage_treatment_nomatch = df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == False].shape[0]

# f. Do any of the rows have missing values?

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 294478 entries, 0 to 294477
Data columns (total 5 columns):
user_id         294478 non-null int64
timestamp       294478 non-null object
group           294478 non-null object
landing_page    294478 non-null object
converted       294478 non-null int64
dtypes: int64(2), object(3)
memory usage: 11.2+ MB

# 2. For the rows where treatment does not match with new_page or control does not match with old_page, we cannot be sure if this row truly received the new or old page. Use Quiz 2 in the classroom to figure out how we should handle these rows.
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz. Store your new dataframe in df2.

newpage_treatment_nomatch_result = df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == False]
# we looked at the group and landing page columns to compare the rows in these columns to see where they did not match (have false result)
df2 = df.drop(list(newpage_treatment_nomatch_result.index.values), inplace=True)
# we use the indexes from the result to drop the rows from the original df (inplace=True)
# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
0

# 3. Use df2 and the cells below to answer questions for Quiz3 in the classroom.

# a. How many unique user_ids are in df2?

df2.user_id.nunique()
290584

# b. There is one user_id repeated in df2. What is it?

df2.loc[df2['user_id'].duplicated()].index.values[0]
2893

# c. What is the row information for the repeat user_id?

df2.loc[df2['user_id'].duplicated()]
user_id	timestamp	group	landing_page	converted
2893	773192	2017-01-14 02:55:59.590927	treatment	new_page	0

# d. Remove one of the rows with a duplicate user_id, but keep your dataframe as df2.

df2.drop(df2.index[2893], inplace=True)

# 4. Use df2 the cells below to answer the quiz questions related to Quiz 4 in the classroom.

# a. What is the probability of an individual converting regardless of the page they receive?

# first find the number of converts (converted is 1) 
count_converted_true = df2.query('converted == 1').converted.count()
​
# then find the total number of users
total_converted = df2.converted.count()
​
# then divide the number of converts by total users
convert_probability = count_converted_true/total_converted
​
print(convert_probability)
0.119597498821

# b. Given that an individual was in the control group, what is the probability they converted?

# find number of individuals in control group
​
control_group = df2.query('group == "control"')
​
count_control_group = df2.query('group == "control"').group.count()
​
# how many of these users converted?
​
count_control_convert = control_group.query('converted == 1').converted.count()
# 17489 total users in control group who have converted
​
# divide by total number users
control_convert_prob = count_control_convert/count_control_group
print(control_convert_prob)
0.1203863045

# c. Given that an individual was in the treatment group, what is the probability they converted?

# find number of individuals in treatment group
​
treatment_group = df2.query('group == "treatment"')
​
probability_treatment_converted = len(treatment_group.query('converted == 1'))/len(treatment_group)
print(probability_treatment_converted)
0.11880888313869065

# d. What is the probability that an individual received the new page?

landing_page = df2.query('landing_page == "new_page"')
​
probability_landing_page = len(landing_page) / len(df2)
print(probability_landing_page)
0.5000602237570677

# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# It does not look like there is sufficient evidence to conclude that the new landing page had led to more conversions. 
# The probabilities above actually show the opposite, with a slightly higher probability that those in the control group would convert than those in the treatment group.