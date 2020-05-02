# Data Cleaning Case Study

# assert statement: 
assert 2 + 2 == 5
# output: assertion error (not true)

for phrase in asap_list:
    assert phrase not in df_clean.StartDate.values
# returns no output, meaning true


# Analysis and Visualization
## after assessing and cleaning

# Question 1: percentage of postings looking for someone to start ASAP.

# asap start dates / all postings
asap_counts = df_clean.StartDate.value_counts()['ASAP']
non_empty_counts = df_clean.StartDate.count()

asap_counts / non_empty_counts
# output: about 71%

# distribution of start dates
import numpy as np 
%matplotlib inline

labels = np.full(len(df_clean.StartDate.value_counts()), "", dtype=object)
labels[0] = 'ASAP']
df_clean.StartDate.value_counts().plot(kind="pie")