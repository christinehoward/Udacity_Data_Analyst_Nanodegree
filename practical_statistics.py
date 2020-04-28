# Case Study in Python
# Use the Jupyter notebook to analyze admission_data.csv to find the following values and for the quizzes below. 
# Indexing, query, and groupby may come in handy!
## Load and view first few lines of dataset
import pandas as pd
import numpy as np
admits = pd.read_csv('admission_data.csv')
admits.head()
Proportion and admission rate for each gender
print (len(admits[admits['gender']=='female']))
print (admits.shape[0])
257
500
# Proportion of students that are female
(len(admits[admits['gender']=='female']))/admits.shape[0]
0.514
# Proportion of students that are male
(len(admits[admits['gender']=='male']))/admits.shape[0]
0.486
# Admission rate for females
len(admits[(admits['gender']=='female') & (admits['admitted'])])/(len(admits[admits['gender']=='female']))
0.28793774319066145
# Admission rate for males
len(admits[(admits['gender']=='male') & (admits['admitted'])])/(len(admits[admits['gender']=='male']))
0.48559670781893005
Proportion and admission rate for physics majors of each gender
# What proportion of female students are majoring in physics?
fem_phys_rate = admits.query("gender == 'female' & major == 'Physics'").count()/ \
    (admits.query("gender == 'female'").count())
print (fem_phys_rate)
student_id    0.120623
gender        0.120623
major         0.120623
admitted      0.120623
dtype: float64
# What proportion of male students are majoring in physics?
fem_phys_rate = admits.query("gender == 'male' & major == 'Physics'").count()/ \
    (admits.query("gender == 'male'").count())
print (fem_phys_rate)
student_id    0.925926
gender        0.925926
major         0.925926
admitted      0.925926
dtype: float64
# Admission rate for female physics majors
len(admits[(admits["gender"]=='female') & (admits["major"] == 'Physics') & admits["admitted"]]) / len(admits[(admits["gender"]=='female') & (admits["major"] == 'Physics')])
0.7419354838709677
# Admission rate for male physics majors
len(admits[(admits["gender"]=='male') & (admits["major"] == 'Physics') & admits["admitted"]]) / len(admits[(admits["gender"]=='male') & (admits["major"] == 'Physics')])
0.5155555555555555
Proportion and admission rate for chemistry majors of each gender
# What proportion of female students are majoring in chemistry?
len(admits[(admits['gender']=='female') & (admits['major'] == 'Chemistry')]) / len(admits[admits['gender']=='female'])
0.8793774319066148
# What proportion of male students are majoring in chemistry?
len(admits[(admits['gender']=='male') & (admits['major'] == 'Chemistry')]) / len(admits[admits['gender']=='male'])
0.07407407407407407
# Admission rate for female chemistry majors
len(admits[(admits['gender']=='female') & (admits['major'] == 'Chemistry') & admits['admitted']]) / len(admits[(admits['gender']=='female') & (admits['major'] == 'Chemistry')])
0.22566371681415928
# Admission rate for male chemistry majors
len(admits[(admits['gender']=='male') & (admits['major'] == 'Chemistry') & admits['admitted']]) / len(admits[(admits['gender']=='male') & (admits['major'] == 'Chemistry')])
0.1111111111111111
Admission rate for each major
# Admission rate for physics majors
len(admits[(admits['major'] == 'Physics') & admits['admitted']]) / len(admits[(admits['major'] == 'Physics')])
0.54296875
# Admission rate for chemistry majors
len(admits[(admits['major'] == 'Chemistry') & admits['admitted']]) / len(admits[(admits['major'] == 'Chemistry')])
0.21721311475409835


# Probability Quiz
# In this quiz, you will simulate coin flips and die rolls to compute proportions.
# When simulating coin flips, use 0 to represent heads and 1 to represent tails. 
# When simulating die rolls, use the correct integers to match the numbers on the sides of a standard 6 sided die.

1. Two fair coin flips produce exactly two heads
# simulate 1 million tests of two fair coin flips
tests = np.random.randint(2, size=(int(1e6), 2))
​
# sums of all tests
test_sums = tests.sum(axis=1)
​
# proportion of tests that produced exactly two heads
(test_sums == 0).mean()
0.24981600000000001
2. Three fair coin flips produce exactly one head
(test_sums == 2).mean()
# simulate 1 million tests of three fair coin flips
tests = np.random.randint(2, size=(int(1e6), 3))
​
# sums of all tests
test_sums = tests.sum(axis=1)
​
# proportion of tests that produced exactly one head
(test_sums == 2).mean()
0.37496200000000002
3. Three biased coin flips with P(H) = 0.6 produce exactly one head
# simulate 1 million tests of three biased coin flips
# hint: use np.random.choice()
tests = np.random.randint(2, size=(int(1e6), 3))

# sums of all tests
test_sums = tests.sum(axis=1)

# proportion of tests that produced exactly one head
(test_sums == 2).mean()
# simulate 1 million tests of three bias coin flips
# hint: use np.random.choice()
tests = np.random.choice([0, 1], size=(int(1e6), 3), p=[0.6, 0.4])
​
# sums of all tests
test_sums = tests.sum(axis=1)
​
# proportion of tests that produced exactly one head
(test_sums == 2).mean()
0.28752800000000001
### 4. A die rolls an even number
# simulate 1 million tests of one die roll
tests = 

# proportion of tests that produced an even number

# simulate 1 million tests of one die roll
tests = np.random.choice(np.arange(1, 7), size=int(1e6))
​
# proportion of tests that produced an even number
(tests % 2 == 0).mean()
0.49986799999999998
5. Two dice roll a double
# simulate the first million die rolls
first = 

# simulate the second million die rolls
second = 

# proportion of tests where the 1st and 2nd die rolled the same number

# simulate the first million die rolls
first = np.random.choice(np.arange(6), size=int(1e6))
​
# simulate the second million die rolls
second = np.random.choice(np.arange(6), size=int(1e6))
​
# proportion of tests where the 1st and 2nd die rolled the same number
(first == second).mean()
0.16661999999999999


# Simulating many coin flips
import numpy as np 
np.random.binomial(n=10, p=0.5)
# output: 4 (number of heads)

# run 20 times:
np.random.binomial(n=10, p=0.5, size=20)
# output: number of heads in each result of 10 coin flips

np.random.binomial(n=10, p=0.5, size=20).mean()
# we would expect the number to be close to 5, and it is at 4.65

import matplotlib.pyplot as plt 
%matplotlib inline
plt.hist(np.random.binomial(n=10, p=0.5, size=10000))
# plotting outcomes in last test, but with size=10000 instead of 20.
# distribution normal and centered around 5 heads

# Simulating Many Coin Flips
import numpy as np
# number of heads from 10 fair coin flips
np.random.binomial(10, 0.5)
5
# results from 20 tests with 10 coin flips
np.random.binomial(10, 0.5, 20)
array([6, 3, 6, 6, 7, 6, 4, 4, 6, 3, 7, 7, 4, 5, 7, 4, 6, 6, 6, 5])
# mean number of heads from the 20 tests
np.random.binomial(10, 0.5, 20).mean()
5.2999999999999998

# reflects the fairness of the coin more closely as # tests increases
np.random.binomial(10, 0.5, 1000000).mean()
4.9999330000000004
import matplotlib.pyplot as plt
% matplotlib inline
plt.hist(np.random.binomial(10, 0.5, 1000000));

# gets more narrow as number of flips increase per test
plt.hist(np.random.binomial(100, 0.5, 1000000));

​
# Binomial Distributions Quiz
# In this quiz, you will simulate coin flips using np.random.binomial to compute proportions for the following outcomes.

1. A fair coin flip produces heads
(tests == 1).mean()
# simulate 1 million tests of one fair coin flip
# remember, the output of these tests are the # successes, or # heads
tests = np.random.binomial(1, 0.5, int(1e6))
​
# # proportion of tests that produced heads
(tests == 1).mean()
0.50045499999999998
2. Five fair coin flips produce exactly one head
# simulate 1 million tests of five fair coin flips
tests =

# proportion of tests that produced 1 head

# simulate 1 million tests of five fair coin flips
tests = np.random.binomial(5, 0.5, int(1e6))
​
# proportion of tests that produced 1 head
(tests == 1).mean()
0.15559000000000001
3. Ten fair coin flips produce exactly four heads
# simulate 1 million tests of ten fair coin flips
tests = np.random.binomial(10, 0.5, int(1e6))
​
# proportion of tests that produced 4 heads
(tests == 4).mean()
0.205152
4. Five biased coin flips with P(H) = 0.8 produce exactly five heads
# simulate 1 million tests of five biased coin flips
tests = np.random.binomial(5, 0.8, int(1e6))
​
# proportion of tests that produced 5 heads
(tests == 5).mean()
0.32783400000000001
5. Ten biased coin flips with P(H) = 0.15 produce at least 3 heads
>=
# simulate 1 million tests of ten biased coin flips
tests = np.random.binomial(10, 0.15, int(1e6))
​
# proportion of tests that produced at least 3 heads
(tests >= 3).mean()
0.18040100000000001


# Cancer Test Results
import pandas as pd
​
df = pd.read_csv('cancer_test_data.csv')
df.head()
patient_id	test_result	has_cancer
0	79452	Negative	False
1	81667	Positive	True
2	76297	Negative	False
3	36593	Negative	False
4	53717	Negative	False
df.shape
(2914, 3)
# number of patients with cancer
df.has_cancer.sum()
306
# number of patients without cancer
(df.has_cancer == False).sum()
2608
# proportion of patients with cancer
df.has_cancer.mean()
0.10501029512697323
# proportion of patients without cancer
1 - df.has_cancer.mean()
0.89498970487302676
# proportion of patients with cancer who test positive
(df.query('has_cancer')['test_result'] == 'Positive').mean()
0.90522875816993464
# proportion of patients with cancer who test negative
(df.query('has_cancer')['test_result'] == 'Negative').mean()
0.094771241830065356
# proportion of patients without cancer who test positive
(df.query('has_cancer == False')['test_result'] == 'Positive').mean()
0.2036042944785276
# proportion of patients without cancer who test negative
(df.query('has_cancer == False')['test_result'] == 'Negative').mean()
0.79639570552147243


# Conditional Probability & Bayes Rule Quiz
# load dataset
import pandas as pd
​
df = pd.read_csv('cancer_test_data.csv')
df.head()
patient_id	test_result	has_cancer
0	79452	Negative	False
1	81667	Positive	True
2	76297	Negative	False
3	36593	Negative	False
4	53717	Negative	False
# What proportion of patients who tested positive has cancer?
df.query('test_result == "Positive"')['has_cancer'].mean()
0.34282178217821785
# What proportion of patients who tested positive doesn't have cancer?
1 - df.query('test_result == "Positive"')['has_cancer'].mean()
0.65717821782178221
# What proportion of patients who tested negative has cancer?
df.query('test_result == "Negative"')['has_cancer'].mean()
0.013770180436847104
# What proportion of patients who tested negative doesn't have cancer?
1 - df.query('test_result == "Negative"')['has_cancer'].mean()
0.98622981956315292


# Sampling Distributions Introduction
# In order to gain a bit more comfort with this idea of sampling distributions, let's do some practice in python.
# Below is an array that represents the students we saw in the previous videos, where 1 represents the students that drink coffee, and 0 represents the students that do not drink coffee.

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
​
students = np.array([1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0])
# 1. Find the proportion of students who drink coffee in the above array. Store this value in a variable p.

students.mean()
0.7142857142857143
p = students.mean()
p
0.7142857142857143
# 2. Use numpy's random.choice to simulate 5 draws from the students array. What is proportion of your sample drink coffee?

sample1 = np.random.choice(students, 5, replace=True)
sample1.mean()
0.59999999999999998
# 3. Repeat the above to obtain 10,000 additional proportions, where each sample was of size 5. Store these in a variable called sample_props.

sample_props = []
for _ in range(10000):
    sample = np.random.choice(students, 5, replace=True)
    sample_props.append(sample.mean())
    
# 4. What is the mean proportion of all 10,000 of these proportions? This is often called the mean of the sampling distribution.

sample_props = np.array(sample_props)
sample_props.mean()
0.71399999999999997
# 5. What are the variance and standard deviation for the original 21 data values?

print('The standard deviation for the original data is {}'.format(students.std()))
print('The variance for the original data is {}'.format(students.var()))
The standard deviation for the original data is 0.45175395145262565
The variance for the original data is 0.20408163265306126
# 6. What are the variance and standard deviation for the 10,000 proportions you created?

print('The standard deviation of the sampling distribution of the mean of 5 draws is {}'.format(sample_props.std()))
print('The variance for the sampling distribution of the mean of 5 draws is {}'.format(sample_props.var()))
The standard deviation of the sampling distribution of the mean of 5 draws is 0.2043624231604235
The variance for the sampling distribution of the mean of 5 draws is 0.041763999999999996
# 7. Compute p(1-p), which of your answers does this most closely match?

p*(1-p) # The variance of the original data
0.20408163265306123
# 8. Compute p(1-p)/n, which of your answers does this most closely match?

p*(1-p)/5 # The variance of the sample mean of size 5
0.040816326530612249
# 9. Notice that your answer to 8. is commonly called the variance of the sampling distribution. If you were to change your first sample to be 20, what would this do for the variance of the sampling distribution? Simulate and calculate the new answers in 6. and 8. to check that the consistency you found before still holds.

##Simulate your 20 draws
sample_props_20 = []
for _ in range(10000):
    sample = np.random.choice(students, 20, replace=True)
    sample_props_20.append(sample.mean())
##Compare your variance values as computed in 6 and 8, 
##but with your sample of 20 values
​
​
print(p*(1-p)/20) # The theoretical variance
print(np.array(sample_props_20).var()) # The simulated variance
0.0102040816327
0.010300994375
# 10. Finally, plot a histgram of the 10,000 draws from both the proportions with a sample size of 5 and the proportions with a sample size of 20. Each of these distributions is a sampling distribution. One is for the proportions of sample size 5 and the other a sampling distribution for proportions with sample size 20.

plt.hist(sample_props, alpha=.5);
plt.hist(np.array(sample_props_20), alpha=.5);

# Notice the 20 is much more normally distributed than the 5


# Bootstrap Sampling
# Below is an array of the possible values you can obtain from a die. Let's consider different methods of sampling from these values.

import numpy as np
np.random.seed(42)
​
die_vals = np.array([1,2,3,4,5,6])
Take a random sample of 20 values from die_vals using the code below, then answer the question in the first quiz below.
np.random.choice(die_vals, size=20)
array([4, 5, 3, 5, 5, 2, 3, 3, 3, 5, 4, 3, 6, 5, 2, 4, 6, 6, 2, 4])


# Building confidence interval

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

coffee_full = pd.read_csv...
coffee_red = coffee_full.sample(200)

coffee_red.shape
coffee_full.shape 
coffee_red.drinks_coffee.mean()
# output: 0.56999
coffee_red[coffee_red['drinks_coffee'] == True]['height'].mean()
# output: 68.5202

bootsample = coffee_red.sample(200, replace=True)
bootsample[bootsample['drinks_coffee'] == True]['height'].mean()

boot_means = []
for _ in range(10000):
   bootsample = coffee_red.sample(200, replace=True)
   boot_means.append(bootsample[bootsample['drinks_coffee'] == True]['height'].mean()) 

plt.hist(boot_means);

np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
# output: 68.0595, 68.9652
# the 2.5 is the amount of the bell curve we will cut off from each side

# confidence interval interpretation:
## we are 95% confident the mean height of all coffee drinkers is between 68.06 and 68.97 inches tall

# the mean of the population was 68.4002
coffee_full[coffee_full('drinks_coffee') == True]['height'].mean()


# Confidence Intervals - Part I
# First let's read in the necessary libraries and the dataset. You also have the full and reduced versions of the data available. The reduced version is an example of you would actually get in practice, as it is the sample. While the full data is an example of everyone in your population.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
​
np.random.seed(42)
​
coffee_full = pd.read_csv('../data/coffee_dataset.csv')
coffee_red = coffee_full.sample(200) #this is the only data you might actually get in the real world.

# 1. What is the proportion of coffee drinkers in the sample? What is the proportion of individuals that don't drink coffee?
coffee_red['drinks_coffee'].mean() # Drink Coffee
0.59499999999999997
1 - coffee_red['drinks_coffee'].mean() # Don't Drink Coffee
0.40500000000000003

# Confidence Intervals - Part I
# First let's read in the necessary libraries and the dataset. You also have the full and reduced versions of the data available. The reduced version is an example of you would actually get in practice, as it is the sample. While the full data is an example of everyone in your population.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
​
np.random.seed(42)
​
coffee_full = pd.read_csv('../data/coffee_dataset.csv')
coffee_red = coffee_full.sample(200) #this is the only data you might actually get in the real world.

# 1. What is the proportion of coffee drinkers in the sample? 
# What is the proportion of individuals that don't drink coffee?
coffee_red['drinks_coffee'].mean() # Drink Coffee
0.59499999999999997
1 - coffee_red['drinks_coffee'].mean() # Don't Drink Coffee
0.40500000000000003

# 2. Of the individuals who do not drink coffee, what is the average height?
coffee_red[coffee_red['drinks_coffee'] == False]['height'].mean()
66.78492279927877

# 3. Simulate 200 "new" individuals from your original sample of 200. What are the proportion of coffee drinkers in your bootstrap sample? How about individuals that don't drink coffee?
bootsamp = coffee_red.sample(200, replace = True)
bootsamp['drinks_coffee'].mean() # Drink Coffee and 1 minus gives the don't drink
0.60499999999999998

# 4. Now simulate your bootstrap sample 10,000 times and take the mean height of the non-coffee drinkers in each sample. 
# Plot the distribution, and pull the values necessary for a 95% confidence interval. 
# What do you notice about the sampling distribution of the mean in this example?
boot_means = []
for _ in range(10000):
    bootsamp = coffee_red.sample(200, replace = True)
    boot_mean = bootsamp[bootsamp['drinks_coffee'] == False]['height'].mean()
    boot_means.append(boot_mean)
    
plt.hist(boot_means); # Looks pretty normal

np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
(65.992913281575198, 67.584027382815734)
​
# 5. Did your interval capture the actual average height of coffee drinkers in the population? 
# Look at the average in the population and the two bounds provided by your 95% confidence interval, and then answer the final quiz question below.
coffee_full[coffee_full['drinks_coffee'] == False]['height'].mean() 
66.44340776214705
# Captured by our interval, but not the exact same as the sample mean
coffee_red[coffee_red['drinks_coffee'] == False]['height'].mean()
66.78492279927877


# Screencast: difference in means

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
​
coffee_full = pd.read_csv('coffee-dataset.csv')
coffee_red = coffee_full.sample(200)

# Question: what's the difference in mean height for coffee versus non-coffee drinkers
# in order to build confidence interval for the difference in the average heights for these 2 groups
# we can do something similar to what we have already done, but for each iteration of taking the mean for each group,
# we are also going to take the difference

bootsample = coffee_red.sample(200, replace=True)
mean_coff = bootsample[bootsample['drinks_coffee'] == True]['height'].mean()
mean_nocoff = bootsample[bootsample['drinks_coffee'] == False]['height'].mean()
mean_coff - mean_nocoff
# output: 1.96

# now we can iterate this process some large number of times and use the resulting differences to build a confidence interval for the difference in the means
diff = []
for _ in range(10000):
    bootsample = coffee_red.sample(200, replace=True)
    mean_coff = bootsample[bootsample['drinks_coffee'] == True]['height'].mean()
    mean_nocoff = bootsample[bootsample['drinks_coffee'] == False]['height'].mean()
    mean_coff - mean_nocoff
# set up difference list and we will append differences into it
# the difference in means for a single iteration of a bootstrap sample is stored in diff

# could again plot the difference in means for the 2 groups
plt.hist(diff)

# could cut off the top and bottom 2.5% to achieve 95% confidence interval
# where we believe the difference in the 2 means to exist

np.percentile(diff, 2.5), np.percentile(diff, 97.5)
# output: 0.5854, 2.3687

# because 0 is not included in the interval, this would suggest a difference in the population means

# conclusion: since a confidence interval for mean_coff-mean_nocoff is (0.59, 2.37), 
# we have evidence of the mean height for coffee drinkers is larger than non-coffee drinkers


# Confidence Interval - Difference In Means
# Here you will look through the example for the last video, but you will also go a couple of steps further into what might actually be going on with this data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
%matplotlib inline
np.random.seed(42)
​
full_data = pd.read_csv('../data/coffee_dataset.csv')
sample_data = full_data.sample(200)

# 1. For 10,000 iterations, bootstrap sample your sample data, compute the difference in the average heights for coffee and non-coffee drinkers. Build a 99% confidence interval using your sampling distribution. Use your interval to start answering the first quiz question below.
diffs = []
for _ in range(10000):
    bootsamp = sample_data.sample(200, replace = True)
    coff_mean = bootsamp[bootsamp['drinks_coffee'] == True]['height'].mean()
    nocoff_mean = bootsamp[bootsamp['drinks_coffee'] == False]['height'].mean()
    diffs.append(coff_mean - nocoff_mean)
    
np.percentile(diffs, 0.5), np.percentile(diffs, 99.5) 
# statistical evidence coffee drinkers are on average taller
(0.10258900080921124, 2.5388333707966568)


# 2. For 10,000 iterations, bootstrap sample your sample data, compute the difference in the average heights for those older than 21 and those younger than 21. Build a 99% confidence interval using your sampling distribution. Use your interval to finish answering the first quiz question below.

diffs_age = []
for _ in range(10000):
    bootsamp = sample_data.sample(200, replace = True)
    under21_mean = bootsamp[bootsamp['age'] == '<21']['height'].mean()
    over21_mean = bootsamp[bootsamp['age'] != '<21']['height'].mean()
    diffs_age.append(over21_mean - under21_mean)
    
np.percentile(diffs_age, 0.5), np.percentile(diffs_age, 99.5)
# statistical evidence that over21 are on average taller
(3.3652749452554795, 5.0932450670661495)


# 3. For 10,000 iterations bootstrap your sample data, compute the difference in the average height for coffee drinkers and the average height non-coffee drinkers for individuals under 21 years old. Using your sampling distribution, build a 95% confidence interval. Use your interval to start answering question 2 below.

diffs_coff_under21 = []
for _ in range(10000):
    bootsamp = sample_data.sample(200, replace = True)
    under21_coff_mean = bootsamp.query("age == '<21' and drinks_coffee == True")['height'].mean()
    under21_nocoff_mean = bootsamp.query("age == '<21' and drinks_coffee == False")['height'].mean()
    diffs_coff_under21.append(under21_nocoff_mean - under21_coff_mean)
    
np.percentile(diffs_coff_under21, 2.5), np.percentile(diffs_coff_under21, 97.5)
# For the under21 group, we have evidence that the non-coffee drinkers are on average taller
(1.0593651244624267, 2.5931557940679042)

# 4. For 10,000 iterations bootstrap your sample data, compute the difference in the average height for coffee drinkers and the average height non-coffee drinkers for individuals under 21 years old. Using your sampling distribution, build a 95% confidence interval. Use your interval to finish answering the second quiz question below. As well as the following questions.
diffs_coff_over21 = []
for _ in range(10000):
    bootsamp = sample_data.sample(200, replace = True)
    over21_coff_mean = bootsamp.query("age != '<21' and drinks_coffee == True")['height'].mean()
    over21_nocoff_mean = bootsamp.query("age != '<21' and drinks_coffee == False")['height'].mean()
    diffs_coff_over21.append(over21_nocoff_mean - over21_coff_mean)
    
np.percentile(diffs_coff_over21, 2.5), np.percentile(diffs_coff_over21, 97.5)
# For the over21 group, we have evidence that on average the non-coffee drinkers are taller
(1.8278953970883667, 4.4026329654774337)
# Within the under 21 and over 21 groups, we saw that on average non-coffee drinkers were taller. But, when combined, we saw that on average coffee drinkers were on average taller. This is again Simpson's paradox, and essentially there are more adults in the dataset who were coffee drinkers. So these individuals made it seem like coffee drinkers were on average taller - which is a misleading result.
# A larger idea for this is the idea of confounding variables altogether. You will learn even more about these in the regression section of the course.

​
# Traditional confidence intervals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
%matplotlib inline
np.random.seed(42)
​
full_data = pd.read_csv('coffee_dataset.csv')
sample_data = full_data.sample(200)

# looking at difference in means with bootstrap approach from earlier
diff = []
for _ in range(10000):
    bootsample = coffee_red.sample(200, replace = True)
    mean_coff = bootsample[bootsample['drinks_coffee'] -- True]['height'].mean()
    mean_nocoff = bootsample[bootsample['drinks_coffee'] == False]['height'].mean()
    diff.append(mean_coff - mean_nocoff)
np.percentile(diff, 2.5), np.percentile(diff, 97.5)

# other option:
import statsmodels.stats.api as sms 
X1 = coffee_red[coffee_red['drinks_coffee'] == True]['height']
X2 = coffee_red[coffee_red['drinks_coffee'] == False]['height']

cm = sms.CompareMeans(sms.DescStatsW(X1), sms.DescrStatsW(X2))
cm.tconfint_diff(usevar='unequal')


