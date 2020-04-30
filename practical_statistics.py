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



# Using a confidence interval to make a decision

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
%matplotlib inline
np.random.seed(42)
​
full_data = pd.read_csv('coffee_dataset.csv')
# sample_data = full_data.sample(200)

# Question: is the average height of all individuals in the coffee dataset greater than 70 inches?
H0 : mean <= 70
H1 : mean > 70

# achieve this sample from our data set
sample_df = df.sample(150)
# we can bootstrap this in the following way
bootsample = sample_df.sample(150, replace=True)

# bootstrap a number of times, and bootstrap the means for each sample
means = [] # created a vector of means, which we will enter each of our bootstrap means into
for _ in range(10000):
    bootsample = sample_df.sample(150, replace=True) # here we have our bootstrap sample
    means.append(boot_mean = bootsample.height.mean())
# now we have all of our means and can create a confidence interval
np.percentile(means, 2.5), np.percentile(means, 97.5)
# lower bound, upper bound
# might want to plot these
plt.hist(means);
plt.axvline(x=low, color='r', linewidth=2);
plt.axvline(x=high, color='r', linewidth=2);
# x=low == lower, x=high == upper


# Simulating from the null
# in hypothesis testing, we first simulate from the closest value to the alternative that is still in the null space
# could use std dev of the sample distribution to determine what the sampling distribution would look like coming from the null hypothesis
# we will simulate from a normal distribution in this case 
# code used in last example:
sample_df = df.sample(150)

means = []
for _ in range(10000):
    bootsample = sample_df.sample(150, replace=True)
    means.append(bootsample.height.mean())

np.std(means)
# result: 0.2658 == std deviation of our sampling distribution

# numpy.random.normal documentation
numpy.random.normal(loc=0.0, scale=1.0, size=None)
    # loc = the mean, in this case 70
    # scale = std dev of our sampling distribution
    # size = number of values we want to simulate (10000 for example)

null_vals = np.random.normal(70, np.std(means), 10000)
plt.hist(null_vals);

# each of the simulated draws here represents a possible mean from the null hypothesis
# where does the sample mean fall in this distribution?

sample_df.height.mean()
# output: 67.633
# this mean falls far outside the distribution from the null
# if the sample mean were to fall closer to the center value of 70, it would be a value that we would expect from the null hypothesis
# therefore we would think the null would be more likely to be true
# in this case, with our sample mean so far out in the tail, it's far enough that we think it likely did not come from our null hypothesized value
# comparing the actual sample mean to this distribution tells us the likelihood of our statistic coming from the null


# Simulating From the Null Hypothesis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
%matplotlib inline
np.random.seed(42)
​
full_data = pd.read_csv('coffee_dataset.csv')
sample_data = full_data.sample(200)

# 1. If you were interested in studying whether the average height for coffee drinkers is the same as for non-coffee drinkers, what would the null and alternative hypotheses be? Write them in the cell below, and use your answer to answer the first quiz question below.
# Since there is no directional component associated with this statement, a not equal to seems most reasonable.

# 𝐻0:𝜇𝑐𝑜𝑓𝑓−𝜇𝑛𝑜=0
# 𝐻1:𝜇𝑐𝑜𝑓𝑓−𝜇𝑛𝑜≠0
# 𝜇𝑐𝑜𝑓𝑓 and 𝜇𝑛𝑜 are the population mean values for coffee drinkers and non-coffee drinkers, respectivley.

# 2. If you were interested in studying whether the average height for coffee drinkers is less than non-coffee drinkers, what would the null and alternative be? Place them in the cell below, and use your answer to answer the second quiz question below.
# In this case, there is a question associated with a direction - that is the average height for coffee drinkers is less than non-coffee drinkers. Below is one of the ways you could write the null and alternative. Since the mean for coffee drinkers is listed first here, the alternative would suggest that this is negative.

# 𝐻0:𝜇𝑐𝑜𝑓𝑓−𝜇𝑛𝑜≥0
# 𝐻1:𝜇𝑐𝑜𝑓𝑓−𝜇𝑛𝑜<0
# 𝜇𝑐𝑜𝑓𝑓 and 𝜇𝑛𝑜 are the population mean values for coffee drinkers and non-coffee drinkers, respectivley.

# 3. For 10,000 iterations: bootstrap the sample data, calculate the mean height for coffee drinkers and non-coffee drinkers, and calculate the difference in means for each sample. You will want to have three arrays at the end of the iterations - one for each mean and one for the difference in means. Use the results of your sampling distribution, to answer the third quiz question below.

nocoff_means, coff_means, diffs = [], [], []
​
for _ in range(10000):
    bootsamp = sample_data.sample(200, replace = True)
    coff_mean = bootsamp[bootsamp['drinks_coffee'] == True]['height'].mean()
    nocoff_mean = bootsamp[bootsamp['drinks_coffee'] == False]['height'].mean()
    # append the info 
    coff_means.append(coff_mean)
    nocoff_means.append(nocoff_mean)
    diffs.append(coff_mean - nocoff_mean)   
    
np.std(nocoff_means) # the standard deviation of the sampling distribution for nocoff
0.40512631277475264
np.std(coff_means) # the standard deviation of the sampling distribution for coff
0.24073763373473001
np.std(diffs) # the standard deviation for the sampling distribution for difference in means
0.46980910743871468
plt.hist(nocoff_means, alpha = 0.5);
plt.hist(coff_means, alpha = 0.5); # They look pretty normal to me!

plt.hist(diffs, alpha = 0.5); # again normal - this is by the central limit theorem

# 4. Now, use your sampling distribution for the difference in means and the docs to simulate what you would expect if your sampling distribution were centered on zero. 
# Also, calculate the observed sample mean difference in sample_data. Use your solutions to answer the last questions in the quiz below.
# We would expect the sampling distribution to be normal by the Central Limit Theorem, and we know the standard deviation 
# of the sampling distribution of the difference in means from the previous question, so we can use this to simulate draws 
# from the sampling distribution under the null hypothesis. If there is truly no difference, then the difference between the means should be zero.

null_vals = np.random.normal(0, np.std(diffs), 10000) # Here are 10000 draws from the sampling distribution under the null
plt.hist(null_vals); #Here is the sampling distribution of the difference under the null


# Finding the p-value

# understanding p-values involves sampling distributions, and conditional probability
# p-value dependent on the alternative hypothesis as it determines what is considered more extreme

# P-Value:
    ## if H0 is true, the probability of obtaining the observed statistic or one more extreme in favor of the alternative hypothesis

# Calculating the p-value
# imagine we have the alternative hypothesis that the population mean is greater than 70
sample_mean = sample_df.height.mean()
# output: 67.633

# then we could calculate the p-value as the proportion of the simulated draws that are larger than our sample mean
(null_vals > sample_mean).mean()
# output: 1
# this means, we should stay with the mean being less than 70 as the value is small

# if our hypotheses looked like this instead, we would calculate p-value differently
H0: mean >= 70
H1: mean < 70
(null_vals < sample_mean).mean()
# because our alternative is < 70, would now look at shaded region to the left of our statistic
# p-value is 0, suggesting we reject the null hypothesis in favor of the alternative
# this suggests the pop mean is less than 70

H0: mean = 70
H1: mean != 70
null_mean = 70
(null_vals < sample_mean).mean() + (null_vals > null_mean + (null_mean - sample_mean)).mean()

# graphing:
low = sample_mean 
high = null_mean + (null_mean - sample_mean)

plt.hist(null_vals);
plt.axvline(x=low, color='r', linewidth=2)
plt.axvline(x=high, color='r', linewidth=2)

# evidence to suggest the null hypothesized value did not generate our sample statistic

# a small p-value suggests it is less likely to observe our statistic from the null
    ## therefore, choose alternative (H1)
# a large p-value suggests it is more likely to observe our statistic from the null
    # therefore, choose null (H0)

# if you are willing to make 5% errors where you choose the alternative incorrectly,
# then your p-value needs to be smaller than this threshold in order to choose the alternative
# however, if your probability of getting the data from the null is 8%, this enough of a chance, where we would stay with the null

# if our p-value is less than the type1 error rate, then we should reject the null, choose alt
# if p-value greater than than the type1 error rate, then we fail to reject the null, and choose null

# H0 is the default
# all are innocent until proven guilty, guilty or not guilty
# either reject null hypothesis, or fail to reject the null hypothesis


# Calulating errors

import numpy as np
import pandas as pd

jud_data = pd.read_csv('judicial_dataset_predictions.csv')
par_data = pd.read_csv('parachute_dataset.csv')

jud_data.head()
par_data.head()

# 1. Above, you can see the actual and predicted columns for each of the datasets. 
# Using the jud_data, find the proportion of errors for the dataset, and furthermore, the percentage of errors of each type.

jud_data[jud_data['actual'] != jud_data['predicted']].shape[0]/jud_data.shape[0] # number of errors
# output: 0.04215

jud_data.query("actual == 'innocent' and predicted == 'guilty'").count()[0]/jud_data.shape[0] # type 1 errors
# output: .0015

jud_data.query("actual == 'guilty' and predicted =='innocent'").count()[0]/jud_data.shape[0] # type 2 errors
# output: 0.0406

# if everyone was predicted to be guilty, then every actual innocent person would be a type 1 error
# type 1 = pred guilty but actually innocent

jud_data[jud_data['actual'] == 'innocent'].shape[0]/jud_data.shape[0]
# output: 0.4516

# if everyone has prediction of guilty the no one is predicted innocent
# therefore there would be no type 2 errors
# type 2 errors = pred innocent but actually guilty
# output = 0

# 2. Using the par_data, find the proportion of errors for the dataset, and furthermore, the percentage of errors of each type. Use the results to answer the questions in quiz 2 below.

result = par_data['actual'] != par_data['predicted']
print(result)
par_data[result].shape[0]/par_data.shape[0] # of errors
0.039972551037913875

par_data.query("actual == 'opens' and predicted == 'fails'").count()[0]/par_data.shape[0] # type 2
0.039800995024875621

par_data.query("actual == 'fails' and predicted == 'opens'").count()[0]/par_data.shape[0] # type 1
0.00017155601303825698


# If every parachute is predicted to fail, what is the proportion
# of type I errors made?
​
# Type I = pred open, but actual = fail
# In the above situation since we have none predicted to open,
# we have no type I errors
​0
0
# If every parachute is predicted to fail, what is
# the proportion of Type II Errors made?  
​
# This would just be the total of actual opens in the dataset, 
# as we would label these all as fails, but actually they open
​
# Type II = pred fail, but actual = open
par_data[par_data['actual'] == 'opens'].shape[0]/par_data.shape[0]
0.9917653113741637


# What is the impact of sample size?

H0 (null hypothesis): mean = 67.6
H1 (alternative hypothesis): mean != 67.6

# 2. What is the population mean? Create a sample set of data using the below code. 
# What is the sample mean? What is the standard deviation of the population? 
# What is the standard deviation of the sampling distribution of the mean of five draws? 
# Simulate the sampling distribution for the mean of five values to see the shape and plot a histogram.

sample1 = full_data.sample(5) # sample
sample1

full_data.height.mean() # population mean
# out: 67.5975

sample1.height.mean() # sample mean
# out: 67.8823

sampling_dist_mean5 = [] # the sampling distribution of the mean of five draws 
sample_of_5 = sample1.sample(5, replace = True)
sample_mean = sample_of_5.height.mean()
sampling_dist_mean5.append(sample_mean)

plt.hist(sampling_dist_mean5); # plot of sampling distribution of the mean of five draws

std_sampling_dist = np.std(sampling_dist_mean5)
std_sampling_dist # standard deviation of the sampling distribution
# out: 1.1414

# 3. Using your null and alternative hypotheses as set up in question 1 and the results of your sampling distribution in question 2, 
# simulate values of the mean values that you would expect from the null hypothesis. 
# Use these simulated values to determine a p-value to make a decision about your null and alternative hypotheses.

H0 (null hypothesis): mean = 67.6
H1 (alternative hypothesis): mean != 67.6

null_mean = 67.60 # population mean from above
null_vals = np.random.normal(null_mean, std_sampling_dist, 10000) # simulating values of the mean we would expect from the null hypothesis
# function draws random samples from a normal (Gaussian) distribution
# null_mean = mean/centre of distribution
# std_sampling_dist = scale, std. dev (spread or width) of distribution
# 10000 = number of samples drawn

plt.hist(null_vals);
plt.axvline(x=sample1.height.mean(), color = 'red'); # where our sample mean falls on null dist

# for a two sided hypothesis, we want to look at anything more extreme from the null in both directions
obs_mean = sample1.height.mean()
# out: 67.8823425205

# # probability of a statistic higher than observed
prob_more_extreme_high = (null_vals > obs_mean).mean() # from the samples pulled in the np.random.normal function
# out: 0.4071

# # probability a statistic is more extreme lower
prob_more_extreme_low = (null_mean - (obs_mean - null_mean) < null_vals).mean()
# prob_more_extreme_low = (67.60 - (67.88 - 67.60) < null_vals).mean(), 67.32 < null_vals.mean()
# probability of the statistic being lower than the mean of the null value mean pulled from random.normal
# out: 0.6021

pval = prob_more_extreme_low + prob_more_extreme_high
# out: 1.0091999999999999

# The above shows a second possible method for obtaining the p-value. 
# These are pretty different, stability of these values with such a small sample size is an issue. 
# We are essentially shading outside the lines below.

upper_bound = obs_mean # 67.8823425205
lower_bound = null_mean - (obs_mean - null_mean)
67.60 - (67.8823425205 - 67.60) = 
# lower bound output: 67.3176574795

plt.hist(null_vals)
plt.axvline(x=lower_bound, color='red'); # where our sample mean falls on null dist
plt.axvline(x=upper_bound, color = 'red'); # where our sample mean falls on null dist

print(upper_bound, lower_bound)
67.8823425205 67.3176574795

# The p-value that you obtain using the null from part 1 and the sample mean and sampling distribution standard deviation for a sample mean of size 5 from part 2 is:
1.009

null_mean = 67.60  
# this is another way to compute the standard deviation of the sampling distribution theoretically  
std_sampling_dist = full_data.height.std()/np.sqrt(5)  
num_sims = 10000

null_sims = np.random.normal(null_mean, std_sampling_dist, num_sims)  
low_ext = (null_mean - (sample1.height.mean() - null_mean))  
high_ext = sample1.height.mean()  

(null_sims > high_ext).mean() + (null_sims < low_ext).mean()


# 4. Now imagine if you received the same sample mean as you calculated from the sample in question 1 above, but that you actually retrieved it from a sample of 300. 
# What would the new standard deviation be for your sampling distribution for the mean of 300 values? 
# Additionally, what would your new p-value be for choosing between the null and alternative hypotheses you set up? 
# Simulate the sampling distribution for the mean of five values to see the shape and plot a histogram.

sample2 = full_data.sample(300)
obs_mean = sample2.height.mean() # = 67.06

sample_dist_mean300 = []
for _ in range(10000):
    sample_of_300 = sample2.sample(300, inplace=True)
    sample_mean = sample_of_300.height.mean() # = 67.09
    sampling_dist_mean300.append(sample_mean)

std_sampling_dist300 = np.std(sampling_dist_mean300) # = 0.185
null_vals = np.random.normal(null_mean, std_sampling_dist300, 10000)

upper_bound = obs_mean # 67.06
lower_bound = null_mean - (obs_mean - null_mean) # 67.60 - (67.06 - 67.60) = 67.6 - (-.54) = 68.14

plt.hist(null_vals)
plt.axvline(x=lower_bound, color = 'red');
plt.axvline(x=upper_bound, color = 'red');

# for a two sided hypothesis, we want to look at anything 
# more extreme from the null in both directions

# probability of a statistic lower than observed
prob_more_extreme_low = (null_vals < lower_bound).mean()
    
# probability a statistic is more extreme higher
prob_more_extreme_high = (upper_bound < null_vals).mean()

pval = prob_more_extreme_low + prob_more_extreme_high
pval  # With such a large sample size, our sample mean that is super
      # close will be significant at an alpha = 0.1 level.

# output: 1.9969000000000001

# Even with a very small difference between a sample mean and a hypothesized population mean, 
# the difference will end up being significant with a very large sample size.


# Multiple Tests
# In this notebook, you will work with a similar dataset to the judicial dataset you were working with before. 
# However, instead of working with decisions already being provided, you are provided with a p-value associated with each individual.

# Here is a glimpse of the data you will be working with:

import numpy as np
import pandas as pd
​
df = pd.read_csv('judicial_dataset_pvalues.csv')
df.head()
defendant_id	actual	pvalue
0	22574	innocent	0.294126
1	35637	innocent	0.417981
2	39919	innocent	0.177542
3	29610	guilty	    0.015023
4	38273	innocent	0.075371

# 1. Remember back to the null and alternative hypotheses for this example. Use that information to determine the answer for Quiz 1 and Quiz 2 below.
# A p-value is the probability of observing your data for more extreme data, if the null is true. Type I errors are when you choose the alternative when the null is true, and vice-versa for Type II. Therefore, deciding an individual is guilty when they are actually innocent is a Type I error. The alpha level is a threshold for the percent of the time you are willing to commit a Type I error.

# 2. If we consider each individual as a single hypothesis test, find the conservative Bonferroni corrected p-value we should use to maintain a 5% type I error rate.

bonf_alpha = 0.05/df.shape[0]
bonf_alpha
6.86530275985171e-06
# The new Type I Error rate after the Bonferroni correction.

# 3. What is the proportion of type I errors made if the correction isn't used? How about if it is used?

# In order to find the number of type I errors made without the correction - 
# we need to find all those that are actually innocent with p-values less than 0.05.

df.query("actual == 'innocent' and pvalue < 0.05").count()[0]/df.shape[0] # If not used
11 / 7283 = 0.0015 # The proportion of Type I Errors committed if the Bonferroni correction isn't used.

df.query("actual == 'innocent' and pvalue < @bonf_alpha").count()[0]/df.shape[0]
0.0 # The proportion of Type I Errors committed if the Bonferroni correction is used.

# 4. Think about how hypothesis tests can be used, and why this example wouldn't exactly work in terms of being able to use hypothesis testing in this way.

# This is looking at individuals, and that is more of the aim for machine learning techniques. Hypothesis testing and confidence intervals are for population parameters. 
# Therefore, they are not meant to tell us about individual cases, and we wouldn't obtain p-values for individuals in this way. We could get probabilities, 
# but that isn't the same as the probabilities associated with the relationship to sampling distributions as you have seen in these lessons.

​

# 1. Match the following characteristics of this dataset:
# total number of actions
# number of unique users
# sizes of the control and experiment groups (i.e., the number of unique users in each group)

# total number of actions
df.shape
(8188, 4)

# number of unique users
df.nunique()
timestamp    8188
id           6328
group           2
action          2
dtype: int64

# size of control group and experiment group
df.groupby('group').nunique()

# 2. How long was the experiment run for?
# Hint: the records in this dataset are ordered by timestamp in increasing order
df.timestamp.max(), df.timestamp.min()
('2017-01-18 10:24:08.629327', '2016-09-24 17:42:27.839496')

# 3. What action types are recorded in this dataset?
# (i.e., What are the unique values in the action column?)
df.action.value_counts()
view     6328
click    1860

# 5. Define the click through rate (CTR) for this experiment.
df.query('action == "click"').id.nunique() / df.query('action == "view"').id.nunique()
1860 / 6328 = 
0.2939317319848293


# Metric - Click Through Rate (CTR)

# for the control group:
control_df = df.query('group == "control"')
# we can extract all the actions from the ctrl group like this.

control_ctr = control_df.query('action == "click"').id.nunique() / control_df.query('action == "view"').id.nunique()
# now to compute CTR, we will divide number of unique users, who actually clicked the explore courses button
# by the total number of unique users who viewed the page
# control_ctr = 0.2797

# for the experiment group
experiment_df = df.query('group == "experiment"')

experiment_ctr = experiment_df.query('action == "click"').id.nunique() / experiment_df.query('action == "view"').id.nunique()
# experiment_ctr = 0.3097

obs_diff = experiment_ctr - control_ctr # observed difference
# 0.0300

# but is this result significant? maybe just due to chance?
# let's bootstrap this sample to simulate the sampling distribution for the difference in proportions

diffs = []
for _ in range(10000):
    b_samp = df.sample(df.shape[0], replace = True)
    control_df = b.samp.query('group == "control"')
    experiment_df = b.samp.query('group == "experiment"')
    control_ctr = control_df.query('action == "click"').id.nunique() / control_df.query('action == "view"').id.nunique()
    experiment_ctr = experiment_df.query('action == "click"').id.nunique() / experiment_df.query('action == "view"').id.nunique()
    diffs.append(experiment_ctr - control_ctr)

plt.hist(diffs);

# finding the p-value:
diffs = np.array(diffs)
null_vals = np.random.normal(0, diffs.std(), diffs.size)

plt.hist(null_vals)
plt.axvline(x=obs_diff, color='red');

(null_vals > obs_diff).mean() # all the nulls greater than the value of our statistic in favor of our alternative
# p-value = 0.0053
# with a p-value of approximately 0.5%, the difference in CTR for the 2 groups does appear to be significant
# with a p-value of less than 0.01, it seems unlikely that our statistic is from the null
# it looks like we can reject the null hypothesis and launch the new version of the home page


# Enrollment rate

# Get dataframe with all records from control group
control_df = df.query('group == "control"')

# Compute click through rate for control group
control_ctr = control_df.query('action == "enroll"').id.nunique() / control_df.query('action == "view"').id.nunique()

# Display click through rate
control_ctr
# 0.2364438839848676

# Get dataframe with all records from control group
experiment_df = df.query('group == "experiment"')

# Compute click through rate for experiment group
experiment_ctr = experiment_df.query('action == "enroll"').id.nunique() / experiment_df.query('action == "view"').id.nunique()

# Display click through rate
experiment_ctr
# 0.2668693009118541

# Compute the observed difference in click through rates
obs_diff = experiment_ctr - control_ctr

# Display observed difference
obs_diff

# Create a sampling distribution of the difference in proportions
# with bootstrapping
diffs = []
size = df.shape[0]
for _ in range(10000):
    b_samp = df.sample(size, replace=True)
    control_df = b_samp.query('group == "control"')
    experiment_df = b_samp.query('group == "experiment"')
    control_ctr = control_df.query('action == "enroll"').id.nunique() / control_df.query('action == "view"').id.nunique()
    experiment_ctr = experiment_df.query('action == "enroll"').id.nunique() / experiment_df.query('action == "view"').id.nunique()
    diffs.append(experiment_ctr - control_ctr)

# Convert to numpy array
diffs = np.array(diffs)

# Plot sampling distribution
plt.hist(diffs);

# Simulate distribution under the null hypothesis
null_vals = np.random.normal(0, diffs.std(), diffs.size)

# Plot the null distribution
plt.hist(null_vals);

# Plot observed statistic with the null distibution
plt.hist(null_vals);
plt.axvline(obs_diff, c='red')

# Compute p-value
(null_vals > obs_diff).mean()
# 0.018800000000000001


# Average reading duration

# the previous analyses were comparing proportions, with this, we will analyze diff in means

# we only care about duration, so let's filter:
views = df.query('action == "view"')

reading_times = views.groupby(['id', 'group'])['duration'].mean()
# let's count each unique user once, by finding their average reading duration, if they have visited the site more than once
# we will also group by group

reading_times = reading_times.reset_index()
# nice to reset index so we keep column names

# now we can find the average reading times for each group like this:
control_mean = df.query('group == "control')['duration'].mean()
experiment_mean = df.query('group == "experiment"')['duration'].mean()
control_mean, experiment_mean
# 115.4071, 130.9442

obs_diff = experiment_mean - control_mean
# 15.5371
# it looks like users in the experiment group spent 15 sec more on course overview page than those in control

# to see if this difference is significant, let's simulate the sampling distribution for the difference in mean reading durations with bootstrapping

diffs = []
for _ in range(10000):
    b_samp = df.sample(df.shape[0], replace=True)
    control_mean = b_samp.query('group == "control"')['duration'].mean()
    experiment_mean = b_samp.query('group == "experiment"')['duration'].mean()
    diffs.append(experiment_mean - control_mean)

diffs = np.array(diffs)

plt.hist(diffs);

# now to find p-value, let's simulate distribution under the null and find the probability that our observed statistic came from this distribution
null_vals = np.random.normal(0, diffs.std(), diffs.size)
# create dist centered at 0, and having the same spread as our sampling dist

plt.hist(null_vals)

plt.axvline(x=obs_diff, color='red')


# Metric - Average Classroom Time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline

np.random.seed(42)

df = pd.read_csv('classroom_actions.csv')
df.head()

# The total_days represents the total amount of time
# each student has spent in classroom.
# get the average classroom time for control group
control_mean = df.query('group == "control"')['total_days'].mean()

# # get the average classroom time for experiment group
experiment_mean = df.query('group == "experiment"')['total_days'].mean()

# # display average classroom time for each group
control_mean, experiment_mean
(73.368990384615387, 74.671593533487297)

# compute observed difference in classroom time
obs_diff = experiment_mean - control_mean

# display observed difference
obs_diff
1.3026031488719099

# create sampling distribution of difference in average classroom times
# with boostrapping
diffs = []
for _ in range(10000):
    b_samp = df.sample(df.shape[0], replace=True)
    control_mean = b_samp.query('group == "control"')['total_days'].mean()
    experiment_mean = b_samp.query('group == "experiment"')['total_days'].mean()
    diffs.append(experiment_mean - control_mean)

# convert to numpy array
diffs = np.array(diffs)

# plot sampling distribution
plt.hist(diffs);

# simulate distribution under the null hypothesis
null_vals = np.random.normal(0, diffs.std(), diffs.size)

# plot null distribution
plt.hist(null_vals)

# plot line for observed statistic
plt.axvline(x=obs_diff, color='red')

# compute p value
(null_vals > obs_diff).mean()
0.034500000000000003


# Metric - Completion Rate

# Create dataframe with all control records
control_df = df.query('group == "control"')
​
# Compute completion rate
control_ctr = control_df['completed'].mean()
​
# Display control complete rate
control_ctr
0.37199519230769229

# Create dataframe with all experiment records
experiment_df = df.query('group == "experiment"')
​
# Compute completion rate
experiment_ctr = experiment_df['completed'].mean()
​
# Display experiment complete rate
experiment_ctr
0.39353348729792148

# Compute observed difference in completion rates
obs_diff = experiment_ctr - control_ctr
​
# Display observed difference in completion rates
obs_diff
0.02153829499022919
# Create sampling distribution for difference in completion rates
# with boostrapping
diffs = []
size = df.shape[0]
for _ in range(10000):
    b_samp = df.sample(size, replace=True)
    control_df = b_samp.query('group == "control"')
    experiment_df = b_samp.query('group == "experiment"')
    control_ctr = control_df['completed'].mean()
    experiment_ctr = experiment_df['completed'].mean()
    diffs.append(experiment_ctr - control_ctr)

# convert to numpy array
diffs = np.array(diffs)
# plot distribution
plt.hist(diffs);

# create distribution under the null hypothesis
null_vals = np.random.normal(0, diffs.std(), diffs.size)
# plot null distribution
plt.hist(null_vals);
​
# plot line for observed statistic
plt.axvline(obs_diff, c='red');

# compute p value
(null_vals > obs_diff).mean()
0.084599999999999995


# Fitting a regression line in python

import pandas as pd
import numpy as np 
import statsmodels.api as sm 

df = pd.read_csv('./house_price_area_only.csv')
df.head()

# need to add a column for our intercept before moving forward

df['intercept'] = 1
# providing the OLS method the y and x variables
# OLS = ordinary least squares
lm = sm.OLS(df['price']) # = y
df[('intercept', area)]) # = list of x-variables

lm = sm.OLS(df['price'], df[['intercept', area]])
results = lm.fit()
results.summary()

# r-squared, closer it is to 1 = better fit line
# the amt of variability in the response that can be explained by the exploratory variable

# Housing Analysis
# In this notebook, you will be replicating much of what you saw in this lesson using the housing data shown below.


import numpy as np
import pandas as pd
import statsmodels.api as sm;
​
df = pd.read_csv('./house_price_area_only.csv')
df.head()

# 1. fit a linear model to predict price based on area. Obtain a summary of the results. Don't forget to add an intercept.

df['intercept'] = 1
​
lm = sm.OLS(df['price'], df[['intercept', 'area']])
results = lm.fit()
results.summary()

# Homes vs. Crime case study

import numpy as np
import pandas as pd
import statsmodels.api as sms;
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
%matplotlib inline

boston_data = load_boston()
df = pd.DataFrame()
df['MedianHomePrice'] = boston_data.target
df2 = pd.DataFrame(boston_data.data)
df['CrimePerCapita'] = df2.iloc[:,0];
df.head()

# The Boston housing data is a built in dataset in the sklearn library of python. You will be using two of the variables from this dataset, which are stored in df. The median home price in thousands of dollars and the crime per capita in the area of the home are shown above.
# 1. Use this dataframe to fit a linear model to predict the home price based on the crime rate. Use your output to answer the first quiz below. Don't forget an intercept.

df['intercept'] = 1

lm = smsOLD(df['MedianHomePrice'], df[['intercept', 'CrimePerCapita']])
results = lm.fit()
results.summary()

# 2.Plot the relationship between the crime rate and median home price below.

plt.scatter(df['CrimePerCapita'], df['MedianHomePrice']);
plt.xlabel('Crime/Capita');
plt.ylabel('Median Home Price');
plt.title('Median Home Price vs. CrimePerCapita');

## To show the line that was fit I used the following code from 
## https://plot.ly/matplotlib/linear-fits/
## It isn't the greatest fit... but it isn't awful either


import plotly.plotly as py
import plotly.graph_objs as go

# MatPlotlib
import matplotlib.pyplot as plt
from matplotlib import pylab

# Scientific libraries
from numpy import arange,array,ones
from scipy import stats

xi = arange(0,100)
A = array([xi, ones(100)])

# (Almost) linear sequence
y = df['MedianHomePrice']
x = df['CrimePerCapita']

# Generated linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
line = slope*xi+intercept 

plt.plot(x,y,'o', xi, line);
plt.xlabel('Crime/Capita')
plt.ylabel('Median Home Price')
pylab.title('Median Home Price vs. CrimePerCapita')


# Fitting a multiple linear regression model

import numpy as np
import pandas as pd
import statsmodels.api as sms;

df = pd.read_csv('./data/house_prices.csv')
df.head()

# we will need to add all of our variables into our list for the x portion

df['intercept'] = 1
lm = sm.OLS(df['price'], df[['intercept', 'bathrooms', 'bedrooms', 'area']])
# interested in predicting the price (left side)
# on the right side, add all variables that are quantitative. adding categorical variables will break
# lm = linear model
# then we will fit the model (to what?)
results = lm.fit()
results.summary()


# How does multiple linear regression (MLR) work?
# using the same data from the above example

df['intercept'] = 1
lm = sm.OLS(df['price'], df[['intercept', 'bathrooms', 'bedrooms', 'area']])
results = lm.fit()
results.summary()

# create x matrix, which will only contain the portions of the output we found with the coef, std err, etc. for the variables
x = df[['intercept', 'bathrooms', 'bedrooms', 'area']]
y = df['price']
# y is just our response, so only the price
# we will take x transpose, x (dot product of these 2 things), invert that, # then dot product with the 
# transpose, then dot product with the response. That should give us our coefficient estimates.
np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),y)
# xtranspose multiplied by X, which is the dot product
# then we want the dot product of x transpose again
# then we want a final dot product with the response
# need inverse of this np.dot(X.transpose(),X)
# then we will multiply the transpose of that times the response

# now we can see that our values match the summary table


# Slope interpretation
# for every one unit increase in x, the expected y increases by the slop, holding all else constant


# Dummy variables for categorical variables

# get a 1 if value exists in column, 0 if not
# create columns for each neighborhood value
# should do for each categorical values

# pandas - get dummies
pd.get_dummies(df['neighborhood'])

# let's store this output:
df[['A', 'B', 'C']] = pd.get_dummies(df['neighborhood'])

# this was for neighborhood, now let's try for style
# your categorical variables will always come back in alphabetical order

df[['lodge', 'ranch', 'victorian']] = pd.get_dummies(['style'])
# we need to drop one column, column we drop called baseline category

# let's drop the victorian column and use the other 2
df['intercept'] = 1
lm = sm.OLS(df['price'], df[['intercept', 'lodge', 'ranch']])
results = lm.fit()
results.summary()

# intercept means that if our home is a victorian home, we predict its price to be 1,046e+06 or $1,046,000
# lodge is meant to be $741,100 less than a victorian, ranch is predicted $471,000 less than victorian
# each is comparing with the baseline category (intercept, which in this case is victorian)


# Dummy variables quiz

import numpy as np
import pandas as pd
import statsmodels.api as sm;
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('./house_prices.csv')
df.head()

# 1. Use the pd.get_dummies documentation to assist you with obtaining dummy variables for the neighborhood column. Then use join to add the dummy variables to your dataframe, df, and store the joined results in df_new.
# Fit a linear model using all three levels of neighborhood neighborhood to predict the price. Don't forget an intercept.

neighborhood_dummies = pd.get_dummies(df['neighborhood'])
df_new = df.join(neighborhood_dummies)
df_new.head()

df_new['intercept'] = 1
lm = sm.OLS(df_new['price'], df.new[['intercept', 'A', 'B', 'C']])
results = lm.fit()
results.summary()

# 2. Now, fit an appropriate linear model for using neighborhood to predict the price of a home. 
# Use neighborhood A as your baseline. 

lm2 = sm.OLS(df_new['price'], df_new[['intercept', 'B', 'C']])
results2 = lm2.fit()
results2.summary()

# 3. Run the two cells below to look at the home prices for the A and C neighborhoods. Add neighborhood B. 
# This creates a glimpse into the differences that you found in the previous linear model.

plt.hist(df_new.query("C == 1")['price'], alpha = 0.3, label = 'C');
plt.hist(df_new.query("A == 1")['price'], alpha = 0.3, label = 'A');
plt.legend()

# 4. Now, add dummy variables for the style of house, as well as neighborhood. Use ranch as the baseline for the style. 
# Additionally, add bathrooms and bedrooms to your linear model. Don't forget an intercept. Home prices are measured in dollars, and this dataset is not real.

type_dummies = pd.get_dummies(df['style'])
df_new = df_new.join(type_dummies)
df_new.head()

lm3 = sm.OLS(df_new['price'], df_new[['intercept', 'B', 'C', 'lodge', 'victorian', 'bedrooms', 'bathrooms']])
results3 = lm3.fit()
results3.summary()


# Assumption of Linear Regression Models
# our x-variables should be correlated with the response but not each other

# in our last example, we can imagine that the area, # of bedrooms and # bathrooms would likely be correlated
# can take a quick look at relationships using seaborn
# in addition to the packages we read in before:
import seaborn as seaborn
import patsy import dmatrices 
from statsmodels.stats.outliers_influence import variance_inflation_factor

sb.pairplot(df[['area', 'bedrooms', 'bathrooms']]) ;  # want to look at each of the 3 x-variables
# allows us to view relationship between each of our variables
# we see strong positive relationships in the graphs

