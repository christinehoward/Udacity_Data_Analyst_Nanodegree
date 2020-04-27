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