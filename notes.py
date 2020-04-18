# create color columns
color_red = np.repeat('red', red_df.shape[0])

# red_df.shape = ['red', 'green', 'blue']
# red_df.shape[0] = ???? 'red'
# Why? Because it gets the first value in the array.
# Arrays index start at ??? 0
# Strings can also be indexed.
# love = 'Christine'
# love[3] = ????? 'C' ? No. 'i' ? Yes
# Strings ('Christine') and Arrays ( ['red', 'green']) are Iterable
# That means that they can be indexed
# 'Christine'[0] will return the 0 character => 'C'
# 'Christine'.tail(1)  will return => ???? 
# get first index of array, in this case red. when red, green, blue, and red_df.shape[1], then green.

x = np.array(
    [
        [1,2],
        [3,4]
    ]
)
np.repeat(x, 2)
array([1, 1, 2, 2, 3, 3, 4, 4])

color_white = np.repeat('white', white_df.shape[0])

amoMinhaEsposa = True,
amoMinhaGatinha = True
minhaEsposaEh = 'Christine Lindona'


Dataframe.append(self, other=True, ignore_index, verify_integrity=False, sort=False) {
    # self? Dataframe self in the documentation, will be never be explained.
    print(other)
    print(ignore_index
    print(verify_integrity)
    print(sort)
    return other
}

result = Dataframe.append()
#None
#None
#False
#False
# What's the value of result ??? 

print(result)
# True

result2 = Dataframe.append(False)
#False

result3 = Dataframe.append('HELLO HELLO')
#
#False
#True
#False
#False

#Whats the value information, that the result2 is holding,  that will be returned from Dataframe.append(False)????
#



np.repeat(3, 4)
array([3, 3, 3, 3])
x = np.array(
    [
        [1,2],
        [3,4]
    ]
)
np.repeat(x, 2)
array([1, 1, 2, 2, 3, 3, 4, 4])
np.repeat(x, 3, axis=1)
array([
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4]
    ])
np.repeat(x, [2, 2], axis=0)
array([
    [1, 2],
    [1, 2]
    [3, 4],
    [3, 4]
])



NaN example

numero = '1'

numero.toInt()
# 1

notreallyanumber = 'Christine'

notreallyanumber.toInt()
# NaN