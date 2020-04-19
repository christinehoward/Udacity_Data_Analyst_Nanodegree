
print()
    ## used to display result

# ARITHMETIC OPERATORS
 ## +, -, *, /   
 ## need to remember PEMDAS when calculating
    # exponentiation
     ## print(3**2) = 9 (3 squared)
    # modulo - %
     ## returns remainder, after you've divided the 1st number by the 2nd
     ## print(9 % 2) = 9/2 with remainder of 1. result = 1
    # integer division
     ## print(7 // 2) = 3


# Quiz: Average Electricity Bill
## My electricity bills for the last three months have been $23, $32 and $64. What is the average monthly electricity bill over the three month period? Write an expression to calculate the mean, and use print() to view the result.

# Write an expression that calculates the average of 23, 32 and 64
# Place the expression in this print statement
print((23+32+64)/3)

# Quiz: Calculate
## In this quiz you're going to do some calculations for a tiler. Two parts of a floor need tiling. One part is 9 tiles wide by 7 tiles long, the other is 5 tiles wide by 7 tiles long. Tiles come in packages of 6.
    ##How many tiles are needed?
    ## You buy 17 packages of tiles containing 6 tiles each. How many tiles will be left over?

# Fill this in with an expression that calculates how many tiles are needed.
print((9*7)+(5*7))

# Fill this in with an expression that calculates how many tiles will be left over.
print((17*6) % 98)


#VARIABLES

# example
mv_population = 74728
mv_population += 4000 - 600
print(mv_population)
# result: 78128


# Quiz: Assign and Modify Variables
## Now it's your turn to work with variables. The comments in this quiz (the lines that begin with #) have instructions for creating and modifying variables. After each comment write a line of code that implements the instruction.
    ##Note that this code uses scientific notation to define large numbers. 4.445e8 is equal to 4.445 * 10 ** 8 which is equal to 444500000.0.

# The current volume of a water reservoir (in cubic metres)
reservoir_volume = 4.445e8
# The amount of rainfall from a storm (in cubic metres)
rainfall = 5e6

# decrease the rainfall variable by 10% to account for runoff
rainfall *= 0.90

# add the rainfall variable to the reservoir_volume variable
reservoir_volume += rainfall

# increase reservoir_volume by 5% to account for stormwater that flows
# into the reservoir in the days following the storm
reservoir_volume *= 1.05

# decrease reservoir_volume by 5% to account for evaporation
reservoir_volume *= 0.95

# subtract 2.5e5 cubic metres from reservoir_volume to account for water
# that's piped to arid regions.
reservoir_volume -= 2.5e5

# print the new value of the reservoir_volume variable
print(reservoir_volume)


# TYPES

print(type(4))
    ## check type of 4 = integer

# can make a float by adding decimal point after whole # (387.)

print(int(49.7))
    ## changes float to an int = 49
    ## no rounding occurs

# Integers and Floats
## There are two Python data types that could be used for numeric values:
## int - for integer values
## float - for decimal or floating point values

## You can create a value that follows the data type by using the following syntax:
x = int(4.7)   # x is now an integer 4
y = float(4)   # y is now a float of 4.0

## You can check the type by using the type function:
>>> print(type(x))
int
>>> print(type(y))
float

## Because the float, or approximation, for 0.1 is actually slightly more than 0.1, when we add several of them together we can see the difference between the mathematically correct answer and the one that Python creates.
>>> print(.1 + .1 + .1 == .3)
False

# BOOLEANS, COMPARISON OPERATORS, LOGICAL OPERATORS

bool = boolean 

age = 14
is_teen = age > 12 and age < 20
print(is_teen) 
result = True

# Quiz: Which is denser, Rio or San Francisco?
## Try comparison operators in this quiz! This code calculates the population densities of Rio de Janeiro and San Francisco.
## Write code to compare these densities. Is the population of San Francisco more dense than that of Rio de Janeiro? Print True if it is and False if not.

sf_population, sf_area = 864816, 231.89
rio_population, rio_area = 6453682, 486.5

san_francisco_pop_density = sf_population/sf_area
rio_de_janeiro_pop_density = rio_population/rio_area

# Write code that prints True if San Francisco is denser than Rio, and False otherwise

is_denser = san_francisco_pop_density > rio_de_janeiro_pop_density
print(is_denser)
# OR
print(san_francisco_pop_density > rio_de_janeiro_pop_density) # = best!
# OR
if (san_francisco_pop_density > rio_de_janeiro_pop_density):
    print (True)
else:
    print (False)

# STRINGS

first_word = "Hello"
second_word = "There"
print(first_word + " " + second_word)
# = Hello World

first_word = "Hello"
second_word = "There"
print(first_word * 2)
# = HelloHello

udacity_length = len("Udacity")
# tells length of string, returns value that can be stored in Udacity length variable

# Quiz: Write a Server Log Message
## In this programming quiz, you’re going to use what you’ve learned about strings to write a logging message for a server.
## You’ll be provided with example data for a user, the time of their visit and the site they accessed. You should use the variables provided and the techniques you’ve learned to print a log message like this one (with the username, url, and timestamp replaced with values from the appropriate variables):
## Yogesh accessed the site http://petshop.com/pets/reptiles/pythons at 16:20.
## Use the Test Run button to see your results as you work on coding this piece by piece.

username = "Kinari"
timestamp = "04:50"
url = "http://petshop.com/pets/mammals/cats"

# TODO: print a log message using the variables above.
# The message should have the same format as this one:
# "Yogesh accessed the site http://petshop.com/pets/reptiles/pythons at 16:20."

print(username + " accessed the site " + url + " at " + timestamp + '.')
# OR
message = username + " accessed the site " + url + " at " + timestamp + "."
print(message)

# QUIZ: fix the quote

# TODO: Fix this string!
ford_quote = 'Whether you think you can, or you think you can\'t--you\'re right.'

# TODO: Fix this string!
ford_quote = "Whether you think you can, or you think you can't--you're right."


# Quiz: len()
## Use string concatenation and the len() function to find the length of a certain movie star's actual full name. Store that length in the name_length variable. Don't forget that there are spaces in between the different parts of a name!
given_name = "William"
middle_names = "Bradley"
family_name = "Pitt"

name_length = len(given_name) + len(middle_names) + len(family_name) + 2
print(name_length)


# TYPE AND TYPE CONVERSION

print(type(633)) # type is run first in (), then print

# Conversion - integer -> string
house_number = 13
street_name = "The Crescent"
town_name = "Belmont"
print(type(house_number))

address = str(house_number) + " " + street_name + ", " + town_name
print (address)

# Conversion - string -> float
grams = "35.0"
print(type(grams))
grams = float(grams)
print(type(grams)) 

# Quiz: Total Sales
## In this quiz, you’ll need to change the types of the input and output data in order to get the result you want.
## Calculate and print the total sales for the week from the data provided. Print out a string of the form "This week's total sales: xxx", where xxx will be the actual total of all the numbers. You’ll need to change the type of the input data in order to calculate that total.

mon_sales = "121"
tues_sales = "105"
wed_sales = "110"
thurs_sales = "98"
fri_sales = "95"

#TODO: Print a string with this format: This week's total sales: xxx
# You will probably need to write some lines of code before the print statement.

week_sales = int(mon_sales) + int(tues_sales) + int(wed_sales) + int(thurs_sales) + int(fri_sales)
print("This week\'s total sales: " + str(week_sales))
# OR
weekly_sales = int(mon_sales) + int(tues_sales) + int(wed_sales) + int(thurs_sales) + int(fri_sales)
weekly_sales = str(weekly_sales)  #convert the type back!!
print("This week's total sales: " + weekly_sales)


# STRING METHODS

print("sebastian thrun".title())
    ## result: Sebastian Thrun

full_name = "sebastian thrun"
print(full_name.islower())
    ## result: True, because there are no uppercase letters

# () with nothing inside, are referring to/disguised as the string object, here: "sebastian thrun"
print("sebastian thrun".title())

# count
print("One fish, two fish, red fish, blue fish.".count('fish'))
    ## counts how many times fish is there, in this case, 4

# Browse the complete list of string methods at:
# https://docs.python.org/3/library/stdtypes.html#string-methods
# and try them out here

print("hello".capitalize())
# Hello
print("The sum of 1 + 2 is {}".format(1+2))
# The sum of 1 + 2 is 3
print("The sum of 1 + 2 is {}.".format("dog"))
# The sum of 1 + 2 is dog.
# 
# # Write two lines of code below, each assigning a value to a variable

my_dog = "sparky"
my_cat = "gatinha"

# Now write a print statement using .format() to print out a sentence and the 
#   values of both of the variables

print("My animals are named {} and {}.".format(my_dog, my_cat))
#  output: My animals are named sparky and gatinha.

# Write two lines of code below, each assigning a value to a variable

my_dog = 11
my_cat = 3

# Now write a print statement using .format() to print out a sentence and the 
#   values of both of the variables

print("The total age of my pets is {}.".format(my_dog + my_cat))
# output: The total age of my pets is 14.

# STRING METHODS PRACTICE

# Version 1
verse = "If you can keep your head when all about you\n  Are losing theirs and blaming it on you,\nIf you can trust yourself when all men doubt you,\n  But make allowance for their doubting too;\nIf you can wait and not be tired by waiting,\n  Or being lied about, don’t deal in lies,\nOr being hated, don’t give way to hating,\n  And yet don’t look too good, nor talk too wise:"
print(verse, "\n")

print("Verse has a length of {} characters.".format(len(verse)))
print("The first occurence of the word 'and' occurs at the {}th index.".format(verse.find('and')))
print("The last occurence of the word 'you' occurs at the {}th index.".format(verse.rfind('you')))
print("The word 'you' occurs {} times in the verse.".format(verse.count('you')))

# Version 2
## Here's another way you could write the print statements and get the same output.
verse = "If you can keep your head when all about you\n  Are losing theirs and blaming it on you,\nIf you can trust yourself when all men doubt you,\n  But make allowance for their doubting too;\nIf you can wait and not be tired by waiting,\n  Or being lied about, don’t deal in lies,\nOr being hated, don’t give way to hating,\n  And yet don’t look too good, nor talk too wise:"
print(verse, "\n")
message = "Verse has a length of {} characters.\nThe first occurence of the \
word 'and' occurs at the {}th index.\nThe last occurence of the word 'you' \
occurs at the {}th index.\nThe word 'you' occurs {} times in the verse."

length = len(verse)
first_idx = verse.find('and')
last_idx = verse.rfind('you')
count = verse.count('you')

print(message.format(length, first_idx, last_idx, count))
# Output:
# If you can keep your head when all about you
#   Are losing theirs and blaming it on you,
# If you can trust yourself when all men doubt you,
#   But make allowance for their doubting too;
# If you can wait and not be tired by waiting,
#   Or being lied about, don’t deal in lies,
# Or being hated, don’t give way to hating,
#   And yet don’t look too good, nor talk too wise: 

# Verse has a length of 362 characters.
# The first occurence of the word 'and' occurs at the 65th index.
# The last occurence of the word 'you' occurs at the 186th index.
# The word 'you' occurs 8 times in the verse.