# Bar charts

import numpy as np 
import pandas as pd 
import matplotlib as plt 
import seaborn as sb 
%matplotlib inline # allows graphs in jupyter

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.shape)
pokemon.head(10)

sb.countplot(data = pokemon, x = 'generation_id', color = base_color);
# countplot, seaborn function for bar chart
# set the data source, pokemon dataframe
# variable of interest, generation_id

# the bars come out multicolored, which i want to change to 1 color
base_color = sb.color_palette()[0]
# returns list of tuples included in the color pallette's arguments
# since we just want first color, want to slice off first tuple in this list
# storing this in base color variable
# then add color value above

sb.countplot(data = pokemon, x = 'generation_id', color = base_color, order = [5,3,2,4...])
# we want to order by number of species introduced in each generation
# can type in the order you want, but better to code this

gen_order = pokemon['generation_id'].value_counts().index
# we can use value counts to help with this
# in order to get the bar order, we want the index values
# we pass this to the order parameter in order to get sorted bar chart

sb.countplot(data = pokemon, x = 'generation_id', color = base_color, order = gen_order);


# we now want to plot the type_1 of the pokemon
base_color = sb.color_palette()[0]
sb.countplot(data = pokemon, x = 'type_1', color = base_color);
plt.xticks(rotation = 90) ;
# we want to fix the overlapping x labels, and to rotate the category labels
# by setting the rotation as 90, we set these at 90 degrees
# could very easily change this to a vertical barchart by changing x above to y


# Absolute vs. relative frequency

# absolute is what we saw above by looking at counts
# relative looks at proportions

# changing to bar chart relative counts
# we start by calculating the longest bar in terms of proportion

n_pokemon = pokemon.shape[0]
max_type_count = type_counts[0]
max_prop = max_type_count / n_pokemon
print(max_prop)

# now we use numpy's arange function to produce a set of evenly spaced proportion values
tick_props = np.arange(0, max_prop, 0.02)
# between 0 and the max_prop in steps of 2% and store in tick_props variable

# also using a list comprehension to create an additional variable, tick_names, to apply to the tick labels
tick_names = ['{:0.2f}'.format(v) for v in tick_props]

type_counts = pkmn_types['type'].value_counts()
type_order = type_counts.index
base_color = sb.color_palette()[0]
sb.countplot(data = pkmn_types, y = 'type', color = base_color, order = type_order);
plt.xticks(tick_props * n_pokemon, tick_names) # need to multiply to get the correct position
# second argument sets the tick labels, we need to use both values here as their positions and values are different
# then we change x label title from count to proportion

# alternatively, maybe we want to have the axis in terms of counts, and use text on the bars to show proportion
# to do this, we use a loop to place text elements 1 by 1
for i in range(type_counts.shape[0]):
    plt.text(count+1, i, pct_string, va = 'center'); # first argument is the text position just after the end of the bar, then we add the y value
    # third argument is the string to be printed
    # finally we add the optional parameter 'va' to center the vertical alignment


# One method of plotting the data in terms of relative frequency on a bar chart is to just relabel the counts axis in terms of proportions. The underlying data will be the same, it will simply be the scale of the axis ticks that will be changed.

# get proportion taken by most common group for derivation
# of tick marks
n_points = df.shape[0]
max_count = df['cat_var'].value_counts().max()
max_prop = max_count / n_points

# generate tick mark locations and names
tick_props = np.arange(0, max_prop, 0.05)
tick_names = ['{:0.2f}'.format(v) for v in tick_props]

# create the plot
base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'cat_var', color = base_color)
plt.yticks(tick_props * n_points, tick_names)
plt.ylabel('proportion')


# Additional Variation
# Rather than plotting the data on a relative frequency scale, you might use text annotations to label the frequencies on bars instead. 
# This requires writing a loop over the tick locations and labels and adding one text element for each bar.

# create the plot
base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'cat_var', color = base_color)

# add annotations
n_points = df.shape[0]
cat_counts = df['cat_var'].value_counts()
locs, labels = plt.xticks() # get the current tick locations and labels

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = cat_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/n_points)

    # print the annotation just below the top of the bar
    plt.text(loc, count-8, pct_string, ha = 'center', color = 'w')
# I use the .get_text() method to obtain the category name, so I can get the count of each category level. At the end, I use the text 
# function to print each percentage, with the x-position, y-position, and string as the three main parameters to the function.


# Counting Missing Data
# One interesting way we can apply bar charts is through the visualization of missing data. 
# We can use pandas functions to create a table with the number of missing values in each column.

df.isna().sum()


# Seaborn's barplot function is built to depict a summary of one quantitative variable against levels of a second, qualitative variable, but can be used here.

na_counts = df.isna().sum()
base_color = sb.color_palette()[0]
sb.barplot(na_counts.index.values, na_counts, color = base_color)
# The first argument to the function contains the x-values (column names), the second argument the y-values (our counts).


# Bar chart practice

# prerequisite package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
​
%matplotlib inline
​
# solution script imports
from solutions_univ import bar_chart_solution_1, bar_chart_solution_2
In this workspace, you'll be working with this dataset comprised of attributes of creatures in the video game series Pokémon. The data was assembled from the database of information found in this GitHub repository.

pokemon = pd.read_csv('./data/pokemon.csv')
pokemon.head()
pokemon = pd.read_csv('./data/pokemon.csv')
pokemon.head()

# Task 1: There have been quite a few Pokémon introduced over the series' history. How many were introduced in each generation? 
# Create a bar chart of these frequencies using the 'generation_id' column.

base_color = sb.color_palette()[0]
sb.countplot(data = pokemon, x = 'generation_id', color = base_color);


# Task 2: Each Pokémon species has one or two 'types' that play a part in its offensive and defensive capabilities. How frequent is each type? The code below creates a new dataframe that puts all of the type counts in a single column.

pkmn_types = pokemon.melt(id_vars = ['id','species'], 
                          value_vars = ['type_1', 'type_2'], 
                          var_name = 'type_level', value_name = 'type').dropna()
pkmn_types.head()

# Your task is to use this dataframe to create a relative frequency plot of the proportion of Pokémon with each type, sorted from most frequent to least. 
# Hint: The sum across bars should be greater than 100%, since many Pokémon have two types. Keep this in mind when considering a denominator to compute relative frequencies.

type_counts = pkmn_types['type'].value_counts()
type_order = type_counts.index
tick_props = np.arange(0, max_prop, 0.02)
tick_names = ['{:0.2f}'.format(v) for v in tick_props]

n_pokemon = pokemon.shape[0]
max_type_count = type_counts[0]
max_prop = max_type_count / n_pokemon
print(max_prop)

sb.countplot(data = pkmn_types, y = 'type', color = base_color, order = type_order);
plt.xticks(tick_props * n_pokemon, tick_names);

# I created a horizontal bar chart since there are a lot of Pokemon types. 
# The unique() method was used to get the number of different Pokemon species. 
# I also added an xlabel call to make sure it was clear the bar length represents a relative frequency.


# PIE CHARTS

# You can create a pie chart with matplotlib's pie function. This function requires that the data be in a summarized form: 
# the primary argument to the function will be the wedge sizes.

sorted_counts = df['cat_var'].value_counts()
plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,
        counterclock = False);
plt.axis('square')

# To create a donut plot, you can add a "wedgeprops" argument to the pie function call. By default, the radius of the pie (circle) is 1; 
# setting the wedges' width property to less than 1 removes coloring from the center of the circle.

sorted_counts = df['cat_var'].value_counts()
plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,
        counterclock = False, wedgeprops = {'width' : 0.4});
plt.axis('square')


# HISTOGRAMS

plt.hist(data = pokemon, x = 'speed');
# default = 10 bins, and bins are not well-aligned with x-ticks

plt.hist(data = pokemon, x = 'speed', bins = 20);
# 20 bins
# bin boundaries are still off, can see these boundaries when removing the ; above

# better to set explicit bin boundaries
bins = np.arange(0, pokemon['speed'].max()+5, 5) # set bins of size 5
# 1st argument: min value, 2nd: max value, 3rd: step size for bins
# adding +5 to max value, as arange will not include max value otherwise

# adding bins list to the hist function
plt.hist(data = pokemon, x = 'speed', bins = bins)

# can also create historgrams with seaborn
sb.distplot(pokemon['speed'])
# main argument is pandas series including all data
# produces a line: density curve estimate of the data distribution, total area under the curve is set to be equal to 1
# y-values are proportions

sb.distplot(pokemon['speed'], kde = False)
# can turn off density curve by setting kde = False
# without this, you can see histogram plotted alone, with counts, not proportions, on y-axis