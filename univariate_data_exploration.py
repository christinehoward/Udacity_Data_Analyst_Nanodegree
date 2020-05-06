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