plt.scatter(data = feul_econ, x = 'displ, y = 'comb);
plt.xlabel('Displacement')
plt.ylabel('Comb. fuel effec.');

# seaborn's scatterplot instead of matplotlib
sb.regplot(data = feul_econ, x = 'displ, y = 'comb);
plt.xlabel('Displacement')
plt.ylabel('Comb. fuel effec.');
# can also turn off the line:
sb.regplot(data = feul_econ, x = 'displ, y = 'comb', fit_reg = False);


# Scatterplot notes
plt.scatter(data = df, x = 'num_var1', y = 'num_var2')

# Alternative Approach
# Seaborn's regplot function combines scatterplot creation with regression function fitting:
sb.regplot(data = df, x = 'num_var1', y = 'num_var2')

def log_trans(x, inverse = False):
    if not inverse:
        return np.log10(x)
    else:
        return np.power(10, x)

sb.regplot(df['num_var1'], df['num_var2'].apply(log_trans))
tick_locs = [10, 20, 50, 100, 200, 500]
plt.yticks(log_trans(tick_locs), tick_locs)


# Overplotting transparency and jitter

# random sampling is one way to have fewer points on scatterplot
# we can otherwise plot all points, but with some transparency on points
# jitter adds a small amount of random noise to each point, so multiple data points with same values are spread over small area


# Scatterplot example
sb.regplot(data = fuel_econ, x = 'year', y = 'comb');

# start by jittering on the x-axis
sb.regplot(data = fuel_econ, x = 'year', y = 'comb', x_jitter = 0.3);
# setting jitter to 0.3 means that the data points can be adjusted to up to 0.3 more or less than actual point

# adding transparency
sb.regplot(data = fuel_econ, x = 'year', y = 'comb', x_jitter = 0.3,
            scatter_kws = {'alpha' : 1/20}); # alpha takes value 0-1 specifying opaqueness, 0 = transparent, 1 = opaque


# Heat Maps

# grids a scatterplot, and shades based on number of data points
# like a 2d version of a histogram, looking top down
# would want to add count annotations on top of each grid cell

# heat map favored over scatterplot when there are 2 discrete variables, quant vs. quant
# good alternative to transparency for a large amount of data

# but bin sizes are very important like in histograms

# plotting heat map
plt.hist2d(data = feul_econ, x = 'displ, y = 'comb', cmin = 0.5,
            cmap = 'viridis_r', bins = [bins_x, bins_y]); # cmin makes values that are 0 color in white
            # cmap = 'viridis_r' = reversed color map
            # finally lets set the bins, can set for x and y axises
# also adding colorbar so there is a color legend adjacent to plot
plt.colorbar()
plt.xlabel('Displacement')
plt.ylabel('Comb. fuel effec.');
# want to choose reversed color pallet where darker colors associated with higher values

# use np.arange to set bins_x and bins_y after looking at describe

bins_x = np.arange(0.6, 7+0.3, 0.3)
bins_y = np.arange(12, 58+3, 3)


# heat maps another example

plt.figure(figsize = [12, 5])

# left plot: scatterplot of discrete data with jitter and transparency
plt.subplot(1, 2, 1)
sb.regplot(data = df, x = 'disc_var1', y = 'disc_var2', fit_reg = False,
           x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/3})

# right plot: heat map with bin edges between values
plt.subplot(1, 2, 2)
bins_x = np.arange(0.5, 10.5+1, 1)
bins_y = np.arange(-0.5, 10.5+1, 1)
plt.hist2d(data = df, x = 'disc_var1', y = 'disc_var2',
           bins = [bins_x, bins_y])
plt.colorbar();

# By adding a cmin = 0.5 parameter to the hist2d call, this means that a cell will only get colored if it contains at least one point.
bins_x = np.arange(0.5, 10.5+1, 1)
bins_y = np.arange(-0.5, 10.5+1, 1)
plt.hist2d(data = df, x = 'disc_var1', y = 'disc_var2',
           bins = [bins_x, bins_y], cmap = 'viridis_r', cmin = 0.5)
plt.colorbar()

# We can get the counts to annotate directly from what is returned by hist2d, 
# which includes not just the plotting object, but an array of counts and two vectors of bin edges.
# hist2d returns a number of different variables, including an array of counts
bins_x = np.arange(0.5, 10.5+1, 1)
bins_y = np.arange(-0.5, 10.5+1, 1)
h2d = plt.hist2d(data = df, x = 'disc_var1', y = 'disc_var2',
               bins = [bins_x, bins_y], cmap = 'viridis_r', cmin = 0.5)
counts = h2d[0]

# loop through the cell counts and add text annotations for each
for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        c = counts[i,j]
        if c >= 7: # increase visibility on darkest cells
            plt.text(bins_x[i]+0.5, bins_y[j]+0.5, int(c),
                     ha = 'center', va = 'center', color = 'white')
        elif c > 0:
            plt.text(bins_x[i]+0.5, bins_y[j]+0.5, int(c),
                     ha = 'center', va = 'center', color = 'black')


# Solutions: scatterplot quiz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def scatterplot_solution_1():
    """
    Solution for Question 1 in scatterplot practice: create a scatterplot of
    city vs. highway fuel mileage.
    """
    sol_string = ["Most of the data falls in a large blob between 10 and 30 mpg city",
                  "and 20 to 40 mpg highway. Some transparency is added via 'alpha'",
                  "to show the concentration of data. Interestingly, for most cars",
                  "highway mileage is clearly higher than city mileage, but for those",
                  "cars with city mileage above about 30 mpg, the distinction is less",
                  "pronounced. In fact, most cars above 45 mpg city have better",
                  "city mileage than highway mileage, contrary to the main trend. It",
                  "might be good to call out this trend by adding a diagonal line to",
                  "the figure using the `plot` function. (See the solution file for that code!)"]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')

    plt.scatter(data = fuel_econ, x = 'city', y = 'highway', alpha = 1/8)
    # plt.plot([10,60], [10,60]) # diagonal line from (10,10) to (60,60)
    plt.xlabel('City Fuel Eff. (mpg)')
    plt.ylabel('Highway Fuel Eff. (mpg)')


def scatterplot_solution_2():
    """
    Solution for Question 2 in scatterplot practice: create a heat map of
    engine displacement vs. co2 production.
    """
    sol_string = ["In the heat map, I've set up a color map that goes from light",
                  "to dark, and made it so that any cells without count don't get",
                  "colored in. The visualization shows that most cars fall in a",
                  "line where larger engine sizes correlate with higher emissions.",
                  "The trend is somewhat broken by those cars with the lowest emissions,",
                  "which still have engine sizes shared by most cars (between 1 and 3 liters)."]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')

    bins_x = np.arange(0.6, fuel_econ['displ'].max()+0.4, 0.4)
    bins_y = np.arange(0, fuel_econ['co2'].max()+50, 50)
    plt.hist2d(data = fuel_econ, x = 'displ', y = 'co2', bins = [bins_x, bins_y], 
               cmap = 'viridis_r', cmin = 0.5)
    plt.colorbar()
    plt.xlabel('Displacement (l)')
    plt.ylabel('CO2 (g/mi)')


# Violin Plots

# plotting quant vs. qual.
# histogram turned on its side

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses);
# changed type to ordered categorical type, putting the classes in order of size
# this way sorting of levels is automatic

sb.violinplot(data = fuel_econ, x = 'VClass', y = 'comb', color = base_color, inner = None);
# clean this up a bit, use same color for all violins, and rotate the x-ticks so we can see more easily
# inner = none cleans up the middle of the plots
base_color = sb.color_palette()[0]
plt.xticks(rotation = 15);
# when we set inner = None, we erase a miniature box plot inside violin

# can set quartiles/median as dashed lines (inner):
sb.violinplot(data = fuel_econ, x = 'VClass', y = 'comb', color = base_color, inner = 'quartile');

# Box plots

base_color = sb.color_palette()[0]
sb.boxplot(data = fuel_econ, x = 'VClass', y = 'comb', color = base_color);
plt.xticks(rotation = 15);

# top of box: 3rd quartile, bottom of box: 1st, quartile, line: median, whiskers: min/max, outliers are individual points


# Violin/box plot practice solution

def violinbox_solution_1():
    """
    Solution for Question 1 in violin and box plot practice: plot the relationship
    between vehicle class and engine displacement.
    """
    sol_string = ["I used a violin plot to depict the data in this case; you might",
                  "have chosen a box plot instead. One of the interesting things",
                  "about the relationship between variables is that it isn't consistent.",
                  "Compact cars tend to have smaller engine sizes than the minicompact",
                  "and subcompact cars, even though those two vehicle sizes are smaller.",
                  "The box plot would make it easier to see that the median displacement",
                  "for the two smallest vehicle classes is greater than the third quartile",
                  "of the compact car class."]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')

    sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
    pd_ver = pd.__version__.split(".")
    if (int(pd_ver[0]) > 0) or (int(pd_ver[1]) >= 21): # v0.21 or later
        vclasses = pd.api.types.CategoricalDtype(ordered = True, categories = sedan_classes)
        fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)
    else: # pre-v0.21
        fuel_econ['VClass'] = fuel_econ['VClass'].astype('category', ordered = True,
                                                         categories = sedan_classes)

    # plotting
    base_color = sb.color_palette()[0]
    sb.violinplot(data = fuel_econ, x = 'VClass', y = 'displ',
                  color = base_color)
    plt.xticks(rotation = 15)


# Clustered bar charts

# 2 categorical variables, quant versus qual variables

fuel_econ['trans_type'] = fuel_econ['trans'].apply(lambda x: x.split()[0])

sb.heatmap(ct_counts) # this takes a 2d array with the values to be depicted 
# need to so some summarization ourself before we can plot

fuel_econ.groupby(['VClass', 'trans_type']).size()
# first: groupby, size functions, to get the number of cars in each combination of the 2 variables levels as pandas series\
ct_counts.reset_index(name = 'count') # then use reset index to convert series to df
ct_counts.pivot(index = 'VClass', columns = 'trans_type', values = 'count')
# finally, use pivot to rearrange data so i have vehicle class on rows, trans types on columns, and values in the cells

# now we can finally plot
sb.heatmap(ct_counts, annot = True, fmt = 'd'); # can add annotations w/ counts to the cells  
# fmt = 'd' means all counts are set to decimal values

# could also show the distribution in counts using clustered bar chart

sb.countplot(data = fuel_econ, x = 'VClass', hue = 'trans_type') # to divide each bar into multiple bars for the diff transmissions,
# just need to add hue type pointing to trans_type variable 
plt.xticks(rotation = 15);


# Categorical plot practice solution

def categorical_solution_1():
    """
    Solution for Question 1 in categorical plot practice: plot the relationship
    between vehicle class and fuel type.
    """
    sol_string = ["I chose a clustered bar chart instead of a heat map in this case",
                  "since there weren't a lot of numbers to plot. If you chose a heat",
                  "map, did you remember to add a color bar and include annotations?",
                  "From this plot, you can see that more cars use premium gas over",
                  "regular gas, and that the smaller cars are biased towards the",
                  "premium gas grade. It is only in midsize sedans where regular",
                  "gasoline was used in more cars than premium gasoline."]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')
    
    sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
    pd_ver = pd.__version__.split(".")
    if (int(pd_ver[0]) > 0) or (int(pd_ver[1]) >= 21): # v0.21 or later
        vclasses = pd.api.types.CategoricalDtype(ordered = True, categories = sedan_classes)
        fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)
    else: # pre-v0.21
        fuel_econ['VClass'] = fuel_econ['VClass'].astype('category', ordered = True,
                                                         categories = sedan_classes)
    fuel_econ_sub = fuel_econ.loc[fuel_econ['fuelType'].isin(['Premium Gasoline', 'Regular Gasoline'])]

    # plotting
    ax = sb.countplot(data = fuel_econ_sub, x = 'VClass', hue = 'fuelType')
    ax.legend(loc = 4, framealpha = 1) # lower right, no transparency
    plt.xticks(rotation = 15)


# Faceting
## multiple copies of the same type of plot visualized on diff subsets of the data
## good for breaking down potentially complex patterns into simpler parts 
## especially useful for categorical variable with lots of levels

# use seaborn's facetgrid
g = sb.FacetGrid(data = fuel_econ, col = 'VClass', col_wrap = 3);
# for whatever plot we facet with, there will be 1 level of the plot made for each vehicle class
g.map(plt.hist, 'comb', bins = bins) # then we say what kind of grid we want
# then variable on x-axis (comb)
# grids will be plotted with default 10 bins
bins = np.arange(12, 58+2, 2) # add bins
# then add col wrap to limit number of facets plotted next to each other


# Adaptations of univariate plots

# adaptation of bar charts
base_color = sb.color_palette()[0]
b.barplot(data = fuel_econ, x = 'VClass', y = 'comb', color = base_color, errwidth = 0);
plt.xticks(rotation = 15);
plt.ylabel('avg comb fuel effec.')
# we will get error bars in the end result (for the mean)
# if we do not want these, set errwidth = 0 as above
# could also set the bars to show std dev of data, using ci = 'sd' (replacing errwidth)


# Line plots
# why line instead of bar?
        # interested in relative change
        # empahsize trends across x-values
plt.errorbar(data = fuel_econ, x = 'VClass', y - 'comb')
plt.xticks(rotation = 15)
plt.ylabel('avg...');
# in order to use error bar, all data needs to be sorted by x variable and we only have 1 y value for each x value

# first, set bin edges and centers
bins_e = np.arange(0.6, 7+0.2, 0.2) #edges
bins_c = bins_e[:-1] + 0.1 #centers are needed so point values are plotted in accurate positions
# leaving out final bin center as that will be an edge and not have a center
displ_binned = pd.cut(fuel_econ['displ'], bins_e, include_lowest = True)
comb_mean = fuel_econ['comb'].groupby(displ_binned).mean() # use pandas cut to find out in which bin each data point should be used in
# first argument: series to be sliced, 2nd argument, set of bins
# final makes sure that the lowest values are included in the bins
# then use groupby to group displacement bins, then take the mean of points that fall in each

plt.errorbar(x = bins_c, y = comb_mean);
plt.xlabel('Displacement (1)')
plt.ylabel('avg...');
comb_mean = fuel_econ['comb'].groupby(displ_binned).mean()
comb_std = fuel_econ['comb'].groupby(displ_binned).std() # with error bar can also plot std dev of fuel effeciencies


# additional plot quiz solution

def additionalplot_solution_1():
    """
    Solution for Question 1 in additional plots practice: plot the distribution
    of combined fuel efficiencies for each manufacturer with at least 80 cars.
    """
    sol_string = ["Due to the large number of manufacturers to plot, I've gone",
                  "with a faceted plot of histograms rather than a single figure",
                  "like a box plot. As part of setting up the FacetGrid object, I",
                  "have sorted the manufacturers by average mileage, and wrapped",
                  "the faceting into a six column by three row grid. One interesting",
                  "thing to note is that there are a very large number of BMW cars",
                  "in the data, almost twice as many as the second-most prominent",
                  "maker, Mercedes-Benz. One possible refinement could be to change",
                  "the axes to be in terms of relative frequency or density to",
                  "normalize the axes, making the less-frequent manufacturers",
                  "easier to read."]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')
    
    THRESHOLD = 80
    make_frequency = fuel_econ['make'].value_counts()
    idx = np.sum(make_frequency > THRESHOLD)

    most_makes = make_frequency.index[:idx]
    fuel_econ_sub = fuel_econ.loc[fuel_econ['make'].isin(most_makes)]

    make_means = fuel_econ_sub.groupby('make').mean()
    comb_order = make_means.sort_values('comb', ascending = False).index

    # plotting
    g = sb.FacetGrid(data = fuel_econ_sub, col = 'make', col_wrap = 6, size = 2,
                     col_order = comb_order)
    # try sb.distplot instead of plt.hist to see the plot in terms of density!
    g.map(plt.hist, 'comb', bins = np.arange(12, fuel_econ_sub['comb'].max()+2, 2))
    g.set_titles('{col_name}')


def additionalplot_solution_2():
    """
    Solution for Question 2 in additional plots practice: plot the average
    combined fuel efficiency for each manufacturer with at least 80 cars.
    """
    sol_string = ["Seaborn's barplot function makes short work of this exercise.",
                  "Since there are a lot of 'make' levels, I've made it a horizontal",
                  "bar chart. In addition, I've set the error bars to represent the",
                  "standard deviation of the car mileages."]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')
    
    THRESHOLD = 80
    make_frequency = fuel_econ['make'].value_counts()
    idx = np.sum(make_frequency > THRESHOLD)

    most_makes = make_frequency.index[:idx]
    fuel_econ_sub = fuel_econ.loc[fuel_econ['make'].isin(most_makes)]

    make_means = fuel_econ_sub.groupby('make').mean()
    comb_order = make_means.sort_values('comb', ascending = False).index

    # plotting
    base_color = sb.color_palette()[0]
    sb.barplot(data = fuel_econ_sub, x = 'comb', y = 'make',
               color = base_color, order = comb_order, ci = 'sd')
    plt.xlabel('Average Combined Fuel Eff. (mpg)')


# Plot types

# scatterplots: 2 quant variables
# clustered bar charts: 2 qual variables
# heat maps: 2d histograms/bar charts
# violin/box plots: used to show relationship between 1 quant/1 qual variable
# faceting multiple univariate plots across subsets of second variable, using 2nd variable's mean instead of count
# barcharts/line plots: show changes in value across time