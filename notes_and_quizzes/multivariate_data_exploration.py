sb.regplot(data = fuel_econ_subset, x = 'displ', y = 'comb', 
            x_jitter = 0.04, fit_reg = False, marker = ) # marker can only take one argument, so we will need to loop
plt.xlabel('Displacement')
plt.ylabel('comb fuel eff')

# set list of lists to have more control over order in which things are plotted
ttype_markers = [['Automatic', 'o'], # cars with auto trans marked with circles (o)
                ['Manual', '^']] # cars with manual trans marked with triangles (^)
for ttype, marker in ttype_markers: # then loop over list elements
    plot_data = fuel_econ_subset.loc[fuel_econ_subset['trans_type'] == ttype] # use ttype to match by transmission type and set marker in reg plot function
    sb.regplot(data = plot_data, x = 'displ', y = 'comb', 
            x_jitter = 0.04, fit_reg = False, marker = marker) # marker can only take one argument, so we will need to loop
plt.xlabel('Displacement')
plt.ylabel('comb fuel eff')
plt.legend(['Auto', 'Manual']) # add a legend

# want to add a 3rd variable: co2 emissionsand differentiate with marker size
sb.regplot(data = fuel_econ, x = 'displ', y = 'comb', 
            x_jitter = 0.04, fit_reg = False, 
            scatter_kws = {'s' : fuel_econ_subset ['co2']/2}) # need to set parameter as part of dictionary on scattered key words parameter
            # and explicitly assign it the full series, not just column name
            # /2 sets marker size smaller
plt.xlabel('Displacement')
plt.ylabel('comb fuel eff')
plt.legend(['size?'])

sizes = [200, 350, 500] # set up a loop to go over 3 co2 values we want to use for size legend reference
base_color = sb.color_palette()[0]
legend_obj = []
for s in sizes:
    plt.scatter([], [], s = s/2, color = base_color) # inside loop, use scatter to set scatterplot objects by setting necessary point sizes
    # x and y set as empty lists so nothing actually plotted
    # first set up a list to store all dummy scatterplot objects
plt.legend(legend_obj, sizes, title = 'CO2 (g/mi)'); # first, object to be depicted in legend, 2nd: list of labels
# also added title element to get the units of the object values


# Color palettes

g = sb.FacetGrid(data = fuel_econ_subset, hue = 'trans_type',
                hue_order = ['Automatic', 'Manual'], size = 4, aspect = 1.5) # switch which value is on top
                # 50% longer than it is tall
g.map(sb.regplot, 'displ', 'comb', x_jitter = 0.04, fit_reg = False)
g.add_legend()
plt.xlabel('Displacement')
plt.ylabel('comb fuel eff')

# if we set hue as 'VClass':
g = sb.FacetGrid(data = fuel_econ_subset, hue = 'VClass', size = 4, aspect = 1.5, palette = 'viridis_r') # switching color palette
g.map(sb.regplot, 'displ', 'comb', x_jitter = 0.04, fit_reg = False)
g.add_legend()
plt.xlabel('Displacement')
plt.ylabel('comb fuel eff')

# if we set hue as 'co2', numeric: instead should use scatter
plt.scatter(data = fuel_econ_subset, x = 'Displ', y = 'comb', c = 'co2', cmap = 'viridis_r')
plt.colorbar(label = 'CO2')
plt.xlabel('Displacement')
plt.ylabel('comb fuel eff')


# Eoncodings practice solutions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def encodings_solution_1():
    """
    Solution for Question 1 in encodings practice: see if pokemon speed has a
    clear relationship with defense and special defense.
    """
    sol_string = ["When creating the plot, I made the figure size bigger and set",
                  "axis limits to zoom into the majority of data points. I might",
                  "want to apply some manual jitter to the data since I suspect",
                  "there to be a lot of overlapping points. From the plot as given,",
                  "I see a slight increase in speed as both defense and special",
                  "defense increase. However, the brightest points seem to be clumped",
                  "up in the center in the 60-80 defense and special defense ranges",
                  "with the two brightest points on the lower left of the diagonal."]
    print((" ").join(sol_string))

    # data setup
    pokemon = pd.read_csv('./data/pokemon.csv')

    # plotting
    plt.figure(figsize = [8,6])
    plt.scatter(data = pokemon, x = 'defense', y = 'special-defense',
                c = 'speed')
    plt.colorbar(label = 'Speed')
    plt.xlim(0,160)
    plt.ylim(15,160)
    plt.xlabel('Defense')
    plt.ylabel('Special Defense')


def encodings_solution_2():
    """
    Solution for Question 2 in encodings practice: compare the heights and
    weights for two extreme types of pokemon, fairy and dragon.
    """
    sol_string = ["After subsetting the data, I used FacetGrid to set up and",
                  "generate the plot. I used the .set() method for FacetGrid",
                  "objects to set the x-scaling and tick marks. The plot shows",
                  "the drastic difference in sizes and weights for the Fairy",
                  "and Dragon Pokemon types."]
    print((" ").join(sol_string))

    # data setup
    pokemon = pd.read_csv('./data/pokemon.csv')
    type_cols = ['type_1','type_2']
    non_type_cols = pokemon.columns.difference(type_cols)
    pkmn_types = pokemon.melt(id_vars = non_type_cols, value_vars = type_cols, 
                              var_name = 'type_level', value_name = 'type').dropna()

    pokemon_sub = pkmn_types.loc[pkmn_types['type'].isin(['fairy','dragon'])]

    # plotting
    g = sb.FacetGrid(data = pokemon_sub, hue = 'type', size = 5)
    g.map(plt.scatter, 'weight','height')
    g.set(xscale = 'log') # need to set scaling before customizing ticks
    x_ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
    g.set(xticks = x_ticks, xticklabels = x_ticks)
    g.add_legend()


# Faceting in 2 directions

g = sb.FacetGrid(data = fuel_econ, col = 'VClass', row = 'trans_type', # now also separating by manual, auto transmissions
                margin_titles=True) # this helps cleans up the titles
g.map(plt.scatter, 'displ', 'comb');


# Other adaptations of bivariate plots

# using pointplot
sb.pointplot(data = fuel_econ, x = 'VClass', y = 'comb', hue = 'trans_type',
            ci = 'sd', linestyles="", dodge = True); # (dodge offsets 2 levels slightly)
plt.xticks(rotation = 15)
plt.ylabel('avg comb effec')

# using barplot
sb.barplot(data = fuel_econ, x = 'VClass', y = 'comb', hue = 'trans_type',
            ci = 'sd'); # gets adapted clustered barchart
plt.xticks(rotation = 15)
plt.ylabel('avg comb effec')

# clustered box plot
sb.boxplot(data = fuel_econ, x = 'VClass', y = 'comb', hue = 'trans_type');
plt.xticks(rotation = 15)
plt.ylabel('avg comb effec')

# can also adjust heatmaps to where color is based on mean of 3rd variable

bins_x = np.arange(0.6, 7+0.3, 0.3)
bins_y = np.arange(12, 58+3, 3)
plt.hist2d(data = fuel_econ, x = 'displ, y = 'comb', cmin = 0.5,
            cmap = 'viridis_r', bins = [bins_x, bins_y], weights = co2weights);
            # weights parameter sets how much each datapoint is worth in aggregation
            # by default, each point's weight = 1, so colors reflect total counts
            # if we change this so that each point's weight is equal to its co2 emissions, divided by number of cars in its bin, 
            # then total within each bin will be the avg co2 emissions
plt.colorbar()
plt.xlabel('Displacement')
plt.ylabel('Comb. fuel effec.');

# first step for weights parameter: in which bin does each point fall
# we compute using pandas cut function:
displ_bins = pd.cut(fuel_econ['displ'], bins_x, right = False, include_lowest = False,
                    labels = False.astype(int))
comb_bins = pd.cut(fuel_econ['comb'], bins_y, right = False, include_lowest = False,
                    labels = False.astype(int))
# adding labels = False argument, so that bins are identified numerically

# next, count up number of points that fall in each bin
# this is done using the groupby/size functions
n_points = fuel_econ.groupby([displ_bins, comb_bins]).size()
n_points = n_points.reset_index().pivot(index = 'displ', columns = 'comb').values
# we will use reset index and pivot to set the counts in array form
# finally we use points to get the number of points as an array
# now we can get our point weights using all of these values

co2_weights = fuel_econ['co2'] / n_points[displ_bins, comb_bins]
#divide co2 series by the number of points in each bin by using the cut bins vectors for indexing into the right places

# add weights to weights parameter:
bins_x = np.arange(0.6, 7+0.3, 0.3)
bins_y = np.arange(12, 58+3, 3)
plt.hist2d(data = fuel_econ, x = 'displ, y = 'comb', cmin = 0.5,
            cmap = 'viridis_r', bins = [bins_x, bins_y], weights = co2weights);
plt.colorbar(label = 'co2')
plt.xlabel('Displacement')
plt.ylabel('Comb. fuel effec.');


# Adapted plot practice solution

def adaptedplot_solution_1():
    """
    Solution for Question 1 in adapted plot practice: plot the city vs. highway
    mileage for each vehicle class.
    """
    sol_string = ["Due to overplotting, I've taken a faceting approach to this task.",
                  "There don't seem to be any obvious differences in the main cluster",
                  "across vehicle classes, except that the minicompact and large",
                  "sedans' arcs are thinner than the other classes due to lower",
                  "counts. The faceted plots clearly show that most of the high-efficiency",
                  "cars are in the mid-size and compact car classes."]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')

    sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
    pd_ver = pd.__version__.split(".")
    if (int(pd_ver[0]) > 0) or (int(pd_ver[1]) >= 21): # v0.21 or later
        vclasses = pd.api.types.CategoricalDtype(ordered = True, categories = sedan_classes)
        fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)
    else: # compatibility for v.20
        fuel_econ['VClass'] = fuel_econ['VClass'].astype('category', ordered = True,
                                                         categories = sedan_classes)

    # plotting
    g = sb.FacetGrid(data = fuel_econ, col = 'VClass', size = 3, col_wrap = 3)
    g.map(plt.scatter, 'city', 'highway', alpha = 1/5)


def adaptedplot_solution_2():
    """
    Solution for Question 2 in adapted plot practice: plot the engine size
    distribution against vehicle class and fuel type.
    """
    sol_string = ["I went with a clustered box plot on this task since there were",
                  "too many levels to make a clustered violin plot accessible.",
                  "The plot shows that in each vehicle class, engine sizes were",
                  "larger for premium-fuel cars than regular-fuel cars. Engine size",
                  "generally increased with vehicle class within each fuel type,",
                  "but the trend was noisy for the smallest vehicle classes."]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')

    sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
    pd_ver = pd.__version__.split(".")
    if (int(pd_ver[0]) > 0) or (int(pd_ver[1]) >= 21): # v0.21 or later
        vclasses = pd.api.types.CategoricalDtype(ordered = True, categories = sedan_classes)
        fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)
    else: # compatibility for v.20
        fuel_econ['VClass'] = fuel_econ['VClass'].astype('category', ordered = True,
                                                         categories = sedan_classes)
    fuel_econ_sub = fuel_econ.loc[fuel_econ['fuelType'].isin(['Premium Gasoline', 'Regular Gasoline'])]

    # plotting
    sb.boxplot(data = fuel_econ_sub, x = 'VClass', y = 'displ', hue = 'fuelType')
    plt.legend(loc = 6, bbox_to_anchor = (1.0, 0.5)) # legend to right of figure
    plt.xticks(rotation = 15)


# Plot matrices

# in a plot matrix, a grid of plots is generated
# each subplot in a matrix, is based on the whole data, but with 2 different variables on the axises
# each row/column corresponds with one variable 
# best for initial exploratory analysis

# we want to look at battle statistics for pokemon (6)
pkmn_stats = ['hp', 'attack', 'defense', 'speed', 'special-attack', 'special-defense']
g = sb.PairGrid(data = pokemon, vars - pkmn_stats)
# first set up df on data parameter
# if vars is not specified, all numeric variables will go into the plot
g = g.map_offdiag(plt.scatter)
g.map_diag(plt.hist) # put histograms instead of scatterplots on diagonal
# use off diag for scatterplots so they only plot not on diagonal

# checking correlations in correlations heatmap
sb.heatmap(pokemon[pkmn_stats].corr(), cmap = 'rocket_r', annot = True,
            fmt = '.2f', vmin = 0); # setting -.02 (min) to 0
# reverse color palette using cmap, then add annotations to format values with 2 decimal places


# Feature engineering

# create new variables as functions of existing variables in your data
# will create new variables based on ratios
pokemon['atk_ratio'] = pokemon['attack'] / pokemon['special-attack']
pokemon['def_ratio'] = pokemon['defense'] / pokemon['special-defense']

#basic scatterplot
plt.scatter(data = pokemon, x = 'atk_ratio', y = 'def_ratio', alpha = 1/3) # alpha = transparency
plt.xlabel('Offensive bias, phys/spec')
plt.ylabel('Defensive bias, phys/spec')
plt.xscale('log') #changing axis scales to log scales
# that way, a bias of 2-1 is evenly spaced from the even ratio of 1 in both the physical and special directions
plt.yscale('log')
tick_loc = [0.25, 0.5, 1, 2, 4]
plt.xticks(tick_loc, tick_loc) # next adding tick marks appropriate for log scale
plt.yticks(tick_loc, tick_loc) 
plt.xlim(2 ** -2.5, 2 ** 2.5)
plt.ylim(2 ** -2.5, 2 ** 2.5) # finally, set axis limits to remove extreme outliers
# these also center plot on 1 on both axises


# Additional plot practice solutions
def additionalplot_solution_1():
    """
    Solution for Question 1 in additional plot practice: create a plot matrix
    for five numeric variables in the fuel economy dataset.
    """
    sol_string = ["I set up my PairGrid to plot scatterplots off the diagonal",
                  "and histograms on the diagonal. The intersections where 'co2'",
                  "meets the fuel mileage measures are fairly interesting in how",
                  "tight the curves are. You'll explore this more in the next task."]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')

    # plotting
    g = sb.PairGrid(data = fuel_econ, vars = ['displ', 'co2', 'city', 'highway', 'comb'])
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)


def additionalplot_solution_2():
    """
    Solution for Question 2 in additional plot practice: plot the relationship
    between engine size and emissions in terms of g/gal, for selected fuel
    types.
    """
    sol_string = ["Due to the high number of data points and their high amount of overlap,",
                  "I've chosen to plot the data in a faceted plot. You can see that engine",
                  "sizes are smaller for cars that use regular gasoline against those that",
                  "use premium gas. Most cars fall in an emissions band a bit below 9 kg CO2",
                  "per gallon; diesel cars are consistently higher, a little above 10 kg CO2",
                  "per gallon. This makes sense, since a gallon of gas gets burned no matter",
                  "how efficient the process. More strikingly, there's a smattering of points",
                  "with much smaller emissions. If you inspect these points more closely you'll",
                  "see that they represent hybrid cars that use battery energy in addition to",
                  "conventional fuel! To pull these mechanically out of the dataset requires",
                  "more data than that which was trimmed to create it - and additional research",
                  "to understand why these points don't fit the normal CO2 bands."]
    print((" ").join(sol_string))

    # data setup
    fuel_econ = pd.read_csv('./data/fuel_econ.csv')
    fuel_econ['co2_gal'] = fuel_econ['comb'] * fuel_econ['co2']
    fuel_econ_sub = fuel_econ.loc[fuel_econ['fuelType'].isin(['Premium Gasoline', 'Regular Gasoline', 'Diesel'])]

    # plotting
    g = sb.FacetGrid(data = fuel_econ_sub, col = 'fuelType', size = 4,
                     col_wrap = 3)
    g.map(sb.regplot, 'co2_gal', 'displ', y_jitter = 0.04, fit_reg = False,
          scatter_kws = {'alpha' : 1/5})
    g.set_ylabels('Engine displacement (l)')
    g.set_xlabels('CO2 (g/gal)')
    g.set_titles('{col_name}')


# Summary

# Multivariate techniques
    # shape size color encodings
# faceting: plot multiple simpler plots across levels of 1 or 2 other variables
# heatmaps: can be used for multiple variate visualization, substituting cound for measuring 3rd variable
# plot matrices and correlation heat maps: allow for high level look at pairwise relationships between your variables 
# feature engineering: generate additional features to strike at complex relationships in data