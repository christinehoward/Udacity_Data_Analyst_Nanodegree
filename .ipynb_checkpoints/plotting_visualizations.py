#Plotting with Pandas

import pandas as pd
% matplotlib inline

df_census = pd.read_csv('census_income_data.csv')
df_census.info()
    ##shows multiple histograms of quantitative data

df_census.hist(figsize=(8, 8));
    ##increases size of the above

df_census['age'].hist();
    ##plot histogram for specific column

df_census['age'].plot(kind='hist');
    ##plot using histogram as plot type
    
df_census['education'].value_counts()
    ##aggregates counts for unique values in column 

df_census['education'].value_counts().plot(kind='bar');
    ##bar chart for above education data

df_census['workclass'].value_counts().plot(kind='pie', figsize=(8,8))
    ##pie chart also requires value_counts

df_cancer = pd.read_csv('cancer_data_edited.csv')
df_cancer.info()

pd.plotting.scatter_matrix(df_cancer, figsize=(15, 15));
    ##scatterplots/1 historgram comparing relationships among all numerical variables

df_cancer.plot(x='compactness', y='concavity', kind='scatter') ;
    ##single scatterplot

df_cancer['concave_points'].plot(kind='box', figsize=(15,15));
    ##create box plot


#Exploring data and visuals quiz

# import and load data
import pandas as pd
%matplotlib inline
df_ppt = pd.read_csv('powerplant_data_edited.csv')
df_ppt.info()

# plot relationship between temperature and electrical output
df_ppt.plot(x='Temperature', y='Net hourly electrical energy output',kind='scatter');

# plot distribution of humidity
df_ppt['Relative Humidity'].hist();

# plot box plots for each variable

df_ppt.plot(kind='box', figsize=(15,15));
    ##all box plots on one graph

df['temperature'].plot(kind='box');
df['exhaust_vacuum'].plot(kind='box');
df['pressure'].plot(kind='box');
df['humidity'].plot(kind='box');
df['energy_output'].plot(kind='box');
    ##separate box plats for each variable



