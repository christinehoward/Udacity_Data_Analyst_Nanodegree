# Reading TSV file in pandas
df= pd.read_csv('bestofrt.tsv', sep='\t')

# Webscraping

# Step 1: getting webpage's data
## stored in html (hypertext markup language) format

# downloading html programmatically
import requests
url = 'url'
response = requests.get(url)
# save html to file
with open("... .html", mode='wb') as file:
    file.write(response.content)

# to download all 100 files, need to put this in a loop
from bs4 import BeautifulSoup
soup = BeautifulSoup(response.content, 'lxml')
# in this way, we do not save this information on our computer, but working live with response content in computer's memory
# 'lxml' - can use the beautifulsoup html parser, to help work with response content directly


# HTML files in Python

# BeautifulSoup
# import
from bs4 import BeautifulSoup   
with open('et-html file') as file:
    soup = BeautifulSoup(file, 'lxml') # need to include a parser or we get an error
soup

# let's find the movie's title using the find() method
soup.find('title')
## this results in title of the webpage and not title of the movie

# to get the title only, we will need to do some string slicing
# we can use .contents to return a list of the tag's children

soup.find('title').contents[0][:-len(' - Rotten Tomatoes')]
# this finds everything before the ' -', or 18 characters before the end

# \xa0 unicode for non-breaking space


# Gathering quiz:

from bs4 import BeautifulSoup 
import os 

# looking at title
df_list = []
folder = 'rt_html'
for movie_html in os.listdir(folder):
    with open(os.path.join(folder, movie_html)) as file: # loops through every file in our rthtml folder
        soup = BeautifulSoup(file, 'lxml') # first we need to create the soup, by passing in the file handle
        # should specify lxml parser
        title = soup.find('title').contents[0][:-len(' - Rotten Tomatoes')] # first thing to grab from HTML was title
        # we find the contents in the title, we want the first element in title ([0]), and we want to slice off  - Rotten Tomatoes
        print(title) # print first step
        break # (break the loop)

# looking at title and audience score
df_list = []
folder = 'rt_html'
for movie_html in os.listdir(folder):
    with open(os.path.join(folder, movie_html)) as file:
        soup = BeautifulSoup(file, 'lxml')
        title = soup.find('title').contents[0][:-len(' - Rotten Tomatoes')]
        audience_score = soup.find('div', class_='audience-score meter').find('span').contents[0][:-1] 
        # we find this within a div class titled "audience-score meter"
        # 72% is within the single span tag within the outer most div tag
        # we can use soup.find again, but first need to find the div with class audience-score meter
        # class needs an underscore under it, because class is a reserved keyword in python
        print(audience_score) # let's loop through once, print, and then loop again
        break 
        # we found the audience score, now just need to look in contents, and it is the only item in the span tag so .contents[0]
        # we don't want % sign so we will slice it. we want everything in string except last character, so [:-1] 

# now grabbing number of audience ratings
# first look at html and find where it says user ratings
# outermost div class is "audience-info hidden-xs superPageFontColor"
# let's zoom in on this using BS
df_list = []
folder = 'rt_html'
for movie_html in os.listdir(folder):
    with open(os.path.join(folder, movie_html)) as file:
        soup = BeautifulSoup(file, 'lxml')
        title = soup.find('title').contents[0][:-len(' - Rotten Tomatoes')]
        audience_score = soup.find('div', class_='audience-score meter').find('span').contents[0][:-1] 
        num_audience_ratings = soup.find('div', class_='audience-info hidden-xs superPageFontColor')
        num_audience_ratings = num_audience_ratings.find_all('div')[1].contents[2].strip().replace(',', '')
        print(num_audience_ratings)
        break 
        # by using the print/break strategy, we can more clearly see what is within this tag
        # the user ratings are in the 2nd div tag within the outer div, so lets find all div tags within this outer div, and use the 2nd
        # we want the third item in this div, so .contents[2]
        # there is white space we want to strip out using python strip function .strip()
        # we will need to convert this string to an integer later, and remove comma .replace(',', '')

# now we need to append these to a pandas df
# first we need to import pandas library
df_list = []
folder = 'rt_html'
for movie_html in os.listdir(folder):
    with open(os.path.join(folder, movie_html)) as file:
        soup = BeautifulSoup(file, 'lxml')
        title = soup.find('title').contents[0][:-len(' - Rotten Tomatoes')]
        audience_score = soup.find('div', class_='audience-score meter').find('span').contents[0][:-1] 
        num_audience_ratings = soup.find('div', class_='audience-info hidden-xs superPageFontColor')
        num_audience_ratings = num_audience_ratings.find_all('div')[1].contents[2].strip().replace(',', '')
        df_list.append({'title': title
                        'audience_score': int(audience_score, # converting string to integer
                        'number_of_audience_ratings': int(num_audience_ratings)})  # converting string to integer
df = pd.DataFrame(df_list, columns = ['title', 'audience_score', 'number_of_audience_ratings'])
# let's process this cell, which may take some time to run
# great, no errors
# let's look at this dataframe

# Solution Test
# Run the cell below the see if your solution is correct. If an AssertionError is thrown, your solution is incorrect. If no error is thrown, your solution is correct.

df_solution = pd.read_pickle('df_solution.pkl')
df.sort_values('title', inplace = True)
df.reset_index(inplace = True, drop = True)
df_solution.sort_values('title', inplace = True)
df_solution.reset_index(inplace = True, drop = True)
pd.testing.assert_frame_equal(df, df_solution)


