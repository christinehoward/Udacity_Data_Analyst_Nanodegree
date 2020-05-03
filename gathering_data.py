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


# Now we can create the word cloud of Roger Ebert's reviews

# First, downloading files from the internet using Python's requests (HTTP for Humans)
# we will use requests.get to get these files

import requests # import requests library
import os # also import the os library too, see we can store the downloaded file in a folder called ebert reviews

folder_name = 'ebert_reviews'
if not os.path.exists(folder_name):
    os.makedirs(folder_name) # this creates folder if it does not already exist

url = 'https://...' # rogert ebert review text file stored on Udacity servers
response = requests.get(url) # we use requests.get on a url and that returns a response
# response # what does response variable look like?
# output: <Response [200]>, this is the http status code for the request has succeeded
# text from text file is currently in our computer's working memory
# it's stored in the body of the response which we can access using .content
response.content
# output: in bytes format, with review text 
# we are now going to save this file to our computer
# we want to open a file, by accessing everything after the last slash in the url before .txt
with open(os.path.join(folder_name, 
                        url.split('/')[-1]), mode='wb') as file: # select last item in the list returned 
    file.write(response.content)
# we need to open this file, which will then write the contents of the response variable too
# we need to open this in wb mode, write binary (mode='wb')
# that's because response.content is in bites, and not text
# then we write to the file handle we have opened: file.write(response.content)

# ^^ That's how you download 1 file programmatically

# Let's check contents of our folder ebert reviews, to make sure the file is there
os.listdir(folder_name)
# output: ['.DS_Store', '11-e.t.-the-extra-terrestrial.txt']
# .DS_Store is a hidden file that stores the attributes of our folder


# Quiz: downloading multiple files from the internet
# In the Jupyter Notebook below, programmatically download all of the Roger Ebert review text files to a folder called ebert_reviews using 
# the Requests library. Use a for loop in conjunction with the provided ebert_review_urls list.

import requests 
import os 

ebert_review_urls = ['many urls here']

folder_name = 'ebert_reviews'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for url in ebert_review_urls: # then we will get the http response via request.get, on whatever iteration we are currently on in that loop
    response = requests.get(url)
    with open(os.path.join(folder_name, url.split('/')[-1]), mode = 'wb') as file: # then the bit of code to open a file and write a response.content to that file is the same as above
        file.write(response.content) # we will process the cell, took 5 secs

os.listdir(folder_name) # check contents of the folder, there should be 88

len(os.listdir(folder_name)) # check if there are 88
# 12 movies on the top 100 list did not have reviews on roger ebert's site

# Solution Test
import filecmp 
dc = filecmp.dircmp('ebert_reviews', 'ebert_reviews_solution')
assert len(dc.common) == 88


# Gathering data from text files

# we have 88 roger ebert reviews to open and read
# we will need a loop to iterate through all of the files in this folder to open and read each
# we can use a library called os and a library called glob

import os 
folder = 'ebert_reviews'
for ebert_review in os.listdir(folder):
    with open(os.path.join(folder, ebert_review)) as file:
# we have been using os's listdir, which is good if you are sure you want to open every file in the folder
# but let's try using glob instead: allows us to use something called glob patterns to specify sets of file names
# these glob patterns use something called wildcard characters
# focusing on glob.glob, which returns a list of pathnames that match pathname, i.e. string parameter we passed.

# we want all file names ending in txt
import glob 
# glob.glob returns a list which we can loop through directly.
for ebert_review in glob.glob('ebert_reviews/*.txt'): # every file in eber_reviews folder, then every file ending in .txt
    # * = wildcard for the glob pattern, match any string of any length
    print(ebert_review) # prints paths to all files

# we can pass this into the open function in python
import glob 
for ebert_review in glob.glob('ebert_reviews/*.txt'):
    with open(ebert_review, encoding='utf-8') as file: 
        print(file.read()) # we would get all text in 1 big chunk
        break
        # should include encoding. doing so means you get correctly decoded unicode, or an error right away, making this easy to debug
        # encoding depends on source of the text, we can inspect source of webpage (write click, view page source) and find encoding is utf-8 (meta charset)

# but we want everything in the first line (title), second line (link), then everything afterwards as separate pieces of data
import glob 
for ebert_review in glob.glob('ebert_reviews/*.txt'):
    with open(ebert_review, encoding='utf-8') as file:
        print(file.readline()[:-1]) # read 1 line, slicing off blank space
        break # this gives us just the 1st line of the 1st file, including whitespace (/n, new line character)

# now we want to grab url and full review text
import glob 
import pandas as pd 
df_list = []
for ebert_review in glob.glob('ebert_reviews/*.txt'):
    with open(ebert_review, encoding='utf-8') as file: 
        title = file.readline()[:-1] # 1st line minus whitespace = title
        review_url = file.readline()[:-1]
        review_text = file.read() # readlines throws an error when checking the solution, so we use read
        # we want to add this into a pandas dataframe, which we can achieve by first creating an empty list, then populate list 1 by 1 as we iterate through the for loop
        df_list.append({'title': title, 
                        'review_url': review_url,
                        'review_text': review_text}) # we will fill this list with dictionaries and this list of dictionaries will later be converted to a pandas df
df = pd.DataFrame(df_list, columns=['title', 'review_url', 'review_text'])

# Solution Test
# Run the cell below the see if your solution is correct. If an AssertionError is thrown, your solution is incorrect. If no error is thrown, your solution is correct.

df_solution = pd.read_pickle('df_solution.pkl')
df.sort_values('title', inplace = True)
df.reset_index(inplace = True, drop = True)
df_solution.sort_values('title', inplace = True)
df_solution.reset_index(inplace = True, drop = True)
pd.testing.assert_frame_equal(df, df_solution)
df_solution = pd.read_pickle('df_solution.pkl')
df.sort_values('title', inplace = True)
df.reset_index(inplace = True, drop = True)
df_solution.sort_values('title', inplace = True)
df_solution.reset_index(inplace = True, drop = True)
pd.testing.assert_frame_equal(df, df_solution)


# Source - APIs

# now getting each movie's poster to form our word cloud
# can scrape image url from the html, but a better way to access is using API
# API: application programming interface
# since each movie has its poster on Wikipedia movie page, can use Wikipedia API

# Rotten Tomatoes API provides audience scores, we could have hit the API instead of scraping off webpage
# does not provide posters, images, but we would need to apply for usage
# always choose API over scraping when available, scraping is brittle, can break when html changes

# example using rt api
import rtsimple as rt 
rt.API_KEY = 'YOUR API KEY HERE' # we only have access to this once RT approves our proposal
movie - rt.Movies('10489') # movie id
movie.ratings['audience_score'] # then we access the ratings 

# because we do not have access to the RT API, we will use MediaWiki, which hosts Wikipedia data


# Quiz
# In the Jupyter Notebook below, get the page object for the E.T. The Extra-Terrestial Wikipedia page. Here is the E.T. Wikipedia page for easy reference.
import wptools
# Your code here: get the E.T. page object
# This cell make take a few seconds to run
page = wptools.page('E.T._the_Extra-Terrestrial').get()
# Accessing the image attribute will return the images for this page
page.data['image']

