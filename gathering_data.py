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
