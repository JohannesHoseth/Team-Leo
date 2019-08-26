# Predicting Academy Award winners in the category 'Best Picture'

This project scrapes IMDB for all Academy Award winning and nominated movies in the category *Best Picture*, with the intention of using
machine learning methods to predict the next winning movie. For each movie a series of data is collected (see section: *dataset*) for the movie, director(s) and actors who worked on it. 

## To do:

- [x] Write final functions for scraping the IMDB sites
- [ ] Add budget and box office numbers to dataset
- [ ] Re-scrape the IMDB sites for updated content
- [ ] Tidy data
- [ ] Apply machine learning algorithm to dataset
- [ ] Analyze the logfiles
- [ ] ***Add more here***

## dataset

The dataset [oscar_movies.csv](oscar_movies.csv) include all variables scraped from IMDB, this includes:

['title', 'year', 'runtime_min', 'genre', 'metascore', 'gross_mil',
'link_movie', 'director', 'actors', 'link_people', 'nom_people_sum', 'won_people_sum',
'country', 'language', 'release_date', 'budget', 'color', 'aspect_ratio',
'won_oscar', 'Adventure', 'Drama', 'War', 'Crime', 'Biography', 'Music',
'History', 'Romance', 'Mystery', 'Western', 'Comedy', 'Action',
'Sci-Fi', 'Horror', 'Thriller', 'Fantasy', 'Musical', 'Family', 'Sport',
'Animation', 'Film-Noir', 'Documentary']

The column *Nominated* expresses the sum of oscar nominations, for the director and the main cast, where the column *Won* expresses the sum of oscar winnings for the same.

The column *won_oscar* expresses weather or not the movie won an oscar for 'Best Picture'.

## How to use:

The [oscar_scraper.ipynb](oscar_scraper.ipynb) notebook includes all functions required to scrape the sites/subsites. There are three main functions:
- `get_movies` (scrapes the list of movies)
- `get_awards` (scrapes the number of acadamy awards each director/actor has been nominated for and won)
- `get_metadata` (scrapes extra data for each movie)

#### get_movies:
This function accepts one of two inputs, *'winners'* or *'nominees'*, which will determine which site to scrape. The *'winners'* input scrapes [this](https://www.imdb.com/search/title/?count=100&groups=oscar_best_picture_winners&sort=year,desc&ref_=nv_ch_osc) site, whereas the *'nominees'* scrapes [this](db.com/search/title/?groups=oscar_best_picture_nominees&start=1&ref_=adv_nxt) and subsequent pages. 

```
get_movies(winners_or_nominees)
```

The output is a dataframe containing following columns:

['index', 'title', 'year', 'runtime_min', 'genre', 'metascore', 'gross_mil', 'link_movie', 'director', 'actors', 'link_people']

The *index* referes to the index of the IMDB page. The two link columns, *link_movie* and *link_people*, include links to the individual movies site on IMDB, and to each of the director(s)/actors awards page. The columns *genre*, *director*, *actors* and *link_people* are all lists.

#### get_awards:
This function accepts one input, a list of urls for director/actor IMDB pages. The list comes from the column *link_people*, from the output of the `get_movies` function. This outputs a list with tuples, that has the sum of nominations and winnings for all the directors/actors of each movie.

```
get_awards(actorlist)
```

#### get_metadata:
This functions accepts one input, a url to the IMDB movie page. The url comes from the columns *link_movie*, from the output of the `get_movies` function. This outputs a dataframe with following columns:

['country', 'language', 'release_date', 'budget', 'color', 'aspect_ratio']

```
get_metadata(movie_url)
```

### Put together

```
movies   = get_movies('nominees')
awards   = [get_awards(i) for i in movies.link_people]
metadata = [get_metadata(i) for i in movies.link_movie]

awards   = pd.DataFrame(awards, columns=['nom_people_sum', 'won_people_sum'])
metadata = pd.DataFrame(metadata, columns = ['country', 'language', 'release_date', 'budget', 'color', 'aspect_ratio'])

df       = movies.merge(awards, left_index = True, right_index = True)
df       = df.merge(metadata, left_index = True, right_index = True)

df.to_csv('oscar_movies.csv')
```

## Contributers:
@JohannesHoseth
@Benny-ucph
@akaisin
@bijantaheri
