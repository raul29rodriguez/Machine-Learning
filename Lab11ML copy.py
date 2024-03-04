'''Exercise 1
Given the following data set: book_archive.zip, develop a Recommender System based on a
given title a customer has purchased'''
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

r_cols = ['user_id', 'book_id', 'rating']
ratings = pd.read_csv('Ratings.csv')
m_cols = ['book_id', 'title']
books = pd.read_csv('Books.csv')
ratings = pd.merge(books, ratings)
ratings.head()
bookRatings =ratings.pivot_table(index=['User_ID'],columns=['title'],values='Book-Rating')
bookRatings.head()
summerlandRatings = bookRatings['Summerland']
summerlandRatings.head()
similarBooks = bookRatings.corrwith(summerlandRatings)
similarBooks = similarBooks.dropna()
bookStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
bookStats.head()
popularBooks = bookStats['rating']['size'] >= 100
bookStats[popularBooks].sort_values([('rating', 'mean')], ascending=False)[:15]
df = bookStats[popularBooks].join(pd.DataFrame(similarBooks, columns=['similarity']))
df.head()
df.sort_values(['similarity'], ascending=False)[:15]