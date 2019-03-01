#Creates sql database to store movie reviews
import sqlite3
import os

if os.path.exists('reviews.sqlite'):
    os.remove('reviews.sqlite')
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()

#create a new database table
c.execute('CREATE TABLE review_db'\


          ' (review TEXT, sentiment INTEGER, date TEXT)')

#adds reviews, date, and sentiment
example1 = 'I love this movie'
c.execute("INSERT INTO review_db"\
          " (review, sentiment, date) VALUES"\
          " (?, ?, DATETIME('now'))", (example1, 1))

example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db"\
          " (review, sentiment, date) VALUES"\
          " (?, ?, DATETIME('now'))", (example2, 0))

#save changes
conn.commit()
conn.close()