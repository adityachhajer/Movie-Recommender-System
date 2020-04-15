import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_sim():
    df = pd.read_csv('movie_dataset.csv')

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['combined_features'])

    cosine_sim = cosine_similarity(count_matrix)
    return df,cosine_sim


def rcmd(m):
    m = m.lower()
    # try:
    #     df.head()
    #     cosine_sim.shape
    # except:
    df, cosine_sim = create_sim()

    if m not in df['title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:

        i = df.loc[df['title']==m].index[0]

        lst = list(enumerate(cosine_sim[i]))


        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)


        lst = lst[1:11]


        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(df['title'][a])
        return l

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')

if __name__ == '__main__':
    app.run(debug=True)