import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("movie_dataset.csv")



df["combined_features"] = df['keywords'] +" "+df['cast']+" "+df["genres"]+" "+df["director"]

cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])


cosine_sim = cosine_similarity(count_matrix)

np.save('similarity_matrix',cosine_sim)

df.to_csv('movie_dataset.csv',index=False)
