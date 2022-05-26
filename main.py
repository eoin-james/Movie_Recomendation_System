import os
import pandas as pd
import numpy as np

from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel


def create_model():
    # Convert data to a matrix of TF-IDF features
    # TF IDF states how important a word is to a document in a corpus
    return TfidfVectorizer(
        min_df=3,  # For the vocab ignore terms that have doc freq less than
        max_features=None,
        strip_accents='unicode',  # Character normalisation
        analyzer='word',  # Word n grams
        token_pattern=r'\w{1,}',  # What constitutes a token
        ngram_range=(1, 3),  # Uni/Bi/Tri grams
        stop_words='english'  # Ignore words
    )


def create_data(credits_path, movies_path):
    # Data as Dataframes
    credits_data = pd.read_csv(credits_path)  # (4803, 4)
    movies_data = pd.read_csv(movies_path)  # (4803, 20)

    # Merge Data to have one input data set
    credits_column_renamed = credits_data.rename(index=str, columns={"movie_id": "id"})
    movies_data_merge = movies_data.merge(credits_column_renamed, on='id')

    # Drop unwanted data
    cleaned = movies_data_merge.drop(
        columns=[
            'homepage',
            'title_x',
            'title_y',
            'status',
            'production_countries'
        ]
    )

    # Drop NAN as description can really be replaced
    return cleaned.dropna()


def give_recommendations(title, sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movie_data['original_title'].iloc[movie_indices]


if __name__ == '__main__':
    # Data paths as string variables
    credits_data_path = 'Data/tmdb_5000_credits.csv'
    movies_data_path = 'Data/tmdb_5000_movies.csv'

    # Create movies data
    movie_data = create_data(credits_data_path, movies_data_path)

    # Create the TF IDF Model
    tf_idf_vector = create_model()

    # Fit the model to the data
    tf_idf_matrix = tf_idf_vector.fit_transform(movie_data['overview'])

    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tf_idf_matrix, tf_idf_matrix)

    # Reverse mapping of indices and movie titles
    indices = pd.Series(movie_data.index, index=movie_data['original_title']).drop_duplicates()

    # Pick a movie to get 10 recommendations from
    movie = 'The Matrix'
    print(give_recommendations(movie, sig))
