import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
import warnings
from rank_bm25 import BM25Okapi

warnings.filterwarnings('ignore')

stop_words = stopwords.words('english')
df = pd.read_csv('ted_talks_en.csv')

#Removing unwanted columns, droping rows with null values
df = df[['speaker_1', 'title', 'description', 'transcript', 'topics', 'url']]
df.dropna(inplace=True)


#Processing on text, lowecasing it, removing puntuation, removing stopwords, removing non-alphabetic words
def preprocess_text(text):
    text = text.lower()

    text_no_punctuation = text.translate(str.maketrans('', '', string.punctuation))
    text = text_no_punctuation

    words_without_stop_words = [word for word in text.split() if word not in stop_words]
    text = ' '.join(words_without_stop_words)

    alphabetical_words = [word for word in text.split() if word.isalpha()]
    text = ' '.join(alphabetical_words)
    return text


#Join columns, preprocess, splot, init bm25
df['processed_text'] = df[['title', 'description', 'transcript', 'topics']].apply(lambda x: ' '.join(x), axis=1).apply(preprocess_text)
processed_text_list = df['processed_text'].apply(lambda x: x.split()).tolist()
bm25 = BM25Okapi(processed_text_list)

vectorizer = TfidfVectorizer()
vectorizer.fit(df['processed_text'])
vectorizer.vocabulary_
vectorizer.idf_

# Cosine Similarities / Pearson Correlations / BM25 Scores
def get_similarity_scores(input_text, data):
    input_vector = vectorizer.transform([input_text])

    cosine_similarities_matrix = cosine_similarity(input_vector, vectorizer.transform(data['processed_text']))
    cosine_similarities = cosine_similarities_matrix.flatten()

    pearson_correlations = []
    for text in data['processed_text']:
        transformed_text = vectorizer.transform([text]).toarray().flatten()
        correlation = pearsonr(input_vector.toarray().flatten(), transformed_text)[0]
        pearson_correlations.append(correlation)
    pearson_correlations = np.array(pearson_correlations)

    bm25_scores = bm25.get_scores(input_text.split())
    return cosine_similarities, pearson_correlations, bm25_scores


def recommend_talks(input_text, data, num_recommendations=5):
    cos_sim, pea_sim, bm25_scores = get_similarity_scores(input_text, data)
    data['cos_sim'], data['pea_sim'], data['bm25_scores'] = cos_sim, pea_sim, bm25_scores
    sorted_data = data.sort_values(by=['cos_sim', 'pea_sim', 'bm25_scores'], ascending=[False, False, False])
    recommendations = sorted_data[['speaker_1', 'title', 'url', 'cos_sim', 'pea_sim', 'bm25_scores']].head(num_recommendations)
    return recommendations

def recommend_talksP(input_text, data, num_recommendations=5):
    cos_sim, pea_sim, bm25_scores = get_similarity_scores(input_text, data)
    data['cos_sim'], data['pea_sim'], data['bm25_scores'] = cos_sim, pea_sim, bm25_scores
    sorted_dataP = data.sort_values(by=['pea_sim'], ascending=[False])
    recommendationsP = sorted_dataP[['speaker_1', 'title', 'url','pea_sim']].head(num_recommendations)
    return recommendationsP

def recommend_talksC(input_text, data, num_recommendations=5):
    cos_sim, pea_sim, bm25_scores = get_similarity_scores(input_text, data)
    data['cos_sim'], data['pea_sim'], data['bm25_scores'] = cos_sim, pea_sim, bm25_scores
    sorted_dataC = data.sort_values(by=['cos_sim'], ascending=[False])    
    recommendationC = sorted_dataC[['speaker_1', 'title', 'url', 'cos_sim']].head(num_recommendations)
    return recommendationC

def recommend_talksB(input_text, data, num_recommendations=5):
    cos_sim, pea_sim, bm25_scores = get_similarity_scores(input_text, data)
    data['cos_sim'], data['pea_sim'], data['bm25_scores'] = cos_sim, pea_sim, bm25_scores
    sorted_dataB = data.sort_values(by=['bm25_scores'], ascending=[False])
    recommendationsB = sorted_dataB[['speaker_1', 'title', 'url', 'bm25_scores']].head(num_recommendations)
    return recommendationsB