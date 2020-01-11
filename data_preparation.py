from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import spacy
from spacy_langdetect import LanguageDetector


def purge_data(df):
    df = df.dropna(axis=0, how='any')
    nlp = spacy.load('en')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    not_en_idxs = []
    for idx, lyrics in enumerate(df['lyrics']):
        text = lyrics[:100]
        doc = nlp(text)
        if doc._.language['language'] != 'en':
            not_en_idxs.append(idx)
    df = df.reset_index()
    df = df.drop(not_en_idxs,axis=0)
    df.drop('index', axis=1, inplace= True)

def train_test_validation_split(df):
    df = shuffle(df)
    train, test = train_test_split(df, test_size=0.3)
    test, val = train_test_split(test, test_size=0.5)
    train.to_csv('train_test_val_raw/train_lyrics.csv', index=False)
    test.to_csv('train_test_val_raw/test_lyrics.csv', index=False)
    val.to_csv('train_test_val_raw/validation_lyrics.csv', index=False)

def to_lower(df):
    df['song'] = df['song'].str.lower()
    df['lyrics'] = df['lyrics'].str.lower()
    df['artist'] = df['artist'].str.lower()

def dash_to_space(df):
    df['song'] = df['song'].apply(lambda x: x.replace('-'," "))
    df['artist'] = df['artist'].apply(lambda x: x.replace('-'," "))

def delete_punctuation(df):
    def remove_punctuation(text):
        return "".join([c for c in text if c not in string.punctuation])
    df['song'] = df['song'].apply(lambda x : remove_punctuation(x))
    df['lyrics'] = df['lyrics'].apply(lambda x : remove_punctuation(x))
    df['artist'] = df['artist'].apply(lambda x : remove_punctuation(x))

def tokenize(df):
    tokenizer = RegexpTokenizer(r'\w+')
    df['song'] = df['song'].apply(lambda x : tokenizer.tokenize(x))
    df['lyrics'] = df['lyrics'].apply(lambda x : tokenizer.tokenize(x))
    df['artist'] = df['artist'].apply(lambda x : tokenizer.tokenize(x))

def stop_words(df):
    def rem(text):
        return [w for w in text if w not in stopwords.words('english')]
    df['lyrics'] = df['lyrics'].apply(lambda x : rem(x))
    df['song'] = df['song'].apply(lambda x: rem(x))
    df['artist'] = df['artist'].apply(lambda x: rem(x))

def lemming(df):
    def w_lem(text):
        lemm = WordNetLemmatizer()
        return  [lemm.lemmatize(i) for i in text]
    df['lyrics'] = df['lyrics'].apply(lambda x : w_lem(x))
    df['song'] = df['song'].apply(lambda x: w_lem(x))
    df['artist'] = df['artist'].apply(lambda x: w_lem(x))

def stemming(df):
    def w_stem(text):
        return " ".join([stemmer.stem(i) for i in text])
    stemmer = PorterStemmer()
    df['lyrics'] = df['lyrics'].apply(lambda x : w_stem(x))
    df['song'] = df['song'].apply(lambda x: w_stem(x))
    df['artist'] = df['artist'].apply(lambda x: w_stem(x))


def create_bag_of_words(df):
    count = CountVectorizer(dtype=np.int8, max_features=1000)
    text_data = df['lyrics'].values
    bag_of_words = count.fit_transform(text_data)
    arr = bag_of_words.toarray()
    return pd.DataFrame(arr)

def create_tf_idf(df):
    count = TfidfVectorizer(max_features=1000)
    text_data = df['lyrics'].values
    bag_of_words = count.fit_transform(text_data)
    arr = bag_of_words.toarray()
    return pd.DataFrame(arr)

def create_datasets():
    bow = pd.read_csv('bow1000_2.csv')
    tf_idf = pd.read_csv('tf_idf1000_2.csv')
    bow = bow[['genre',*[str(i) for i in range(1000)]]]
    tf_idf = tf_idf[['genre',*[str(i) for i in range(1000)]]]
    bow_lyrics = bow[[*[str(i) for i in range(1000)]]]
    tf_idf_lyrics = tf_idf[[*[str(i) for i in range(1000)]]]
    data_norm_bow = normalize(bow_lyrics.values)
    data_norm_tf_idf = normalize(tf_idf_lyrics.values)

    pca_bow_10 = PCA(n_components=10)
    pca_tf_idf_10 = PCA(n_components=10)
    pca_bow_100 = PCA(n_components=100)
    pca_tf_idf_100 = PCA(n_components=100)

    pca_bow_10X_transform = pca_bow_10.fit_transform(data_norm_bow)
    pca_tf_idf_10X_transform = pca_tf_idf_10.fit_transform(data_norm_tf_idf)
    pca_bow_100X_transform = pca_bow_100.fit_transform(data_norm_bow)
    pca_tf_idf_100X_transform = pca_tf_idf_100.fit_transform(data_norm_tf_idf)

    df1 = pd.DataFrame(data_norm_bow)
    df2 = pd.DataFrame(data_norm_tf_idf)
    df3 = pd.DataFrame(pca_bow_10X_transform)
    df4 = pd.DataFrame(pca_tf_idf_10X_transform)
    df5 = pd.DataFrame(pca_bow_100X_transform)
    df6 = pd.DataFrame(pca_tf_idf_100X_transform)
    genre = pd.DataFrame(bow.genre.values, columns=['genre'])
    pd.concat([df1, genre],axis=1).to_csv('pca_bow_tf-idf/bow.csv', index=False)
    pd.concat([df2, genre],axis=1).to_csv('pca_bow_tf-idf/tf_idf.csv', index=False)
    pd.concat([df3, genre],axis=1).to_csv('pca_bow_tf-idf/bow_pca10.csv', index=False)
    pd.concat([df4, genre],axis=1).to_csv('pca_bow_tf-idf/tf_idf_pca10.csv', index=False)
    pd.concat([df5, genre],axis=1).to_csv('pca_bow_tf-idf/bow_pca100.csv', index=False)
    pd.concat([df6, genre],axis=1).to_csv('pca_bow_tf-idf/tf_idf_pca100.csv', index=False)

def load_mlp(file_name, out_file_name):
    df = pd.read_csv(file_name)
    l = list(set(df['genre'].values))
    df['genre'] = df['genre'].apply(lambda x : l.index(x))
    df = shuffle(df)
    train, test = train_test_split(df, test_size=0.3)
    test, val = train_test_split(test, test_size=0.5)
    train.to_csv('{:}_train.csv'.format(out_file_name), index=False)
    test.to_csv('{:}_test.csv'.format(out_file_name), index=False)
    val.to_csv('{:}_val.csv'.format(out_file_name), index=False)




#df = pd.read_csv('lyrics_cleaned.csv')
#to_lower(df)
#dash_to_space(df)
#delete_punctuation(df)
#tokenize(df)
#stop_words(df)
#stemming(df)
#df = pd.read_csv('lyrics_pre_processed2.csv')
#df_bow = create_bag_of_words(df)
#df_tf_idf = create_tf_idf(df)
#load_mlp("pca_bow_tf-idf/bow_pca10.csv","models/data/bow_pca10")