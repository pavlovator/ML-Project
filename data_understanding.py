import spacy
from spacy_langdetect import LanguageDete
def genre_bar(df):
    df['genre'].value_counts().plot(kind='bar')

def year_bar(df):
    df['genre'].value_counts().plot(kind='bar')


#df = pd.read_csv('lyrics_cleaned.csv')
#df = pd.read_csv('bow1000.csv')