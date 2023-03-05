import pandas as pd
import re
import googletrans
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

translator = Translator(service_urls=['translate.google.com'])
df = pd.read_excel("politifact.xlsx")
def clean_text(text):
    text = str(text)
    text = text.lower() 
    text = re.sub("[^a-zA-Z ]", "", text) 
    text = re.sub("\s+", " ", text)
    return text
df["statement"] = df["statement"].apply(lambda x: clean_text(x))
df = df.dropna(subset=['veracity'])
df.veracity.fillna(0, inplace=True)
df["veracity"] = df["veracity"].astype(str)
vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
X = vectorizer.fit_transform(df["statement"])

X_train, X_test, y_train, y_test = train_test_split(X, df["veracity"], test_size=0.001, random_state=0)

nb = MultinomialNB()
nb.fit(X_train, y_train)

while True:
    example_statement = input("Enter a statement to classify: ")
    example_statement_en = translator.translate(example_statement, dest='en').text
    
    example_statement_en = vectorizer.transform([clean_text(example_statement_en)])
    
    prediction = nb.predict(example_statement_en)
    print("Prediction:", prediction)
