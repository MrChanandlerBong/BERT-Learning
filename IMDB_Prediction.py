import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 
import re
import string
from sklearn.metrics import accuracy_score

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s!?]', '', text)
    return text


df = pd.read_csv("IMDB Dataset.csv")

df = df.rename(columns={
    'review':'text',
    'sentiment':'label'})

df['clean_text']=df['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X = df['clean_text']
X_raw = X

y = df['label'].map({'positive':0 , 'negative':1})

X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(
    X, y, X_raw,
    test_size=0.3,
    stratify=y,
    random_state=42
)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#model

model = LinearSVC()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

accuracy_svm = accuracy_score(y_test,predictions)

print("Accuracy using SVM :",accuracy_svm)

df_test = pd.DataFrame({
    'text': X_raw_test,
    'true': y_test,
    'pred': predictions
})

errors = df_test[df_test['true'] != df_test['pred']]
errors.sample(100)['text'].values
for text in errors.sample(1)['text']:
    print("----")
    print(text)


