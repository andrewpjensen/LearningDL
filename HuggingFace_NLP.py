import pandas as pd
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

# Script copied from here
reviews = []
for fileid in movie_reviews.fileids():
    tag, filename = fileid.split('/')
    reviews.append((tag, movie_reviews.raw(fileid)))
sample = pd.DataFrame(reviews, columns=['target', 'document'])
print(f'Dimensions: {sample.shape}')
print(sample.head(), '\n')
print(sample['target'].value_counts(), '\n')

X_train, X_test, y_train, y_test = train_test_split(sample['document'], sample['target'], test_size=0.3, random_state=123)
print(f'Train dimensions: {X_train.shape, y_train.shape}')
print(f'Test dimensions: {X_test.shape, y_test.shape}')
# Check out target distribution
print(y_train.value_counts())
print(y_test.value_counts())

def preprocess_text(text):
    # Tokenise words while ignoring punctuation
    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(text)
    
    # Lowercase and lemmatise 
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
    
    # Remove stop words
    keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    return keywords

# Create an instance of TfidfVectorizer
vectoriser = TfidfVectorizer(analyzer=preprocess_text)
# Fit to the data and transform to feature matrix
X_train_tfidf = vectoriser.fit_transform(X_train)
X_train_tfidf.shape

sgd_clf = SGDClassifier(random_state=123)
sgf_clf_scores = cross_val_score(sgd_clf, X_train_tfidf, y_train, cv=5)
print(sgf_clf_scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (sgf_clf_scores.mean(), sgf_clf_scores.std() * 2))

cross_val_score(sgd_clf, X_train_tfidf, y_train, cv=5, scoring='accuracy')
sgf_clf_pred = cross_val_predict(sgd_clf, X_train_tfidf, y_train, cv=5)
print(confusion_matrix(y_train, sgf_clf_pred))
