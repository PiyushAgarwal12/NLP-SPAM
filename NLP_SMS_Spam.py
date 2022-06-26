#Use NLTK for SMS SPAM PREDICTION
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
#nltk.download_shell()       #C:\Users\hp\AppData\Roaming\nltk_data.

#SMS SPAM COLLECTION
messages = [line.rstrip() for line in open('SMSSpamCollection')]

import pandas as pd
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'msg'])
messages.describe()

#Data interpretation
messages.groupby('label').count()
messages.groupby('msg').count().head(5)
messages.groupby('label').describe()

messages['length'] = messages['msg'].apply(lambda x : len(x))

sns.barplot(x='label', y='length', data=messages)
messages[messages['length'] == max(messages['length'])]

#Remove Punctuations
#Remove Stopwords
import string
from nltk.corpus import stopwords
def rem_punc_stopwords(mg):
    punc = string.punctuation
    nopunc = [c for c in mg if c not in punc ]
    nopunc = ''.join(nopunc)
    
    lst_words = nopunc.split()
    c = [word for word in lst_words if word not in stopwords.words('english')]
    return c

messages['rem_punc_stopwords'] = messages['msg'].apply(rem_punc_stopwords)
messages.drop('rem_punc_stopwords', axis=1, inplace=True)


#VECTORIZATION
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=rem_punc_stopwords).fit(messages['msg'])
len(bow_transformer.vocabulary_)

#TEST SOME EXAMPLE MESSAGE
#bow53 = bow_transformer.transform([messages['msg'][53]])
#bow53.shape
#bow_transformer.get_feature_names()[10133]

#SPARSE WHOLE DATA
message_bow = bow_transformer.transform(messages['msg'])
message_bow.nnz             #Number of Non-zeros

#TFIDF - weight the frequency count so that frequent tokens get lower weights
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(message_bow)

#TFID the whole data
message_tfidf = tfidf_transformer.transform(message_bow)
message_tfidf.id


#NOW USE A CLASSIFICATION MODEL
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
spam_ham_classifier = MultinomialNB().fit(message_tfidf, messages['label'])

pred = spam_ham_classifier.predict(message_tfidf)
full_report = classification_report(pred, messages['label'])


#BUILD PIPELINE USING SKLEAR FOR TRAIN-TEST DATA
#NAIVE BIAS
from sklearn.pipeline import Pipeline
pipeline  = Pipeline([
            ('bow', CountVectorizer(analyzer=rem_punc_stopwords)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])

from sklearn.cross_validation import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(messages['msg'], messages['label'],
                                                                test_size=0.4)
pipeline.fit(msg_train, label_train)
all_predict_nv = pipeline.predict(msg_test)

report_nv = classification_report(label_test, all_predit)


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
pipeline_rnf  = Pipeline([
            ('bow', CountVectorizer(analyzer=rem_punc_stopwords)),
            ('tfidf', TfidfTransformer()),
            ('classifier', RandomForestClassifier())
        ])
pipeline_rnf.fit(msg_train, label_train)
all_predict_rnf =  pipeline_rnf.predict(msg_test)
report_rnf = classification_report(label_test, all_predit_rnf)


#KNN CLASSIFICATION
from sklearn.neighbors import KNeighborsClassifier
pipeline_knn  = Pipeline([
            ('bow', CountVectorizer(analyzer=rem_punc_stopwords)),
            ('tfidf', TfidfTransformer()),
            ('classifier', RandomForestClassifier())
        ])

pipeline_knn.fit(msg_train, label_train)
all_predict_knn = pipeline_knn.predict(msg_test)
report_knn = classification_report(label_test, all_predict_knn)



