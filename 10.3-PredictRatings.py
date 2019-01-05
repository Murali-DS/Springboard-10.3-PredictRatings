# -*- coding: utf-8 -*-

#%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from six.moves import range

# Setup Pandas
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

# Setup Seaborn
sns.set_style("whitegrid")
sns.set_context("poster")

critics = pd.read_csv('./critics.csv')
#let's drop rows with missing quotes
critics = critics[~critics.quote.isnull()]
critics.head()

n_reviews = len(critics)
n_movies = critics.rtid.unique().size
n_critics = critics.critic.unique().size


print("Number of reviews: {:d}".format(n_reviews))
print("Number of critics: {:d}".format(n_critics))
print("Number of movies:  {:d}".format(n_movies))

df = critics.copy()
df['fresh'] = df.fresh == 'fresh'
grp = df.groupby('critic')

counts = grp.critic.count()  # number of reviews by each critic
means = grp.fresh.mean()     # average freshness for each critic

means[counts > 100].hist(bins=10, edgecolor='w', lw=1)
plt.xlabel("Average Rating per critic")
plt.ylabel("Number of Critics")
plt.yticks([0, 2, 4, 6, 8, 10]);

from sklearn.feature_extraction.text import CountVectorizer

text = ['Hop on pop', 'Hop off pop', 'Hop Hop hop']
print("Original text is\n{}".format('\n'.join(text)))

vectorizer = CountVectorizer(min_df=0)

# call `fit` to build the vocabulary
vectorizer.fit(text)

# call `transform` to convert text to a bag of words
x = vectorizer.transform(text)

# CountVectorizer uses a sparse array to save memory, but it's easier 
# in this assignment to 
# convert back to a "normal" numpy array
x = x.toarray()

print("")
print("Transformed text vector is \n{}".format(x))

# `get_feature_names` tracks which word is associated with each 
# column of the transformed x
print("")
print("Words for each feature:")
print(vectorizer.get_feature_names())

# Notice that the bag of words treatment doesn't preserve information 
# about the *order* of words, 
# just their frequency

def make_xy(critics, vectorizer=None):
    #Your code here    
    if vectorizer is None:
        vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(critics.quote)
    X = X.tocsc()  # some versions of sklearn return COO format
    y = (critics.fresh == 'fresh').values.astype(np.int)
    return X, y
X, y = make_xy(critics)


#your turn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from IPython.display import display

def display_matrix(data, col_names, index_names):
    """Converts Matrix into data frame and display in nice format."""
    df_to_display = \
    pd.DataFrame(data, index=index_names, columns=col_names)
    display(df_to_display)


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,random_state=5)
clf = MultinomialNB().fit(Xtrain, ytrain)
training_accuracy = clf.score(Xtrain, ytrain)
ypred = clf.predict(Xtest)
test_accuracy = clf.score(Xtest, ytest)
print('Training Accuracy {0}%'.format(round(training_accuracy*100, 2)))
print('Testing Accuracy {0}%'.format(round(test_accuracy*100,2)))

confusion_matrix_results = confusion_matrix(ytest, ypred)
cols = ['Actual Positive', 'Actual Negative']
inds = ['Predicted Positive', 'Predicted Negative']
display(display_matrix(confusion_matrix_results, cols, inds))

# Your turn.
import matplotlib.pyplot as plt
import seaborn as sns

def ecdf(data):
    #"""Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n  # y-data for the ECDF: y
    return x, y

def make_ecdf_plot(data, x_label, y_label, fig_no, x_lim):
    x, y = ecdf(data)
    plt.figure(fig_no)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.xlim(x_lim)
    plt.margins(0.02)
    plt.show()

X_array = X.toarray()
print('There are {0} words. These are being monitored across {1} documents.'.format(X_array.shape[0], X_array.shape[1]))
X_df = np.sort(X_array.sum(axis=0))
make_ecdf_plot(X_df, '% of Words that appear in fewer than n Documents', 'Percentage', 1, [0, 100])

from sklearn.model_selection import KFold
def cv_score(clf, X, y, scorefunc):
    result = 0.
    nfold = 5
    for train, test in KFold(nfold).split(X): 
        # split data into train/test groups, 5 times
        clf.fit(X[train], y[train]) 
        # fit the classifier, passed is as clf.
        result += scorefunc(clf, X[test], y[test]) 
        # evaluate score function on held-out data
    return result / nfold # average

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    rotten = y == 0
    fresh = ~rotten
    return prob[rotten, 0].sum() + prob[fresh, 1].sum()

from sklearn.model_selection import train_test_split
_, itest = train_test_split(range(critics.shape[0]), train_size=0.7)
mask = np.zeros(critics.shape[0], dtype=np.bool)
mask[itest] = True

from sklearn.naive_bayes import MultinomialNB

#the grid of parameters to search over
alphas = [.1, 1, 5, 10, 50]
scores = np.empty(5)
best_min_df = 1 # YOUR TURN: put your value of min_df here.

#Find the best value for alpha and min_df, and the best classifier
best_alpha = None
maxscore=-np.inf
count = 0
for alpha in alphas:
    vectorizer = CountVectorizer(min_df=best_min_df)       
    Xthis, ythis = make_xy(critics, vectorizer)
    Xtrainthis = Xthis[mask]
    ytrainthis = ythis[mask]
    # your turn
    clf = MultinomialNB(alpha=alpha).fit(Xtrainthis, ytrainthis)
    scores[count]=cv_score(clf, Xthis, ythis, log_likelihood)
    count = count+1

best_alpha = alphas[np.argmax(scores)]
print("alpha: {}".format(best_alpha))

vectorizer = CountVectorizer(min_df=best_min_df)
X, y = make_xy(critics, vectorizer)
xtrain=X[mask]
ytrain=y[mask]
xtest=X[~mask]
ytest=y[~mask]

clf = MultinomialNB(alpha=best_alpha).fit(xtrain, ytrain)

#your turn. Print the accuracy on the test and training dataset
training_accuracy = clf.score(xtrain, ytrain)
test_accuracy = clf.score(xtest, ytest)

print("Accuracy on training data: {:2f}".format(training_accuracy))
print("Accuracy on test data:     {:2f}".format(test_accuracy))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, clf.predict(xtest)))

words = np.array(vectorizer.get_feature_names())

x = np.eye(xtest.shape[1])
probs = clf.predict_log_proba(x)[:, 0]
ind = np.argsort(probs)

good_words = words[ind[:10]]
bad_words = words[ind[-10:]]

good_prob = probs[ind[:10]]
bad_prob = probs[ind[-10:]]

print("Good words\t     P(fresh | word)")
for w, p in zip(good_words, good_prob):
    print("{:>20}".format(w), "{:.2f}".format(1 - np.exp(p)))
    
print("Bad words\t     P(fresh | word)")
for w, p in zip(bad_words, bad_prob):
    print("{:>20}".format(w), "{:.2f}".format(1 - np.exp(p)))
    
    
x, y = make_xy(critics, vectorizer)

prob = clf.predict_proba(x)[:, 0] # Probability that sample belongs to 
# the first class, i.e Class 0 or Rotten.
predict = clf.predict(x)

temp = np.transpose(np.vstack([y, predict, prob, 1-np.exp(prob)]))

bad_rotten = np.argsort(prob[y == 0])[:5]
bad_fresh = np.argsort(prob[y == 1])[-5:]

print("Mis-predicted Rotten quotes")
print('---------------------------')
for row in bad_rotten:
    print(critics[y == 0].quote.iloc[row])
    print("")

print("Mis-predicted Fresh quotes")
print('--------------------------')
for row in bad_fresh:
    print(critics[y == 1].quote.iloc[row])
    print("")
    

#your turn
text = ['This movie is not remarkable, touching, or superb in any way']
print("Original text is\n{}".format('\n'.join(text)))

#vectorizer = CountVectorizer(min_df=1)

# call `fit` to build the vocabulary
#vectorizer.fit(text)

# call `transform` to convert text to a bag of words
#x_sample = vectorizer.transform(text)

# CountVectorizer uses a sparse array to save memory, but it's easier in this assignment to 
# convert back to a "normal" numpy array
#x_sample = x_sample.toarray()
x_sample = vectorizer.fit_transform(text)
x_sample = x_sample.tocsc()  # some versions of sklearn return COO format
#x_sample_1 = x_sample.toarray()


predict_sample = clf.predict(x_sample)