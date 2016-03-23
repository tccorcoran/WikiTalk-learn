import corpus_utils
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
n_authors = 10
# Load data
print("Loading data...")
x, y = corpus_utils.loadDataSparse(n_authors,return_onehot=False)

# Randomly shuffle data
x_splits,y_splits = corpus_utils.traindevtestSplit(x,y)
x_train, x_dev, x_test = x_splits
y_train, y_dev, y_test = y_splits


def NB():
    clf = MultinomialNB()
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    return f1_score(y_test,pred,average='weighted')