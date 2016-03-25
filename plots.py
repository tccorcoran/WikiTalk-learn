import corpus_utils
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix

def authorship():
    x,y = corpus_utils.loadData(10,return_onehot=False)
    a,b = [],[]
    for i in set(y):
        a.append(i)
        b.append(list(y).count(i))
    plt.bar(a, b, align='center', alpha=0.5)
    plt.xticks(a)
    plt.ylabel('Number of Posts')
    plt.xlabel('Author')
    plt.title('Number of Posts per Author')
    plt.show()
    
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Code adapted from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def show_confusion_matrix():
    """Code adapted from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    with open('results/true.json') as fo:
        y_test = json.load(fo)
    with open('results/pred.json') as fo:
        y_pred = json.load(fo)
        
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    
    plt.show()
show_confusion_matrix()
    