import corpus_utils
import numpy as np
import matplotlib.pyplot as plt

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
    