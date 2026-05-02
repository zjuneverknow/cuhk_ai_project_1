import torch
from scipy import stats
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
The parameters of functions and the implementation of each TODO part are not given and can be modified by yourself.
New functions can also be added if you want.
'''

# spearman correlation using function from scipy library. you can also implement by yourself.
def spearman(W, q):
    return torch.tensor(stats.spearmanr(q, W)[0])

def cos():
    # TODO
    pass

# simlex-999 golden standard
with open('./simlex-999.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

standard = []
calculated = []
embed_vocabulary = []
for line in lines:
    word1, word2, std = tuple(line.split())
    word1, word2, std = word1.lower(), word2.lower(), eval(std)
    print("\nWord pair: {} and {}".format(word1, word2))

    if word1 in embed_vocabulary and word2 in embed_vocabulary:
        cal = 0
        # TODO
        print("The standard similarity is {} and the one calculated by embeddings is {}".format(std, cal))
        standard.append(std)
        calculated.append(cal)
    else:
        print("At least one of the words in this pair is not presented in the vocabulary.")

print("\nThe spearman correlation between the standard and calculated similarity is: {}".format(spearman(standard, calculated).item()))