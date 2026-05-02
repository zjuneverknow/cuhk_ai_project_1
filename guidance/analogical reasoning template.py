import torch

'''
The parameters of functions and the implementation of each TODO part are not given and can be modified by yourself.
New functions can also be added if you want.
'''

# analogical reasoning task dataset
with open('./analogical reasoning task.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

def cos():
    # TODO
    pass

def knn():
    # TODO
    pass

num_exist = 0
correct = []
for line in lines:
    # TODO
    pass

print("For {} analogy questions with all words presented in the vocabulary, {} of them are correctly answered."
      .format(num_exist, len(correct)))