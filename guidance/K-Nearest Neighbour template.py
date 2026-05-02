import torch

'''
The parameters of functions and the implementation of each TODO part are not given and can be modified by yourself.
New functions can also be added if you want.
'''

# query word list
word_list = ["july", "reliable", "play", "willing", "good", "very", "patient", "concerned", "important", "powerful", "quickly", "generally", "gradually", "happy", "able", "close", "near",
             "saturday", "friend", "company", "road", "plane", "war", "politics", "building", "student", "university", "realm", "china", "experience", "police",
             "give", "create", "tell", "become", "lack", "win", "help", "gain", "get", "take", "use", "set", "find", "increase",
             "difficult", "go", "man", "ten", "year"]

# calculate cosine similarity
def cos():
    # TODO
    pass

# find k-nearest neighbors
def knn():
    # TODO
    pass

# evaluation process for each word
def evaluate(word):
    knn_words = []
    knn_sims = []
    # TODO
    print("\nQuery:", word)
    for i in range(len(knn_words)):
        print("The similarity of {} is {}.".format(knn_words[i], knn_sims[i]))

# TODO
for word in word_list:
    evaluate(word)