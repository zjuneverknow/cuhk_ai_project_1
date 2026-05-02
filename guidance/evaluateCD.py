# Useful codes for evaluate knn and analogy as defined in task evaluation for Project 1 and 3

import torch
from utils import load_pretrained

def knn(W, x, k):
    similarities = torch.matmul(x, W.transpose(1, 0)) / (torch.norm(W, dim=1) * torch.norm(x) + 1e-9) # compute cosine
    knn = similarities.topk(k=k)
    return knn.values.tolist(), knn.indices.tolist()

def find_similar_words(embeds, vocab, query, k=5): #finding most similar words using cosine
    knn_values, knn_indices = knn(embeds, embeds[vocab[query]], k + 1)
    knn_words = vocab.convert_ids_to_tokens(knn_indices)
    print(f">>> Query word: {query}")
    for i in range(k):
        print(f"cosine similarity={knn_values[i + 1]:.4f}: {knn_words[i + 1]}")

word_list = ["july", "reliable", "play", "willing", "good", "very", "patient", "concerned", "important", "powerful", "quickly", "generally", "gradually", "happy", "able", "close", "near",
             "saturday", "friend", "company", "road", "plane", "war", "politics", "building", "student", "university", "realm", "china", "experience", "police",
             "give", "create", "tell", "become", "lack", "win", "help", "gain", "get", "take", "use", "set", "find", "increase",
             "difficult", "go", "man", "ten", "year"]

word_sim_query = ["china", "august", "good", "paris"]
my_four_words_query =[] # student should generate his own query by using his ID to initialize random number seed
                        # and randomly select 4 words from the above word_list

vocab, embeds = load_pretrained("cbow.vec")
for w in my_four_words_query:
    find_similar_words(embeds, vocab, w)


def find_analogy(embeds, vocab, word_a, word_b, word_c):
    vecs = embeds[vocab.convert_tokens_to_ids([word_a, word_b, word_c])]
    x = vecs[2] + vecs[1] - vecs[0]
    knn_values, knn_indices = knn(embeds, x, k=1)
    analogies = vocab.convert_ids_to_tokens(knn_indices)
    print(f">>> Query: {word_a}, {word_b}, {word_c}")
    print(f"{analogies}")

word_analogy_queries = [["brother", "sister", "man"],
                        ["paris", "france", "berlin"]]
vocab, embeds = load_pretrained("cbow.vec")
for w_a, w_b, w_c in word_analogy_queries:
    find_analogy(embeds, vocab, w_a, w_b, w_c)

