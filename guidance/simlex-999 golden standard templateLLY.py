import torch
import torch.nn.functional as F
from scipy import stats
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''
Word Similarity Golden Standard Evaluation
'''

# ---------------- CONFIGURATION ----------------
MODEL_PATH = 'skipgram.vec'


# -----------------------------------------------

# spearman correlation using function from scipy library.
def spearman(W, q):
    return torch.tensor(stats.spearmanr(q, W)[0])


def load_embeddings(file_path):
    print(f"Loading embeddings from {file_path}...")
    embeddings = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines[0].split()) == 2:
            lines = lines[1:]  # Skip header
        for line in lines:
            parts = line.strip().split()
            word = parts[0]
            vec = torch.tensor([float(x) for x in parts[1:]])
            embeddings[word] = vec
    return embeddings


embeddings = load_embeddings(MODEL_PATH)


def cos(v1, v2):
    # Calculate cosine similarity
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


# simlex-999 golden standard [cite: 240]
with open('./simlex-999.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Usually the first line of SimLex-999 is a header (word1\tword2\tPOS...), remove it if present
if 'word1' in lines[0].lower():
    lines = lines[1:]

standard = []
calculated = []
# embed_vocabulary list for checking existence
embed_vocabulary = set(embeddings.keys())

count = 0

for line in lines:
    parts = line.strip().split()  # SimLex is typically tab or space separated
    # Handling potential format variations, usually first two are words, then score is somewhere later
    # Assuming format: word1 word2 similarity ...
    word1, word2 = parts[0], parts[1]
    std_score = float(parts[2])  # SimLex-999 standard format: word1 word2 POS similarity_score

    word1, word2 = word1.lower(), word2.lower()

    # Optional: limit print output to random sample or just process all
    # print("\nWord pair: {} and {}".format(word1, word2))

    if word1 in embed_vocabulary and word2 in embed_vocabulary:
        vec1 = embeddings[word1]
        vec2 = embeddings[word2]

        # Calculate raw cosine similarity [-1, 1]
        raw_cos = cos(vec1, vec2)

        # Scale to [0, 10] as per PDF instruction
        # Formula: (x + 1) * 5
        cal = (raw_cos + 1) * 5

        # [cite: 243]
        # print("The standard similarity is {} and the one calculated by embeddings is {:.2f}".format(std_score, cal))
        standard.append(std_score)
        calculated.append(cal)
        count += 1
    else:
        pass
        # print("At least one of the words in this pair is not presented in the vocabulary.")

print(f"\nEvaluated {count} pairs.")
print("\nThe spearman correlation between the standard and calculated similarity is: {}".format(
    spearman(standard, calculated).item()))