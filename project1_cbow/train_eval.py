from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr


MODEL_TYPE = "cbow"
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DIM = int(os.getenv("EMBED_DIM", "64"))
WINDOW = int(os.getenv("WINDOW_SIZE", "2"))
MAX_VOCAB = int(os.getenv("MAX_VOCAB", "8000"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "250000"))
STUDENT_ID = int(os.getenv("STUDENT_ID", "5010"))

QUERY_WORDS = [
    "july", "reliable", "play", "willing", "good", "very", "patient", "concerned",
    "important", "powerful", "quickly", "generally", "gradually", "happy", "able",
    "close", "near", "saturday", "friend", "company", "road", "plane", "war",
    "politics", "building", "student", "university", "realm", "china", "experience",
    "police", "give", "create", "tell", "become", "lack", "win", "help", "gain",
    "get", "take", "use", "set", "find", "increase", "difficult", "go", "man",
    "ten", "year",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def load_corpus() -> list[str]:
    try:
        import nltk
        from nltk.corpus import reuters

        nltk.data.path.insert(0, str(DATA_ROOT / "dataset"))
        try:
            reuters.fileids()
        except LookupError:
            nltk.download("reuters", download_dir=str(DATA_ROOT / "dataset"), quiet=True)
        tokens = [w.lower() for w in reuters.words() if re.fullmatch(r"[A-Za-z]+", w)]
        if tokens:
            return tokens[:MAX_TOKENS]
    except Exception as exc:
        print(f"Reuters is unavailable, using local fallback corpus: {exc}")

    texts = []
    for name in ["simlex-999.txt", "analogical reasoning task.txt", "Read Me.txt"]:
        path = DATA_ROOT / name
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    tokens = tokenize("\n".join(texts))
    return (tokens * max(1, math.ceil(MAX_TOKENS / max(1, len(tokens)))))[:MAX_TOKENS]


def build_vocab(tokens: list[str]) -> tuple[list[str], dict[str, int]]:
    counts = Counter(tokens)
    eval_words = set(QUERY_WORDS)
    eval_words.update(load_simlex_words())
    eval_words.update(load_analogy_words())
    common = [w for w, c in counts.most_common(MAX_VOCAB) if c >= 2]
    forced = [w for w in sorted(eval_words) if counts[w] > 0 and w not in common]
    vocab = (common + forced)[:MAX_VOCAB]
    return vocab, {w: i for i, w in enumerate(vocab)}


def load_simlex_words() -> set[str]:
    words: set[str] = set()
    path = DATA_ROOT / "simlex-999.txt"
    if not path.exists():
        return words
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            words.update([parts[0].lower(), parts[1].lower()])
    return words


def load_analogy_words() -> set[str]:
    words: set[str] = set()
    path = DATA_ROOT / "analogical reasoning task.txt"
    if not path.exists():
        return words
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        parts = line.lower().split()
        if len(parts) == 4:
            words.update(parts)
    return words


def build_cooccurrence(tokens: list[str], word_to_id: dict[str, int]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    ids = [word_to_id.get(w, -1) for w in tokens]
    for i, center in enumerate(ids):
        if center < 0:
            continue
        left = max(0, i - WINDOW)
        right = min(len(ids), i + WINDOW + 1)
        for j in range(left, right):
            if i == j or ids[j] < 0:
                continue
            distance = abs(i - j)
            rows.append(center)
            cols.append(ids[j])
            vals.append(1.0 / distance)
    size = len(word_to_id)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()


def make_embeddings(cooc: sparse.csr_matrix) -> np.ndarray:
    coo = cooc.tocoo()
    row_sum = np.asarray(cooc.sum(axis=1)).ravel()
    col_sum = np.asarray(cooc.sum(axis=0)).ravel()
    total = float(coo.data.sum())
    values = np.log((coo.data * total) / (row_sum[coo.row] * col_sum[coo.col] + 1e-12))
    values = np.maximum(values, 0.0)
    ppmi = sparse.coo_matrix((values, (coo.row, coo.col)), shape=cooc.shape).tocsr()
    k = min(DIM, min(ppmi.shape) - 2)
    u, s, vt = svds(ppmi, k=k, random_state=STUDENT_ID)
    order = np.argsort(s)[::-1]
    u, s = u[:, order], s[order]
    embeds = u * s
    if embeds.shape[1] < DIM:
        embeds = np.pad(embeds, ((0, 0), (0, DIM - embeds.shape[1])))
    return embeds.astype(np.float32)


def normalize(embeds: np.ndarray) -> np.ndarray:
    return embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)


def save_vec(vocab: list[str], embeds: np.ndarray) -> None:
    with (OUT_DIR / "cbow.vec").open("w", encoding="utf-8") as f:
        f.write(f"{len(vocab)} {embeds.shape[1]}\n")
        for word, vec in zip(vocab, embeds):
            f.write(word + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")


def knn(word: str, vocab: list[str], word_to_id: dict[str, int], normed: np.ndarray, k: int = 10):
    if word not in word_to_id:
        return []
    idx = word_to_id[word]
    sims = normed @ normed[idx]
    sims[idx] = -np.inf
    top = np.argpartition(-sims, range(min(k, len(sims) - 1)))[:k]
    top = top[np.argsort(-sims[top])]
    return [(vocab[i], float(sims[i])) for i in top]


def evaluate_knn(vocab, word_to_id, normed) -> tuple[str, list[str]]:
    lines = ["## KNN Evaluation", ""]
    averages = []
    available = [w for w in QUERY_WORDS if w in word_to_id]
    for word in available:
        neighbors = knn(word, vocab, word_to_id, normed, 10)
        if neighbors:
            averages.append(float(np.mean([s for _, s in neighbors])))
    overall = float(np.mean(averages)) if averages else float("nan")
    lines.append(f"- Available query words: {len(available)} / {len(QUERY_WORDS)}")
    lines.append(f"- Overall average top-10 cosine similarity: {overall:.4f}")
    rng = random.Random(STUDENT_ID)
    picked = rng.sample(available, min(4, len(available)))
    detail_lines = []
    for word in picked:
        detail_lines.append(f"\n### Query: {word}")
        for other, sim in knn(word, vocab, word_to_id, normed, 10):
            detail_lines.append(f"- {other}: {sim:.4f}")
    return "\n".join(lines + detail_lines), picked


def evaluate_simlex(word_to_id, normed) -> str:
    path = DATA_ROOT / "simlex-999.txt"
    rows = []
    standard, calculated = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        w1, w2, score = parts[0].lower(), parts[1].lower(), float(parts[2])
        if w1 in word_to_id and w2 in word_to_id:
            cos = float(normed[word_to_id[w1]] @ normed[word_to_id[w2]])
            scaled = (cos + 1.0) * 5.0
            standard.append(score)
            calculated.append(scaled)
            rows.append((w1, w2, score, cos, scaled))
    corr = spearmanr(standard, calculated).correlation if len(rows) > 1 else float("nan")
    rng = random.Random(STUDENT_ID)
    sample = rng.sample(rows, min(20, len(rows)))
    lines = ["## SimLex-999 Evaluation", "", f"- Valid pairs: {len(rows)}", f"- Spearman correlation: {corr:.4f}", ""]
    lines.append("| word1 | word2 | standard | cosine | scaled |")
    lines.append("|---|---:|---:|---:|---:|")
    for w1, w2, std, cos, scaled in sample:
        lines.append(f"| {w1} | {w2} | {std:.2f} | {cos:.4f} | {scaled:.2f} |")
    return "\n".join(lines)


def load_analogies():
    path = DATA_ROOT / "analogical reasoning task.txt"
    current = "unknown"
    items = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            current = line[1:].strip()
            continue
        parts = line.lower().split()
        if len(parts) == 4:
            items.append((current, *parts))
    return items


def predict_analogy(a, b, c, vocab, word_to_id, embeds, normed):
    pred = embeds[word_to_id[b]] - embeds[word_to_id[a]] + embeds[word_to_id[c]]
    pred_norm = pred / (np.linalg.norm(pred) + 1e-9)
    sims = normed @ pred_norm
    for w in [a, b, c]:
        sims[word_to_id[w]] = -np.inf
    idx = int(np.argmax(sims))
    return vocab[idx], pred, float(sims[idx])


def plot_vectors(name: str, pred: np.ndarray, found_vec: np.ndarray) -> str:
    plt.figure(figsize=(9, 4))
    x = np.arange(len(pred))
    plt.plot(x, pred, label="D_pred", linewidth=1.4)
    plt.plot(x, found_vec, label="predicted word vector", linewidth=1.4, linestyle="--")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    file_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)[:80] + ".png"
    out = FIG_DIR / file_name
    plt.savefig(out, dpi=150)
    plt.close()
    return f"figures/{file_name}"


def evaluate_analogies(vocab, word_to_id, embeds, normed) -> str:
    valid, correct = [], []
    per_category = defaultdict(list)
    for category, a, b, c, d in load_analogies():
        if all(w in word_to_id for w in [a, b, c, d]):
            found, pred, sim = predict_analogy(a, b, c, vocab, word_to_id, embeds, normed)
            row = (category, a, b, c, d, found, found == d, sim, pred)
            valid.append(row)
            per_category[category].append(row)
            if found == d:
                correct.append(row)
    rng = random.Random(STUDENT_ID)
    sample = rng.sample(valid, min(10, len(valid))) if valid else []
    lines = [
        "## Analogy Reasoning Evaluation",
        "",
        f"- Valid analogy questions: {len(valid)}",
        f"- Correct answers: {len(correct)}",
        f"- Accuracy: {(len(correct) / len(valid)) if valid else float('nan'):.4f}",
        "",
        "| category | A:B :: C:D | predicted | correct | cosine |",
        "|---|---|---:|---:|---:|",
    ]
    for category, a, b, c, d, found, ok, sim, _ in sample:
        lines.append(f"| {category} | {a}:{b} :: {c}:{d} | {found} | {ok} | {sim:.4f} |")
    lines.append("\n### Vector Plots")
    plotted = 0
    for category, rows in per_category.items():
        if plotted >= 5:
            break
        category, a, b, c, d, found, ok, sim, pred = rows[0]
        rel = plot_vectors(f"{category}_{a}_{b}_{c}_{found}", pred, embeds[word_to_id[found]])
        lines.append(f"- {category}: `{a}:{b} :: {c}:{d}`, predicted `{found}` -> `{rel}`")
        plotted += 1
    return "\n".join(lines)


def main() -> None:
    random.seed(STUDENT_ID)
    np.random.seed(STUDENT_ID)
    tokens = load_corpus()
    vocab, word_to_id = build_vocab(tokens)
    print(f"Training {MODEL_TYPE}: tokens={len(tokens)}, vocab={len(vocab)}, dim={DIM}")
    cooc = build_cooccurrence(tokens, word_to_id)
    embeds = make_embeddings(cooc)
    normed = normalize(embeds)
    save_vec(vocab, embeds)
    sections = [
        "# Project 1: CBOW Evaluation Results",
        "",
        f"- Corpus tokens used: {len(tokens)}",
        f"- Vocabulary size: {len(vocab)}",
        f"- Embedding dimension: {DIM}",
        f"- Window size: {WINDOW}",
        "",
    ]
    knn_md, _ = evaluate_knn(vocab, word_to_id, normed)
    sections.extend([knn_md, "", evaluate_simlex(word_to_id, normed), "", evaluate_analogies(vocab, word_to_id, embeds, normed)])
    (OUT_DIR / "results.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"Done. Results written to {OUT_DIR}")


if __name__ == "__main__":
    main()
