from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
WEIGHT_INIT_RANGE = 0.1

QUERY_WORDS = [
    "july", "reliable", "play", "willing", "good", "very", "patient", "concerned",
    "important", "powerful", "quickly", "generally", "gradually", "happy", "able",
    "close", "near", "saturday", "friend", "company", "road", "plane", "war",
    "politics", "building", "student", "university", "realm", "china", "experience",
    "police", "give", "create", "tell", "become", "lack", "win", "help", "gain",
    "get", "take", "use", "set", "find", "increase", "difficult", "go", "man",
    "ten", "year",
]


def log(message: str) -> None:
    print(f"[word2vec] {message}", flush=True)


def tqdm_kwargs(**kwargs):
    base = {
        "dynamic_ncols": True,
        "mininterval": 1.0,
        "leave": True,
    }
    base.update(kwargs)
    return base


class Vocab:
    def __init__(self, tokens: list[str]):
        if UNK_TOKEN not in tokens:
            tokens = [UNK_TOKEN] + tokens
        self.idx_to_token = []
        self.token_to_idx = {}
        for token in tokens:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)
        self.unk = self.token_to_idx[UNK_TOKEN]

    @classmethod
    def build(cls, text: list[list[str]], min_freq: int = 1, reserved_tokens: list[str] | None = None):
        token_freqs = Counter(token for sentence in text for token in sentence)
        uniq_tokens = [UNK_TOKEN] + (reserved_tokens or [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != UNK_TOKEN]
        return cls(uniq_tokens)

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices) -> list[str]:
        return [self.idx_to_token[int(index)] for index in indices]


class CbowDataset(Dataset):
    def __init__(self, corpus: list[list[int]], vocab: Vocab, context_size: int = 2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, **tqdm_kwargs(desc="CBOW dataset", total=len(corpus))):
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size * 2 + 1:
                continue
            for i in range(context_size, len(sentence) - context_size):
                context = sentence[i - context_size:i] + sentence[i + 1:i + context_size + 1]
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return inputs, targets


class SkipGramDataset(Dataset):
    def __init__(self, corpus: list[list[int]], vocab: Vocab, context_size: int = 2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, **tqdm_kwargs(desc="Skip-Gram dataset", total=len(corpus))):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                w = sentence[i]
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                context = sentence[left_context_index:i] + sentence[i + 1:right_context_index + 1]
                self.data.extend((w, c) for c in context)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return inputs, targets


class CbowModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        init_weights(self)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden = embeds.mean(dim=1)
        output = self.output(hidden)
        return F.log_softmax(output, dim=1)


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        init_weights(self)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.output(embeds)
        return F.log_softmax(output, dim=1)


@dataclass
class RunConfig:
    model_type: str
    project_title: str
    output_vec_name: str
    root_dir: Path
    output_dir: Path
    embedding_dim: int = 64
    context_size: int = 2
    batch_size: int = 1024
    num_epoch: int = 10
    min_freq: int = 1
    max_sentences: int = 0
    student_id: int = 5010
    alpha_only: bool = False


def init_weights(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "embedding" not in name:
            nn.init.uniform_(param, a=-WEIGHT_INIT_RANGE, b=WEIGHT_INIT_RANGE)


def read_config(model_type: str, project_title: str, output_vec_name: str, project_dir: Path) -> RunConfig:
    root_dir = project_dir.parent
    alpha_only = os.getenv("ALPHA_ONLY", "0").strip().lower() in {"1", "true", "yes", "y"}
    embedding_dim = int(os.getenv("EMBED_DIM", "64"))
    context_size = int(os.getenv("CONTEXT_SIZE", "2"))
    batch_size = int(os.getenv("BATCH_SIZE", "1024"))
    num_epoch = int(os.getenv("NUM_EPOCH", "10"))
    min_freq = int(os.getenv("MIN_FREQ", "1"))
    max_sentences = int(os.getenv("MAX_SENTENCES", "0"))
    student_id = int(os.getenv("STUDENT_ID", "5010"))

    output_dir = project_dir / output_dir_name(
        alpha_only=alpha_only,
        min_freq=min_freq,
        num_epoch=num_epoch,
        max_sentences=max_sentences,
    )

    return RunConfig(
        model_type=model_type,
        project_title=project_title,
        output_vec_name=output_vec_name,
        root_dir=root_dir,
        output_dir=output_dir,
        embedding_dim=embedding_dim,
        context_size=context_size,
        batch_size=batch_size,
        num_epoch=num_epoch,
        min_freq=min_freq,
        max_sentences=max_sentences,
        student_id=student_id,
        alpha_only=alpha_only,
    )


def output_dir_name(alpha_only: bool, min_freq: int, num_epoch: int, max_sentences: int) -> str:
    parts = ["outputs", f"alpha_only_{int(alpha_only)}"]
    if min_freq != 1:
        parts.append(f"min_freq_{min_freq}")
    if num_epoch != 10:
        parts.append(f"epochs_{num_epoch}")
    if max_sentences != 0:
        parts.append(f"max_sentences_{max_sentences}")
    return "_".join(parts)


def load_reuters(config: RunConfig) -> tuple[list[list[int]], Vocab]:
    import nltk
    from nltk.corpus import reuters

    dataset_dir = config.root_dir / "dataset"
    nltk.data.path.insert(0, str(dataset_dir))
    log(f"Loading Reuters corpus from {dataset_dir}")
    try:
        fileids = reuters.fileids()
    except LookupError:
        log("Reuters corpus not found locally; downloading with nltk...")
        nltk.download("reuters", download_dir=str(dataset_dir), quiet=True)
        fileids = reuters.fileids()

    log("Converting Reuters documents to lowercase token sentences")
    if config.alpha_only:
        log("ALPHA_ONLY=1: keeping only tokens that match ^[a-z]+$")
    text = []
    sentence = []
    for fileid in tqdm(fileids, **tqdm_kwargs(desc="Reuters files", total=len(fileids))):
        for word in reuters.words(fileid):
            token = word.lower()
            if token in {".", "?", "!", ";"}:
                if sentence:
                    text.append(sentence)
                    sentence = []
                continue
            if config.alpha_only and not re.fullmatch(r"[a-z]+", token):
                continue
            if re.search(r"[a-z0-9]", token):
                sentence.append(token)
        if sentence:
            text.append(sentence)
            sentence = []

    if config.max_sentences > 0:
        text = text[:config.max_sentences]
        log(f"Using first {len(text)} Reuters sentences for this run")
    else:
        log(f"Using all {len(text)} Reuters sentences")

    log(f"Building vocabulary with MIN_FREQ={config.min_freq}")
    vocab = Vocab.build(text, min_freq=config.min_freq, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    log(f"Vocabulary size: {len(vocab)}")
    log("Converting tokens to ids")
    corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]
    return corpus, vocab


def save_pretrained(vocab: Vocab, embeds: torch.Tensor, save_path: Path) -> None:
    with save_path.open("w", encoding="utf-8") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.idx_to_token):
            vec = " ".join(f"{float(x):.4f}" for x in embeds[idx])
            writer.write(f"{token} {vec}\n")
    print(f"Pretrained embeddings saved to: {save_path}")


def guidance_file(root_dir: Path, file_name: str) -> Path:
    candidates = [root_dir / file_name, root_dir / "guidance" / file_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {file_name} in {root_dir} or {root_dir / 'guidance'}")


def train_model(config: RunConfig) -> tuple[Vocab, torch.Tensor, list[float], str]:
    random.seed(config.student_id)
    np.random.seed(config.student_id)
    torch.manual_seed(config.student_id)

    corpus, vocab = load_reuters(config)
    if config.model_type == "cbow":
        log("Constructing CBOW training examples")
        dataset = CbowDataset(corpus, vocab, context_size=config.context_size)
        model = CbowModel(len(vocab), config.embedding_dim)
    elif config.model_type == "skipgram":
        log("Constructing Skip-Gram training examples")
        dataset = SkipGramDataset(corpus, vocab, context_size=config.context_size)
        model = SkipGramModel(len(vocab), config.embedding_dim)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    data_loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=dataset.collate_fn, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    nll_loss = nn.NLLLoss()

    log(
        f"Training {config.model_type}: device={device}, sentences={len(corpus)}, "
        f"vocab={len(vocab)}, examples={len(dataset)}, epochs={config.num_epoch}"
    )
    losses = []
    model.train()
    for epoch in range(config.num_epoch):
        total_loss = 0.0
        progress = tqdm(data_loader, **tqdm_kwargs(desc=f"Training Epoch {epoch + 1}/{config.num_epoch}", total=len(data_loader)))
        for batch in progress:
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            log_probs = model(inputs)
            loss = nll_loss(log_probs, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{float(loss.item()):.4f}")
        losses.append(total_loss)
        log(f"Epoch {epoch + 1}/{config.num_epoch} loss: {total_loss:.2f}")

    embeds = model.embeddings.weight.detach().cpu()
    return vocab, embeds, losses, str(device)


def normalized(embeds: torch.Tensor) -> np.ndarray:
    arr = embeds.numpy().astype(np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def knn(word: str, vocab: Vocab, normed: np.ndarray, k: int = 10):
    if word not in vocab.token_to_idx:
        return []
    idx = vocab[word]
    sims = normed @ normed[idx]
    sims[idx] = -np.inf
    top = np.argpartition(-sims, range(min(k, len(sims) - 1)))[:k]
    top = top[np.argsort(-sims[top])]
    return [(vocab.idx_to_token[int(i)], float(sims[int(i)])) for i in top]


def evaluate_knn(vocab: Vocab, normed: np.ndarray, seed: int) -> str:
    lines = ["## KNN Evaluation", ""]
    available = [word for word in QUERY_WORDS if word in vocab.token_to_idx]
    averages = []
    for word in available:
        neighbors = knn(word, vocab, normed, 10)
        if neighbors:
            averages.append(float(np.mean([score for _, score in neighbors])))
    lines.append(f"- Available query words: {len(available)} / {len(QUERY_WORDS)}")
    lines.append(f"- Overall average top-10 cosine similarity: {np.mean(averages) if averages else float('nan'):.4f}")

    rng = random.Random(seed)
    for word in rng.sample(available, min(4, len(available))):
        lines.append(f"\n### Query: {word}")
        for other, score in knn(word, vocab, normed, 10):
            lines.append(f"- {other}: {score:.4f}")
    return "\n".join(lines)


def evaluate_simlex(root_dir: Path, vocab: Vocab, normed: np.ndarray, seed: int) -> str:
    rows = []
    standard, calculated = [], []
    for line in guidance_file(root_dir, "simlex-999.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        w1, w2, score = parts[0].lower(), parts[1].lower(), float(parts[2])
        if w1 in vocab.token_to_idx and w2 in vocab.token_to_idx:
            cos = float(normed[vocab[w1]] @ normed[vocab[w2]])
            scaled = (cos + 1.0) * 5.0
            rows.append((w1, w2, score, cos, scaled))
            standard.append(score)
            calculated.append(scaled)
    corr = spearmanr(standard, calculated).correlation if len(rows) > 1 else float("nan")
    sample = random.Random(seed).sample(rows, min(20, len(rows))) if rows else []
    lines = ["## SimLex-999 Evaluation", "", f"- Valid pairs: {len(rows)}", f"- Spearman correlation: {corr:.4f}", ""]
    lines.append("| word1 | word2 | standard | cosine | scaled |")
    lines.append("|---|---:|---:|---:|---:|")
    for w1, w2, std, cos, scaled in sample:
        lines.append(f"| {w1} | {w2} | {std:.2f} | {cos:.4f} | {scaled:.2f} |")
    return "\n".join(lines)


def load_analogies(root_dir: Path):
    current = "unknown"
    for line in guidance_file(root_dir, "analogical reasoning task.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            current = line[1:].strip()
            continue
        parts = line.lower().split()
        if len(parts) == 4:
            yield (current, *parts)


def plot_vectors(fig_dir: Path, name: str, pred: np.ndarray, found_vec: np.ndarray) -> str:
    plt.figure(figsize=(9, 4))
    x = np.arange(len(pred))
    plt.plot(x, pred, label="D_pred", linewidth=1.4)
    plt.plot(x, found_vec, label="predicted word vector", linewidth=1.4, linestyle="--")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    file_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)[:80] + ".png"
    plt.savefig(fig_dir / file_name, dpi=150)
    plt.close()
    return f"figures/{file_name}"


def evaluate_analogies(config: RunConfig, vocab: Vocab, embeds: torch.Tensor, normed: np.ndarray) -> str:
    arr = embeds.numpy().astype(np.float32)
    valid, correct = [], []
    per_category = defaultdict(list)
    for category, a, b, c, d in load_analogies(config.root_dir):
        if all(word in vocab.token_to_idx for word in [a, b, c, d]):
            pred = arr[vocab[b]] - arr[vocab[a]] + arr[vocab[c]]
            pred_norm = pred / (np.linalg.norm(pred) + 1e-9)
            sims = normed @ pred_norm
            for word in [a, b, c]:
                sims[vocab[word]] = -np.inf
            found_idx = int(np.argmax(sims))
            found = vocab.idx_to_token[found_idx]
            row = (category, a, b, c, d, found, found == d, float(sims[found_idx]), pred)
            valid.append(row)
            per_category[category].append(row)
            if found == d:
                correct.append(row)

    sample = random.Random(config.student_id).sample(valid, min(10, len(valid))) if valid else []
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
        rel = plot_vectors(config.output_dir / "figures", f"{category}_{a}_{b}_{c}_{found}", pred, arr[vocab[found]])
        lines.append(f"- {category}: `{a}:{b} :: {c}:{d}`, predicted `{found}` -> `{rel}`")
        plotted += 1
    return "\n".join(lines)


def run_pipeline(model_type: str, project_title: str, output_vec_name: str, project_dir: Path) -> None:
    config = read_config(model_type, project_title, output_vec_name, project_dir)
    config.output_dir.mkdir(exist_ok=True)
    (config.output_dir / "figures").mkdir(exist_ok=True)

    log("Starting PyTorch Word2Vec pipeline")
    vocab, embeds, losses, device = train_model(config)
    log(f"Saving embeddings to {config.output_dir / output_vec_name}")
    save_pretrained(vocab, embeds, config.output_dir / output_vec_name)
    log("Running KNN, SimLex-999, and analogy evaluations")
    normed = normalized(embeds)
    sections = [
        f"# {config.project_title}",
        "",
        f"- Training implementation: original PyTorch {config.model_type} from chp5 style",
        f"- Device: {device}",
        f"- Vocabulary size: {len(vocab)}",
        f"- Embedding dimension: {config.embedding_dim}",
        f"- Context size: {config.context_size}",
        f"- Batch size: {config.batch_size}",
        f"- Epochs: {config.num_epoch}",
        f"- Min frequency: {config.min_freq}",
        f"- Alpha-only token filter: {config.alpha_only}",
        f"- Max sentences: {config.max_sentences if config.max_sentences > 0 else 'all Reuters sentences'}",
        f"- Losses: {', '.join(f'{loss:.2f}' for loss in losses)}",
        "",
        evaluate_knn(vocab, normed, config.student_id),
        "",
        evaluate_simlex(config.root_dir, vocab, normed, config.student_id),
        "",
        evaluate_analogies(config, vocab, embeds, normed),
    ]
    (config.output_dir / "results.md").write_text("\n".join(sections), encoding="utf-8")
    log(f"Done. Results written to {config.output_dir}")
