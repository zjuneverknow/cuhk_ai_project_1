# Project 1: CBOW

This folder contains a uv-managed runnable version of Project 1.

Run the original PyTorch CBOW pipeline:

```powershell
uv run python train_pytorch.py
```

Outputs are written to a parameterized output folder, for example:

- `outputs/output_alpha_only_0_min_freq_1_epoch_10`: default full Reuters tokens
- `outputs/output_alpha_only_1_min_freq_1_epoch_10`: alphabetic tokens only
- `outputs/output_alpha_only_1_min_freq_3_epoch_10`: alphabetic tokens only with `MIN_FREQ=3`

- `cbow.vec`: generated word embeddings
- `results.md`: KNN, SimLex-999, and analogy evaluation results
- `figures/`: analogy vector comparison plots

Optional environment variables:

- `STUDENT_ID`: random seed for selecting examples, default `5010`
- `EMBED_DIM`: embedding dimension, default `64`
- `CONTEXT_SIZE`: context window size, default `2`
- `BATCH_SIZE`: batch size, default `1024`
- `NUM_EPOCH`: number of training epochs, default `10`
- `MIN_FREQ`: minimum token frequency in the Reuters vocabulary, default `1`
- `MAX_SENTENCES`: number of Reuters sentences to use; `0` means all sentences
- `ALPHA_ONLY`: keep only alphabetic tokens when set to `1`, default `0`

Quick smoke test:

```powershell
$env:NUM_EPOCH="1"; $env:MAX_SENTENCES="200"; $env:MIN_FREQ="2"; uv run python train_pytorch.py
```

Alpha-only experiment:

```powershell
$env:ALPHA_ONLY="1"; $env:MIN_FREQ="3"; uv run python train_pytorch.py
```
