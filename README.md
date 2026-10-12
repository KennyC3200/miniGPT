# miniGPT

A character-level GPT: implemented causal multi-head self-attention, positional embeddings, feed-forward blocks, residual connections, and layer normalization, then trained the model with cross-entropy loss and the AdamW optimizer. Evaluated using train/validation loss and generated text autoregressively. Inspired by Andrej Karpathy's GPT from scratch.

![Transformer architecture](transformer_architecture.png)

## Features

- **Causal multi-head self-attention** with a lower-triangular mask so each position only attends to earlier characters
- **Token and positional embeddings** summed to give the model both identity and order information
- **Feed-forward blocks** (Linear → ReLU → Linear, 4× hidden expansion) with dropout
- **Residual connections and layer normalization** (pre-norm) around each sub-layer
- **Training** with cross-entropy loss and the AdamW optimizer
- **Evaluation** via averaged train/validation loss, plus autoregressive text generation by sampling from the softmax distribution

## Project structure

| Path | Description |
| --- | --- |
| `main.py` | Model definition, data loading, and training loop |
| `eval.py` | Loads saved weights and generates text |
| `main.ipynb` | Notebook version of the walkthrough |
| `books/` | Training text (Alice in Wonderland, The Time Machine, The Wizard of Oz) |
| `shakespeare/` | Tiny Shakespeare dataset |

## Hyperparameters

| Name | Value |
| --- | --- |
| Batch size | 64 |
| Context length (`block_size`) | 256 characters |
| Embedding size | 384 |
| Layers / heads | 6 / 6 (64 dimensions per head) |
| Dropout | 0.2 |
| Learning rate | 3e-4 |
| Training iterations | 5000 |
| Train/validation split | 90% / 10% |

## Getting started

Requires Python 3.12+ and PyTorch.

```bash
pip install torch
```

**Train** (uses CUDA if available, otherwise CPU). Train and validation loss are printed every 500 steps, and the weights are saved to `books.pth`:

```bash
python main.py
```

**Generate text** from the trained weights:

```bash
python eval.py
```

To train on a different corpus, change the file path in `main.py` (for example `shakespeare/input.txt`). The vocabulary is built from the unique characters in that file. To change the generation prompt, edit the `context` line in `eval.py`.

## Acknowledgements

Inspired by Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and the accompanying [nanoGPT](https://github.com/karpathy/nanoGPT) project.
