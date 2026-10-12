import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 # number of independent sequences will be processed before updating gradients
block_size = 256 # max context length (in this case, number of characters to form a context)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embedding_size = 384 # 364 / 6 = 64 -> every head is 64-dimensional
n_layer = 6
num_heads = 6
dropout = 0.2

torch.manual_seed(1337)

with open('books/books.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { stoi[ch]:ch for ch in stoi }
vocab_size = len(chars)

# We need to encode and decode the numbers because neural networks only understand numbers
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Data
data = torch.tensor(encode(text), dtype=torch.long)

# Split up the data into train and validation sets
# First 90% for training, remaining 10% for validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data

    # Generate random offsets in the data
    offsets = torch.randint(len(data) - block_size, size=(batch_size,))

    # Inputs
    x = torch.stack([data[i:i+block_size] for i in offsets])

    # Targets
    y = torch.stack([data[i+1:i+1+block_size] for i in offsets])

    # REMEMBER: move the data to that device's memory
    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}

    # Set the model into evaluation phase
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[iter] = loss.item()
        out[split] = losses.mean()

    # Set the model back to training phase
    model.train()

    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

        # Used for masking (dimesion: context size by context size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x) # (B, T, d_k)
        Q = self.query(x) # (B, T, d_k)

        # Calculate softmax(Q * K^T/sqrt(d_k))
        weights = Q @ K.transpose(-2, -1) * C ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)

        V = self.value(x) # (B, T, d_k)
        out = weights @ V # (B, T, d_k)

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))

        # Right now, this is kinda pointless b/c projection is suppose to change from one dimension
        # to another. Later one we will change this
        self.proj = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the results from the distinct heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()

        # For the last layer, we could have a self.proj like in MultiHeadAttention, but for now
        # let's just do nn.Linear() to make it simple
        # For residual connections, we first project it from embedding_size -> 4 * embedding_size
        # Then, b/c this is residual, we need to project it back to embedding_size to be added
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size), 
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Transfomer block: communication followed by computation

    def __init__(self, embedding_size, num_heads):
        super().__init__()
        head_size = embedding_size // num_heads

        # Self attention (4 heads of 8-dimensional self-attention with our parameters)
        self.sa = MultiHeadAttention(num_heads, head_size)

        # Feed forward
        self.ffwd = FeedForward(embedding_size)

        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        # Putting x + before is the residual connections
        x = x + self.sa(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)

        # Language modeling head. This is a linear LAYER that has `embedding_size` input features
        # and `vocab_size` output features
        self.lm_head = nn.Linear(embedding_size, vocab_size) 

        self.blocks = nn.Sequential(*[Block(embedding_size, num_heads=num_heads) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, indices, targets=None):
        B, T = indices.shape

        token_embedding = self.token_embedding_table(indices) # (B,T,C)

        # Use the lookup table to get the embeddings for that position via torch.arange which
        # gives [1,T-1]
        position_embedding = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)

        # Add the token embedding to the position embedding. This is essentially passing the
        # position information
        x = token_embedding + position_embedding # (B,T,C)
        x = self.blocks(x) # (B,T,C)

        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Cross-entropy conceptually wants:
            # logits: (# of examples, # of classes)
            # targets: (# of examples)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, indices, max_new_tokens):
        # Indices is (B,T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            # Crop indices to the last block_size tokens
            indices_cond = indices[:, -block_size:]

            # Receive the predictions. Since targets=None, the reshaping does not happen for logits
            logits, loss = self(indices_cond) # logits: (B,T,C) 

            # Logits for only the last time-step to generate the next character
            # This kind of defeats the idea of "context" because it is solely using the last char
            # to predict the next one, but this is the bigram model. And as you can guess, it sucks
            logits = logits[:, -1, :] # becomes (B,C) for T=-1

            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_idx = torch.multinomial(probabilities, num_samples=1) # (B,1)

            indices = torch.cat((indices, next_idx), dim=1) # (B,T+1)

        return indices

if __name__ == '__main__':
    model = BigramLanguageModel()
    m = model.to(device) # REMEMBER: move the model parameters to the device

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # Every once in a while evaluate the loss on the train and validation runs
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model parameters
    torch.save(model.state_dict(), 'books.pth')