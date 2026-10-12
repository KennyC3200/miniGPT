import torch
from main import BigramLanguageModel, decode, encode, device

model = BigramLanguageModel()
model = model.to(device)

state_dict = torch.load('books.pth', map_location=device)
model.load_state_dict(state_dict)

model.eval()

context = torch.tensor([encode('Ivan')], dtype=torch.long, device=device)
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))