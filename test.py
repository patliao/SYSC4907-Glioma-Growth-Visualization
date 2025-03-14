import torch


print(f"{torch.cuda.is_available()}")
torch.cuda.empty_cache()