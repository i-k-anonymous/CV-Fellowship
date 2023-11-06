import torch
if torch.cuda.is_available():
  print("CUDA is available. Use device string 'cuda'")
elif torch.backends.mps.is_available():
  print("MPS is available. Use device string 'mps'")
else:
  print("CPU is available. Use device string 'cpu'")