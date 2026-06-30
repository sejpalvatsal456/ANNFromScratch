try:
  import cupy as xp
  GPU = True
  print("Running on GPU")
except ImportError:
  import numpy as xp
  GPU = False
  print("Running on CPU")