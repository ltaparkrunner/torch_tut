import os
import numpy as np
import re

from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
from torch.utils.data import BatchSampler, SequentialSampler
import torch.utils.data as data
import torchvision
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

# from navec import Navec
# path = 'hudlit_12B_500K_300d_100q.tar'
# navec = Navec.load(path)

from navec import Navec

path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)
print(navec['навек'][:20])

print(navec.vocab.words[300000:300020])

# Check total vocabulary size (500,000 words + <unk> and <pad>)
print(len(navec.vocab.words))
# print('кот' in navec)        # True
# print('абракадабра' in navec) # False (вернет вектор <unk>)
# print(navec['абракадабра'])

# 1. Unpack the full 500,002 x 300 embedding matrix into RAM (~600 MB)
matrix = navec.pq.unpack()

# 2. Pick a target vector (e.g., for the word 'навек' or any 300d array)
target_vec = navec['навек']

# 3. Compute cosine similarity with every word in the vocabulary
norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(target_vec)
# Avoid division by zero for the <pad> token (which is all zeros)
norms[norms == 0] = 1e-10 
similarities = np.dot(matrix, target_vec) / norms

# 4. Get the top 10 closest word indices
top_indices = np.argsort(similarities)[::-1][:10]

for idx in top_indices:
    word = navec.vocab.words[idx]
    score = similarities[idx]
    print(f"{word}: {score:.4f}")
