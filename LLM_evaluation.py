import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LLM_model import GPT

torch.cuda.is_available()

import gc
torch.cuda.empty_cache()
gc.collect()

'''LLM config'''
config = {
    "embed_dim" : 16,
    "num_heads" : 4,
    "block_size" : 2,
    "attn_pdrop" : 0.1,
    "resid_pdrop" : 0.1,
    "vocab_size" : 100,
    "embd_pdrop" : 0.1,
    "n_layer" : 6,
    "device" : 'cuda',
    "weight_decay" : 0.1,
    "learning_rate" : 3e-4,
    "betas" : (0.9, 0.95),
    "grad_norm_clip" : 1.0,
    "batch_size" : 4
}

'''Load LLM model'''
model=GPT(config)
model = model.cuda()
PATH = "./regress_GPT.pt"
model.load_state_dict(torch.load(PATH))

'''LLM prediction evaluation'''
np.random.seed(32)
X1 = np.random.randint(1, 100, size=1000)
X2 = np.random.randint(1, 100, size=1000)
Y = np.log(X1) + np.sin(X2)

lst = []
lst1 = []
for x in X1:
    inp = torch.tensor([[x, 12]], dtype=torch.long).to('cuda')
    with torch.no_grad():
        cat = model.generate(inp, 1)
    result = cat[:,:].cpu().numpy()[0][0]
    lst.append([x, result])
    lst1.append([x, np.log(x) + np.sin(12)])

lst = sorted(lst, key=lambda x: x[0])
lst1 = sorted(lst1, key=lambda x: x[0])
plt.plot([x[0] for x in lst], [x[1] for x in lst])
plt.plot([x[0] for x in lst1], [x[1] for x in lst1])

'''LLM gradient evaluation'''
dct = defaultdict(list)
for x in range(1,100):
    inp = torch.tensor([[x, 1]], dtype=torch.long).to('cuda')
    y = torch.tensor([[0.]]).to('cuda')
    logits, loss = model(inp, y) # loss fn = (Y_pred - 0)^2
    loss.backward()
    np.random.seed(x)
    k = np.random.randint(0, 10000, size=config["embed_dim"])
    embed_grad = [param.grad for param in model.transformer.wte.parameters()][0]
    dct['X'].append(x)
    for idx, val in enumerate(k):
        dct[f"X_{idx+1}"].append(embed_grad[val].item()/(2*logits.item())) # since d(loss)/dx = (1/2Y) * dY/dx 
    model.zero_grad(set_to_none=True)

df = pd.DataFrame(dct)
lst = []
for i in range(df.shape[0]):
    lst.append([df.iloc[i,0],np.abs(df.iloc[i, 1:]).sum()])
lst = sorted(lst, key=lambda x: x[0])
fig, ax = plt.subplots() 
ax.plot([x[0] for x in lst], [x[1] for x in lst], color='#1d1f1e', label = 'Predicted Gradient', linewidth=0.8)
bbox = dict(boxstyle ="round", fc ="0.8")
ax.annotate('Continuously decreasing function', xy =(60, 25), 
                xytext =(40, 55),  
                arrowprops = dict(facecolor ='green', 
                                  shrink = 0.05),
                                  bbox = bbox)
plt.legend()
plt.title("Gradient of Y with respect to X1")
plt.xlabel("X1")
plt.ylabel(r'$\frac{\partial Y}{\partial X_1}$', fontsize=16)
plt.show()